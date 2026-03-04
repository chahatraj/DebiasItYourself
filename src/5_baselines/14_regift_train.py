"""
ReGiFT: Reasoning-Guided Fine-Tuning for Bias Mitigation
Paper: "Reasoning Towards Fairness: Mitigating Bias in Language Models through Reasoning-Guided Fine-Tuning"
arXiv:2504.05632v3

This script implements:
1. Reasoning trace extraction from an advanced reasoning model
2. Fine-tuning a target model on correct reasoning traces
"""

import os
import json
import random
import re
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser(description="ReGiFT Training Script")
    parser.add_argument("--reasoning_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model for reasoning trace extraction")
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to fine-tune")
    parser.add_argument("--num_traces", type=int, default=500,
                        help="Number of examples for trace extraction")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, default="/scratch/craj/diy/outputs/3_baselines/regift/models",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--traces_file", type=str, default="/scratch/craj/diy/outputs/3_baselines/regift/models/reasoning_traces_correct.json",
                        help="File to save/load reasoning traces")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True,
                        help="Use 4-bit quantization (pass --no-use_4bit to disable)")
    return parser.parse_args()


def get_quantization_config():
    """Get 4-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )


def check_answer_match(predicted, gold):
    """Check if predicted answer matches gold answer"""
    pred_lower = predicted.lower().strip()
    gold_lower = gold.lower().strip()
    
    unknown_variants = [
        "unknown", "not in background", "cannot be determined",
        "can't be determined", "not answerable", "not known",
        "not enough info", "not enough information"
    ]
    
    pred_is_unknown = any(v in pred_lower for v in unknown_variants)
    gold_is_unknown = gold_lower == "not in background" or len(gold_lower) == 0
    
    if pred_is_unknown and gold_is_unknown:
        return True
    if gold_lower in pred_lower or pred_lower in gold_lower:
        return True
    return False


def extract_reasoning_traces(model, tokenizer, dataset, num_examples=500):
    """
    Extract reasoning traces from SQuAD-v2 using reasoning model.
    Only keeps traces where the model's answer is correct.
    
    Args:
        model: Reasoning model (e.g., DeepSeek-R1)
        tokenizer: Model tokenizer
        dataset: SQuAD-v2 dataset
        num_examples: Number of examples to process
    
    Returns:
        List of correct reasoning traces
    """
    model.eval()
    traces = []
    
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    
    prompt_template = """You must first reason step-by-step inside <think></think> tags, then provide your final answer inside <answer></answer> tags.

Context: {context}
Question: {question}

Start with <think>"""

    print(f"Extracting reasoning traces from {num_examples} examples...")
    
    for idx in tqdm(indices, desc="Extracting traces"):
        example = dataset[idx]
        context = example['context']
        question = example['question']
        answers = example['answers']
        
        gold_answer = answers['text'][0] if len(answers['text']) > 0 else "Not in background"
        
        prompt = prompt_template.format(context=context, question=question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse reasoning and answer
        reasoning = None
        predicted_answer = None
        
        if '</think>' in generated:
            parts = generated.split('</think>', 1)
            reasoning = parts[0].strip()
            reasoning = re.sub(r'^<think>\s*', '', reasoning)
            
            after_think = parts[1] if len(parts) > 1 else ""
            
            answer_match = re.search(r'<answer>(.*?)</answer>', after_think, re.DOTALL)
            if answer_match:
                predicted_answer = answer_match.group(1).strip()
            else:
                clean_text = re.sub(r'^[\s\n:]*', '', after_think)
                if clean_text:
                    first_line = clean_text.split('\n')[0].strip()
                    if first_line:
                        predicted_answer = first_line
        
        # Only keep correct traces with sufficient reasoning
        if reasoning and len(reasoning) > 50 and predicted_answer and len(predicted_answer) > 2:
            is_correct = check_answer_match(predicted_answer, gold_answer)
            if is_correct:
                traces.append({
                    'context': context,
                    'question': question,
                    'gold_answer': gold_answer,
                    'reasoning': reasoning,
                    'predicted_answer': predicted_answer
                })
    
    print(f"Extracted {len(traces)} correct traces from {num_examples} examples")
    return traces


class ReasoningDataset(Dataset):
    """Dataset for reasoning-guided fine-tuning"""
    
    def __init__(self, traces, tokenizer, max_length=512):
        self.traces = traces
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        
        prompt = f"""Context: {trace['context']}
Question: {trace['question']}
Answer with reasoning:"""
        
        target = f"<think>{trace['reasoning']}</think><answer>{trace['predicted_answer']}</answer>"
        
        full_text = prompt + " " + target
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


def train(args):
    """Main training function"""
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    
    print("="*60)
    print("ReGiFT: Reasoning-Guided Fine-Tuning")
    print("="*60)
    
    quant_config = get_quantization_config() if args.use_4bit else None
    
    # Step 1: Extract reasoning traces (or load from file)
    if os.path.exists(args.traces_file):
        print(f"\nLoading existing traces from {args.traces_file}")
        with open(args.traces_file, "r") as f:
            correct_traces = json.load(f)
        print(f"Loaded {len(correct_traces)} correct traces")
    else:
        print("\nStep 1: Loading reasoning model...")
        reasoning_tokenizer = AutoTokenizer.from_pretrained(args.reasoning_model)
        reasoning_tokenizer.pad_token = reasoning_tokenizer.eos_token
        reasoning_tokenizer.padding_side = "left"
        
        reasoning_model = AutoModelForCausalLM.from_pretrained(
            args.reasoning_model,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print(f"✓ Loaded {args.reasoning_model}")
        
        # Load SQuAD-v2
        print("\nLoading SQuAD-v2...")
        squad_dataset = load_dataset("squad_v2", split="train")
        print(f"✓ Loaded {len(squad_dataset)} examples")
        
        # Extract traces
        print("\nStep 2: Extracting reasoning traces...")
        correct_traces = extract_reasoning_traces(
            reasoning_model, reasoning_tokenizer, squad_dataset, args.num_traces
        )
        
        # Save traces
        traces_dir = os.path.dirname(args.traces_file)
        if traces_dir:
            os.makedirs(traces_dir, exist_ok=True)
        with open(args.traces_file, "w") as f:
            json.dump(correct_traces, f, indent=2)
        print(f"✓ Saved {len(correct_traces)} traces to {args.traces_file}")
        
        # Free memory
        del reasoning_model
        torch.cuda.empty_cache()
    
    # Step 2: Fine-tune target model
    print(f"\nStep 3: Loading target model {args.target_model}...")
    
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_tokenizer.pad_token = target_tokenizer.eos_token
    target_tokenizer.padding_side = "right"
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare for training with LoRA
    target_model = prepare_model_for_kbit_training(target_model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    target_model = get_peft_model(target_model, lora_config)
    print("✓ Model prepared with LoRA")
    target_model.print_trainable_parameters()
    
    # Create dataset
    train_dataset = ReasoningDataset(correct_traces, target_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"✓ Dataset ready: {len(train_dataset)} examples")
    
    # Training
    optimizer = AdamW(target_model.parameters(), lr=args.learning_rate)
    target_model.train()
    
    print(f"\nStep 4: Fine-tuning for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            input_ids = batch["input_ids"].to(target_model.device)
            attention_mask = batch["attention_mask"].to(target_model.device)
            labels = batch["labels"].to(target_model.device)
            
            outputs = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    target_model.save_pretrained(args.output_dir)
    target_tokenizer.save_pretrained(args.output_dir)
    print(f"\n✓ Model saved to {args.output_dir}")
    
    print("\n" + "="*60)
    print("ReGiFT Training Complete!")
    print("="*60)


if __name__ == "__main__":
    args = parse_args()
    train(args)
