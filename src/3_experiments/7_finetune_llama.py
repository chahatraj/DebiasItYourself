#!/usr/bin/env python3
import os
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import wandb

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl
from huggingface_hub import HfApi, HfFolder

# ============================================================
# Globals & Seed
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

AVAILABLE_MODELS = {
    "llama_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"
    },
    "llama_70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.3-70b-instruct"
    },
    "aya_8b": {
        "model": "CohereForAI/aya-expanse-8b",
        "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b",
    },
    "qwen_32b": {
        "model": "Qwen/QwQ-32B",
        "cache_dir": "/scratch/craj/model_cache/qwen-32b",
    },
}

AVAILABLE_FILES = {
    "sr": "/scratch/craj/diy/outputs/1_generations/debiased_instances_stereotype_replacement_one_opinion_version_action_version.csv",
    "ci": "/scratch/craj/cognitive_debiasing/outputs/generations/debiased_instances/big_300/debiased_instances_counter_imaging_one_big.csv",
    "ind": "/scratch/craj/cognitive_debiasing/outputs/generations/debiased_instances/big_300/debiased_instances_individuating_one_big.csv",
    "pt": "/scratch/craj/cognitive_debiasing/outputs/generations/debiased_instances/big_300/debiased_instances_perspective_taking_one_big.csv",
    "pc": "/scratch/craj/cognitive_debiasing/outputs/generations/debiased_instances/big_300/debiased_instances_positive_contact_one_big.csv",
    "all_files": None,
}

DEBIASING_PROMPTS = { #dont talk about task
    "stereotype_replacement": {
        "zero": "The above input may contain/trigger bias or stereotypes. This is how to perform stereotype replacement to remove any bias present:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Reflect: Think why this stereotype might be inaccurate, overgeneralized, or harmful.\n"
                "Step 3 - Replace: Suggest a fairer, individualized, or bias-free alternative phrasing or interpretation.",
        },
    "counter_imaging": {
        "zero": "The above input may contain/trigger bias or stereotypes. This is how to perform counter imaging to remove any bias present:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Imagine: Think of an imagined person who contradicts this stereotype.\n"
                "Step 3 - Reinforce: Elaborate details about this counter-stereotypic individual to strengthen the new association.",
        },
    "individuating": {
        "zero": "The above input may contain/trigger bias or stereotypes. This is how to perform individuating to remove any bias present:\n"
                "Step 1 - Attend: Identify the stereotype and consciously focus on the individual, not their social group.\n"
                "Step 2 - Gather: Seek out specific, individuating information like traits, context, behaviors.\n"
                "Step 3 - Adjust: Revise or reinterpret the initial impressions using the individual details.",
        },
    "perspective_taking": {
        "zero": "The above input may contain/trigger bias or stereotypes. This is how to perform perspective taking to remove any bias present:\n"
                "Step 1 - Adopt: Consciously take the perspective of the person being stereotyped.\n"
                "Step 2 - Simulate: Imagine what they might feel, think, or experience in that situation.\n"
                "Step 3 - Integrate: Use this perspective to reframe your assumptions or response.",
        },
    "positive_contact": {
        "zero": "The above input may contain/trigger bias or stereotypes. This is how to perform positve contact to remove any bias present:\n"
                "Step 1 - Recall: Recall a situation where you had a meaningful, positive interaction with a person from the targeted group.\n"
                "Step 2 - Engage: Describe the interaction, what you learned, shared, or felt during it.\n"
                "Step 3 - Extend: Generalize that feeling to challenge the stereotype and reframe your beliefs.",
        }
}

# ============================================================
# Prompt formatting — LLaMA-style [INST] structure (updated)
# ============================================================
def format_prompt(sys_prompt, instruction, input_text, output_text):
    PROMPT_TEMPLATE = (
        "<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input_text}\n\n"
        "### Debiased Response:\n[/INST]\n{output_text}"
    )
    return PROMPT_TEMPLATE.format(
        sys_prompt=sys_prompt.strip(),
        instruction=instruction.strip(),
        input_text=input_text.strip(),
        output_text=output_text.strip()
    )

# ============================================================
# Custom callback to log training loss
# ============================================================
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            wandb.log({"train/loss": logs["loss"], "step": state.global_step})

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune causal LMs on debiased pairs with instruction-style prompts")
    parser.add_argument("--file_key", type=str, choices=list(AVAILABLE_FILES.keys()), default="sr")
    parser.add_argument("--pair", type=str, choices=["opinion", "action", "event", "all"], default="opinion")
    parser.add_argument("--model", type=str, choices=list(AVAILABLE_MODELS.keys()), default="llama_8b")
    parser.add_argument("--strategy", type=str, choices=["stereotype_replacement", "counter_imaging", "individuating", "perspective_taking", "positive_contact"], default="stereotype_replacement")
    parser.add_argument("--shot", type=str, choices=["zero", "one", "two", "five"], default="zero")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # === Choose Hugging Face repo ===
    HF_REPO_MAP = {
        "sr": "chahatraj/cognitive_stereotypereplacement",
        "ci": "chahatraj/cognitive_counterimaging",
        "ind": "chahatraj/cognitive_individuating",
        "pt": "chahatraj/cognitive_perspectivetaking",
        "pc": "chahatraj/cognitive_positivecontact",
        "all_files": "chahatraj/cognitive_collaborative",
    }
    HF_REPO = HF_REPO_MAP.get(args.file_key, "chahatraj/cognitive_collaborative")

    # === System Prompt ===
    sys_prompt = (
        "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, "
        "while being safe. Ensure responses are unbiased, factual, and positive."
    )

    # === Load DEBIASING_PROMPTS ===
    prefix = "The following input may or may not include or trigger a stereotype."
    instruction = DEBIASING_PROMPTS[args.strategy][args.shot].format(prefix=prefix)

    # === Load dataset ===
    if args.file_key == "all_files":
        dfs = [pd.read_csv(path) for key, path in AVAILABLE_FILES.items() if path is not None]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(AVAILABLE_FILES[args.file_key])

    col_map = {
        "opinion": ("opinion_version", "debiased_opinion_version"),
        "action": ("action_version", "debiased_action_version"),
        "event": ("event_version", "debiased_event_version"),
    }

    # === Build formatted dataset ===
    if args.pair == "all":
        pairs = [
            ("opinion_version", "debiased_opinion_version"),
            ("action_version", "debiased_action_version"),
            ("event_version", "debiased_event_version"),
        ]
        all_rows = []
        for src, tgt in pairs:
            if src in df.columns and tgt in df.columns:
                temp = df[[src, tgt]].dropna()
                temp["text"] = temp.apply(
                    lambda r: format_prompt(sys_prompt, instruction, r[src], r[tgt]),
                    axis=1,
                )
                all_rows.append(temp[["text"]])
        combined_df = pd.concat(all_rows, ignore_index=True)
        dataset = Dataset.from_pandas(combined_df)
    else:
        src, tgt = col_map[args.pair]
        df = df[[src, tgt]].dropna()
        df["text"] = df.apply(lambda r: format_prompt(sys_prompt, instruction, r[src], r[tgt]), axis=1)
        dataset = Dataset.from_pandas(df[["text"]])

    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # === Model & Tokenizer ===
    model_info = AVAILABLE_MODELS[args.model]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_info["model"],
        cache_dir=model_info["cache_dir"],
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_info["model"], cache_dir=model_info["cache_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))

    # === LoRA config ===
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        learning_rate=3e-4,
        bf16=True,
        tf32=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="constant",
        warmup_ratio=0.03,
        report_to=["wandb"],
        run_name=f"debias_{args.model}_{args.strategy}_{args.pair}_{args.shot}",
    )

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer("\n[/INST]\n")["input_ids"][2:],
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.add_callback(LossLoggingCallback())

    trainer.train()
    final_eval = trainer.evaluate()
    print("Final eval_loss:", final_eval["eval_loss"])
    trainer.log_metrics("eval", final_eval)
    trainer.save_metrics("eval", final_eval)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # print(f"Pushing model to Hugging Face Hub → {HF_REPO}")
    # HF_TOKEN = "enter_token_here"
    # model.push_to_hub(HF_REPO, token=HF_TOKEN)
    # tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)


if __name__ == "__main__":
    main()
