#!/usr/bin/env python3
"""
BBA: Bi-directional Bias Attribution for Debiasing LLMs
Paper: "Bi-directional Bias Attribution: Debiasing Large Language Models without Modifying Prompts"
arXiv:2602.04398
GitHub: https://github.com/XMUDeepLIT/Bi-directional-Bias-Attribution

This script performs the NEURON ATTRIBUTION phase (NO fine-tuning).
It identifies bias-related neurons in the final hidden layer (lm_head input)
via Integrated Gradients and saves their indices for inference-time intervention.

Algorithm (BBA Paper Section 3):
1. Stereotype Cue Selection (Section 3.1): entropy minimization over candidate words
   - Template: "The {attribute} of this {cue} person is"
   - Select cues with lowest entropy over demographic predictions
2. Forward-IG (Section 3.2): neurons predicting demographic from stereotype cues
   - F_fwd = -H(p(demographic | scaled_hidden_state))
3. Backward-IG (Section 3.3): neurons causing demographic-dependent stereotype word predictions
   - F_bwd = JSD proxy via KL(p_k || mean_p) over cue tokens across demographics
4. Select top-N = beta * hidden_dim neurons by combined |Forward-IG| + |Backward-IG| score
5. Save neuron_ids + constant C to JSON config (no model weights saved)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

BASE_DIR = Path(__file__).resolve().parent
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

DATASETS = ("bbq", "crowspairs", "stereoset")

# ============================================================
# Stereotype cue vocabulary and demographic groups per bias type
# Adapted from BBA paper Section 3.1 "Stereotype Cue Selection"
# ============================================================
BIAS_CONFIGS = {
    "gender": {
        "attribute": "gender",
        "demographics": ["male", "female", "man", "woman"],
        "cues": [
            "nurse", "doctor", "secretary", "engineer", "sensitive", "aggressive",
            "emotional", "rational", "nurturing", "assertive", "cheerleader", "mechanic",
            "librarian", "surgeon", "receptionist", "pilot", "babysitter", "programmer",
        ],
    },
    "race": {
        "attribute": "race",
        "demographics": ["Black", "White", "Asian", "Hispanic"],
        "cues": [
            "violent", "criminal", "lazy", "smart", "athletic", "poor",
            "dangerous", "hardworking", "educated", "loud", "submissive", "aggressive",
        ],
    },
    "religion": {
        "attribute": "religion",
        "demographics": ["Muslim", "Christian", "Jewish", "Hindu", "Buddhist"],
        "cues": [
            "extremist", "terrorist", "devout", "peaceful", "traditional",
            "charitable", "radical", "tolerant", "fundamentalist", "violent",
        ],
    },
    "age": {
        "attribute": "age group",
        "demographics": ["young", "old", "elderly", "teenage"],
        "cues": [
            "forgetful", "energetic", "irresponsible", "experienced",
            "incompetent", "innovative", "traditional", "naive", "wise",
        ],
    },
    "nationality": {
        "attribute": "nationality",
        "demographics": ["American", "Chinese", "Mexican", "African"],
        "cues": [
            "educated", "intelligent", "poor", "dangerous", "hardworking",
            "corrupt", "lazy", "skilled", "primitive", "sophisticated",
        ],
    },
    "sexual_orientation": {
        "attribute": "sexual orientation",
        "demographics": ["straight", "gay", "lesbian", "bisexual"],
        "cues": [
            "immoral", "confused", "flamboyant", "deviant",
            "theatrical", "promiscuous", "dangerous", "normal",
        ],
    },
    "disability": {
        "attribute": "disability status",
        "demographics": ["disabled", "able-bodied"],
        "cues": [
            "incapable", "dependent", "pitiful", "inspiring",
            "burden", "courageous", "unemployable", "productive",
        ],
    },
    "ses": {
        "attribute": "socioeconomic status",
        "demographics": ["wealthy", "poor", "working-class"],
        "cues": [
            "lazy", "greedy", "criminal", "hardworking",
            "irresponsible", "intelligent", "wasteful",
        ],
    },
}

DATASET_BIAS_CATEGORIES = {
    "bbq": ["gender", "race", "religion", "age", "nationality", "sexual_orientation", "disability", "ses"],
    "crowspairs": ["gender", "race", "religion", "age", "nationality", "sexual_orientation", "disability"],
    "stereoset": ["gender", "race", "religion", "nationality"],
}


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _get_lm_head_input(model, tokenizer, text: str, max_length: int = 64) -> torch.Tensor:
    """
    Get the lm_head input (= model.model.norm output) for the last token of text.

    For Llama: output_hidden_states[-1] is BEFORE final norm.
    We apply model.model.norm to get the true lm_head input.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0, -1, :]  # pre-norm, last token
        normed = model.model.norm(last_hidden.unsqueeze(0).unsqueeze(0)).squeeze().float()
    return normed.detach()


def select_stereotype_cues(
    model,
    tokenizer,
    bias_config: Dict,
    max_cues: int = 10,
    max_length: int = 64,
) -> List[str]:
    """
    Section 3.1: Stereotype Cue Selection via Entropy Minimization.

    Template: "The {attribute} of this {cue} person is"
    Select the cues with lowest entropy over demographic predictions (most biased).
    """
    attribute = bias_config["attribute"]
    demographics = bias_config["demographics"]
    cues = bias_config["cues"]

    demo_token_ids = []
    for d in demographics:
        toks = tokenizer.encode(" " + d, add_special_tokens=False)
        if toks:
            demo_token_ids.append(toks[0])

    if not demo_token_ids:
        return cues[:max_cues]

    demo_ids_tensor = torch.tensor(demo_token_ids, device=model.device)

    cue_entropies = []
    for cue in cues:
        prompt = f"The {attribute} of this {cue} person is"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        p_demo = probs[demo_ids_tensor]
        p_demo = p_demo / (p_demo.sum() + 1e-10)
        H = -(p_demo * torch.log(p_demo + 1e-10)).sum().item()
        cue_entropies.append((cue, H))

    cue_entropies.sort(key=lambda x: x[1])  # ascending entropy = most biased first
    return [c for c, _ in cue_entropies[:max_cues]]


def compute_forward_ig(
    model,
    tokenizer,
    attribute: str,
    cues: List[str],
    demographics: List[str],
    n_step: int = 50,
    max_length: int = 64,
) -> np.ndarray:
    """
    Section 3.2: Forward Bias Attribution via Integrated Gradients.

    Forward-IG(h_j) = h_bar_j * sum_{k=1}^{n_step} d[-H(p_demo | alpha_k*h_bar)] / d(h_j) * 1/n_step

    F_fwd = -H(p(demographic | h)) ... neurons making model certain about demographic.
    Gradient is computed through lm_head (linear layer) applied to scaled h_bar.
    """
    device = model.device
    W = model.lm_head.weight.data.float()  # [vocab_size, hidden_dim]
    b = model.lm_head.bias.data.float() if model.lm_head.bias is not None else None

    demo_token_ids = []
    for d in demographics:
        toks = tokenizer.encode(" " + d, add_special_tokens=False)
        if toks:
            demo_token_ids.append(toks[0])
    if not demo_token_ids:
        return np.zeros(W.shape[1])
    demo_ids_tensor = torch.tensor(demo_token_ids, device=device)

    hidden_dim = W.shape[1]
    ig_scores = torch.zeros(hidden_dim)
    count = 0

    for cue in tqdm(cues, desc="Forward-IG"):
        prompt = f"The {attribute} of this {cue} person is"
        h_bar = _get_lm_head_input(model, tokenizer, prompt, max_length).to(device)

        grad_sum = torch.zeros(hidden_dim, device=device)

        for k in range(1, n_step + 1):
            alpha = k / n_step
            h_k = (alpha * h_bar).clone().detach().requires_grad_(True)

            logits = h_k @ W.T.to(device)
            if b is not None:
                logits = logits + b.to(device)

            probs = torch.softmax(logits, dim=-1)
            p_demo = probs[demo_ids_tensor]
            p_demo_norm = p_demo / (p_demo.sum() + 1e-10)
            H = -(p_demo_norm * torch.log(p_demo_norm + 1e-10)).sum()
            F = -H  # negative entropy: maximize certainty about demographics

            F.backward()
            with torch.no_grad():
                grad_sum += h_k.grad.detach()

        ig = (h_bar * grad_sum / n_step).cpu()
        ig_scores += ig.abs()
        count += 1

    if count > 0:
        ig_scores /= count
    return ig_scores.numpy()


def compute_backward_ig(
    model,
    tokenizer,
    attribute: str,
    cues: List[str],
    demographics: List[str],
    n_step: int = 50,
    max_length: int = 64,
) -> np.ndarray:
    """
    Section 3.3: Backward Bias Attribution via Integrated Gradients.

    Backward-IG identifies neurons causing the model to predict stereotype words
    differently across demographic groups.

    F_bwd = KL(p_k || mean_p) over cue token positions (JSD proxy).
    Gradient computed through lm_head applied to scaled mean h_bar.
    """
    device = model.device
    W = model.lm_head.weight.data.float()  # [vocab_size, hidden_dim]
    b = model.lm_head.bias.data.float() if model.lm_head.bias is not None else None

    cue_token_ids = []
    for c in cues:
        toks = tokenizer.encode(" " + c, add_special_tokens=False)
        if toks:
            cue_token_ids.append(toks[0])
    if not cue_token_ids:
        return np.zeros(W.shape[1])
    cue_ids_tensor = torch.tensor(cue_token_ids, device=device)

    hidden_dim = W.shape[1]
    ig_scores = torch.zeros(hidden_dim)
    count = 0

    for cue in tqdm(cues, desc="Backward-IG"):
        # Get hidden states for each demographic
        h_bars = []
        for demo in demographics:
            prompt = f"The {attribute} of this {demo} is {cue}"
            h = _get_lm_head_input(model, tokenizer, prompt, max_length).to(device)
            h_bars.append(h)

        if len(h_bars) < 2:
            continue

        h_bar = torch.stack(h_bars).mean(0)  # mean across demographics

        grad_sum = torch.zeros(hidden_dim, device=device)

        for k in range(1, n_step + 1):
            alpha = k / n_step
            h_k = (alpha * h_bar).clone().detach().requires_grad_(True)

            # Distributions for each demographic at scaled activations
            all_probs = []
            for h_d in h_bars:
                h_scaled = (alpha * h_d).detach()
                logits_d = h_scaled @ W.T.to(device)
                if b is not None:
                    logits_d = logits_d + b.to(device)
                all_probs.append(torch.softmax(logits_d, dim=-1))

            mean_probs = torch.stack(all_probs).mean(0).detach()

            # KL(p_k || mean_probs) on cue token positions (differentiable w.r.t. h_k)
            logits_k = h_k @ W.T.to(device)
            if b is not None:
                logits_k = logits_k + b.to(device)
            probs_k = torch.softmax(logits_k, dim=-1)

            kl = (probs_k[cue_ids_tensor] * (
                torch.log(probs_k[cue_ids_tensor] + 1e-10) -
                torch.log(mean_probs[cue_ids_tensor] + 1e-10)
            )).sum()

            kl.backward()
            with torch.no_grad():
                grad_sum += h_k.grad.detach()

        ig = (h_bar * grad_sum / n_step).cpu()
        ig_scores += ig.abs()
        count += 1

    if count > 0:
        ig_scores /= count
    return ig_scores.numpy()


def select_top_neurons(
    forward_ig: np.ndarray,
    backward_ig: np.ndarray,
    beta: float = 0.01,
) -> np.ndarray:
    """
    Section 3.4: Select top-N neurons where N = beta * M (M = hidden_dim).
    Combined score = |Forward-IG| + |Backward-IG|.
    """
    combined = np.abs(forward_ig) + np.abs(backward_ig)
    M = len(combined)
    N = max(1, int(beta * M))
    top_indices = np.argsort(combined)[::-1][:N]
    return top_indices.astype(np.int64)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/bba/models_bbq",
            "model_tag": "bbq_all",
        },
        "crowspairs": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/bba/models_crowspairs",
            "model_tag": "crowspairs_all",
        },
        "stereoset": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/bba/models_stereoset",
            "model_tag": "stereoset_all",
        },
    }
    for k, v in defaults[args.dataset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BBA: Bi-directional Bias Attribution - Neuron Attribution (no fine-tuning)"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument(
        "--beta", type=float, default=0.01,
        help="Fraction of neurons to select: N = beta * hidden_dim (paper: grid search)"
    )
    parser.add_argument(
        "--constant_c", type=float, default=0.0,
        help="Constant C to fix selected neurons to at inference (paper: grid search, default 0.0)"
    )
    parser.add_argument(
        "--n_step", type=int, default=50,
        help="Number of IG approximation steps"
    )
    parser.add_argument(
        "--max_cues", type=int, default=10,
        help="Max stereotype cues per bias category (after entropy ranking)"
    )
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    _apply_defaults(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"BBA: Bi-directional Bias Attribution ({args.dataset})")
    print("Paper: arXiv:2602.04398")
    print("Method: Integrated Gradients for neuron attribution (NO fine-tuning)")
    print("=" * 60)

    quant_config = get_quantization_config() if args.use_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=args.hf_token,
    )
    model.eval()

    bias_categories = DATASET_BIAS_CATEGORIES.get(args.dataset, list(BIAS_CONFIGS.keys()))
    hidden_dim = model.config.hidden_size

    forward_ig_total = np.zeros(hidden_dim)
    backward_ig_total = np.zeros(hidden_dim)
    n_categories = 0

    for cat in bias_categories:
        if cat not in BIAS_CONFIGS:
            continue

        cfg = BIAS_CONFIGS[cat]
        print(f"\n[{cat.upper()}] Step 1: Stereotype cue selection...")
        selected_cues = select_stereotype_cues(model, tokenizer, cfg, max_cues=args.max_cues)
        print(f"  Selected cues: {selected_cues}")

        print(f"[{cat.upper()}] Step 2: Computing Forward-IG...")
        fwd_ig = compute_forward_ig(
            model, tokenizer,
            attribute=cfg["attribute"],
            cues=selected_cues,
            demographics=cfg["demographics"],
            n_step=args.n_step,
        )

        print(f"[{cat.upper()}] Step 3: Computing Backward-IG...")
        bwd_ig = compute_backward_ig(
            model, tokenizer,
            attribute=cfg["attribute"],
            cues=selected_cues,
            demographics=cfg["demographics"],
            n_step=args.n_step,
        )

        forward_ig_total += np.abs(fwd_ig)
        backward_ig_total += np.abs(bwd_ig)
        n_categories += 1

    if n_categories > 0:
        forward_ig_total /= n_categories
        backward_ig_total /= n_categories

    # Select top-N neurons by combined attribution score
    neuron_ids = select_top_neurons(forward_ig_total, backward_ig_total, beta=args.beta)
    print(f"\nStep 4: Selected {len(neuron_ids)} neurons (beta={args.beta}, hidden_dim={hidden_dim})")
    print(f"  Top-10 indices: {neuron_ids[:10].tolist()}")

    # Save config (no model weights saved - inference-time method only)
    config_path = os.path.join(args.output_dir, f"bba_config_{args.model_tag}.json")
    bba_config = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "model_tag": args.model_tag,
        "hidden_dim": int(hidden_dim),
        "beta": float(args.beta),
        "constant_c": float(args.constant_c),
        "n_step": int(args.n_step),
        "neuron_ids": neuron_ids.tolist(),
        "n_neurons": int(len(neuron_ids)),
        "bias_categories": bias_categories,
        "seed": int(args.seed),
    }

    with open(config_path, "w") as f:
        json.dump(bba_config, f, indent=2)
    print(f"\nSaved BBA config to: {config_path}")
    print("Run 8_bba_evaluate.py to apply inference-time neuron intervention.")


if __name__ == "__main__":
    main()
