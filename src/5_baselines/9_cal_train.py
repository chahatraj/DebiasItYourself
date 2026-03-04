#!/usr/bin/env python3
"""
CAL: Causal-Guided Active Learning for LLM Debiasing - Bias Pattern Induction
Paper: "Causal-Guided Active Learning for Debiasing Large Language Models"
arXiv:2412.12140

This script implements the bias pattern induction phase of CAL (Section 3):
- Extracts hidden representations from biased instances (causal invariance principle)
- PCA dimensionality reduction + DBSCAN clustering to discover recurring bias patterns
- LLM-based summarization of each cluster into natural language bias descriptions
- Saves patterns as JSON for use during ICL-based evaluation (9_cal_evaluate.py)

NO fine-tuning occurs. Output is a JSON config with prompt patterns.

Algorithm overview:
1. For each dataset, collect "bias representations":
   - BBQ: hidden states of ambiguous-context questions
   - CrowS-Pairs: difference vectors H(stereo) - H(anti) per pair
   - StereoSet: difference vectors H(stereo) - H(anti) per context
2. L2-normalize → PCA(n=50) → DBSCAN clustering
3. LLM summarizes each cluster → natural language bias description
4. Build zero-shot ICL prefix:
   "We should treat people from different [patterns] equally.
    When we lack information, choose 'unknown' rather than assuming based on stereotypes."
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


BASE_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None) -> ModuleType:
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dataset_common(dataset: str):
    shared_mod = _load_module(
        module_name="paper_baselines_shared_adapter_cal_train",
        module_path=BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    return shared_mod.get_dataset_common(dataset)


def extract_hidden_states_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract last-layer hidden state (after final norm) at last non-padding token.
    This gives the true lm_head input representation for each text.
    """
    all_states = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]  # [batch, seq, hidden_dim] pre-norm
            # Index last non-padding token per example
            mask = enc["attention_mask"]           # [batch, seq]
            last_idx = mask.sum(dim=1) - 1         # [batch]
            last_hidden = last_hidden[torch.arange(last_hidden.size(0)), last_idx, :]
            # Apply final norm (model.model.norm) to get true lm_head input
            normed = model.model.norm(last_hidden)
        all_states.append(normed.float().cpu().numpy())
    return np.concatenate(all_states, axis=0)


def find_bias_representations_bbq(
    model, tokenizer, df, batch_size: int = 8
) -> Tuple[np.ndarray, List[str]]:
    """
    BBQ: causal invariance identifies ambiguous-context questions as potential bias sites.
    Collect hidden states of these questions as bias representations.
    """
    if "context_condition" in df.columns:
        bias_df = df[df["context_condition"] == "ambig"]
    else:
        bias_df = df

    if len(bias_df) > 500:
        bias_df = bias_df.sample(500, random_state=42)

    bias_texts = [
        f"{row['context']} {row['question']}" for _, row in bias_df.iterrows()
    ]
    if not bias_texts:
        return np.array([]), []

    hidden_states = extract_hidden_states_batch(model, tokenizer, bias_texts, batch_size=batch_size)
    return hidden_states, bias_texts


def find_bias_representations_crowspairs(
    model, tokenizer, pair_df, batch_size: int = 8
) -> Tuple[np.ndarray, List[str]]:
    """
    CrowS-Pairs: bias vector = H(stereo) - H(anti) for each pair.

    Causal invariance principle: pairs with similar meaning but different stereotypicality
    expose the model's bias direction. The difference vector isolates that direction.
    """
    if "sent_more" in pair_df.columns:
        stereo_texts = pair_df["sent_more"].tolist()
        anti_texts = pair_df["sent_less"].tolist()
    elif "stereotype" in pair_df.columns:
        stereo_texts = pair_df["stereotype"].tolist()
        anti_texts = pair_df["anti"].tolist()
    else:
        raise ValueError("pair_df must have sent_more/sent_less or stereotype/anti columns")

    if len(stereo_texts) > 500:
        stereo_texts = stereo_texts[:500]
        anti_texts = anti_texts[:500]

    H_stereo = extract_hidden_states_batch(model, tokenizer, stereo_texts, batch_size=batch_size)
    H_anti = extract_hidden_states_batch(model, tokenizer, anti_texts, batch_size=batch_size)
    bias_vectors = H_stereo - H_anti
    return bias_vectors, stereo_texts


def find_bias_representations_stereoset(
    model, tokenizer, examples: List[Dict], batch_size: int = 8
) -> Tuple[np.ndarray, List[str]]:
    """
    StereoSet: bias vector = H(context+stereo) - H(context+anti) for each item.
    """
    stereo_texts: List[str] = []
    anti_texts: List[str] = []

    for ex in examples[:500]:
        context = ex.get("context", "")
        stereo_sent = None
        anti_sent = None
        for sent in ex.get("sentences", []):
            label = sent.get("gold_label", "")
            if label == "stereotype" and stereo_sent is None:
                stereo_sent = sent["sentence"]
            elif label == "anti-stereotype" and anti_sent is None:
                anti_sent = sent["sentence"]
        if stereo_sent and anti_sent:
            stereo_texts.append(f"{context} {stereo_sent}")
            anti_texts.append(f"{context} {anti_sent}")

    if not stereo_texts:
        return np.array([]), []

    H_stereo = extract_hidden_states_batch(model, tokenizer, stereo_texts, batch_size=batch_size)
    H_anti = extract_hidden_states_batch(model, tokenizer, anti_texts, batch_size=batch_size)
    bias_vectors = H_stereo - H_anti
    return bias_vectors, stereo_texts


def cluster_bias_representations(
    bias_vectors: np.ndarray,
    pca_components: int = 50,
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 3,
) -> np.ndarray:
    """
    CAL clustering pipeline (Section 3.2):
    1. L2-normalize bias vectors
    2. PCA to pca_components dimensions
    3. DBSCAN clustering with cosine metric

    Returns cluster labels (-1 = noise).
    """
    vectors_norm = normalize(bias_vectors, norm="l2")

    n_components = min(pca_components, vectors_norm.shape[0] - 1, vectors_norm.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(vectors_norm)
    reduced = normalize(reduced, norm="l2")

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine")
    labels = clustering.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points from {len(labels)} examples")
    return labels


def summarize_cluster_with_llm(
    model,
    tokenizer,
    cluster_texts: List[str],
    dataset: str,
    max_new_tokens: int = 80,
) -> str:
    """
    LLM-based pattern summarization: prompt the model with cluster examples
    and ask for a short description of the common bias pattern.
    """
    sample_texts = cluster_texts[:5]
    examples_str = "\n".join(f"- {t[:120]}" for t in sample_texts)

    if dataset == "bbq":
        prompt = (
            "The following questions exhibit demographic bias where stereotypes about "
            "social groups may influence the expected answer.\n\n"
            f"{examples_str}\n\n"
            "What common demographic bias pattern do these questions share? "
            "Describe the bias pattern in 10 words or less:"
        )
    else:
        prompt = (
            "The following sentences exhibit stereotypical bias about social groups.\n\n"
            f"{examples_str}\n\n"
            "What common stereotypical bias pattern do these sentences share? "
            "Describe the bias pattern in 10 words or less:"
        )

    enc = tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][enc["input_ids"].shape[1]:]
    description = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Take only first line
    description = description.split("\n")[0].strip().rstrip(".")
    return description if description else "demographic stereotypes and social group biases"


def induce_bias_patterns(
    model,
    tokenizer,
    bias_vectors: np.ndarray,
    bias_texts: List[str],
    dataset: str,
    max_patterns: int = 5,
    pca_components: int = 50,
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 3,
) -> List[Dict]:
    """
    Full CAL bias pattern induction:
    1. Cluster bias representations via PCA + DBSCAN
    2. LLM-summarize each cluster into a natural language bias description
    3. Return list of patterns sorted by cluster size (largest first)
    """
    if len(bias_vectors) == 0:
        print("No bias vectors found; using generic fallback pattern.")
        return [{
            "id": 0,
            "description": "demographic stereotypes and social group biases",
            "n_examples": 0,
            "sample_texts": [],
        }]

    labels = cluster_bias_representations(
        bias_vectors, pca_components, dbscan_eps, dbscan_min_samples
    )

    # Sort non-noise clusters by size (descending)
    unique_labels = [l for l in set(labels) if l != -1]
    unique_labels.sort(key=lambda l: int((labels == l).sum()), reverse=True)

    patterns = []
    for label in unique_labels[:max_patterns]:
        cluster_mask = labels == label
        cluster_texts = [t for t, m in zip(bias_texts, cluster_mask) if m]
        cluster_size = int(cluster_mask.sum())

        print(f"Cluster {label}: {cluster_size} examples. Summarizing...")
        description = summarize_cluster_with_llm(model, tokenizer, cluster_texts, dataset)
        print(f"  Pattern: {description}")

        patterns.append({
            "id": int(label),
            "description": description,
            "n_examples": cluster_size,
            "sample_texts": cluster_texts[:3],
        })

    if not patterns:
        print("No clusters found; using generic fallback pattern.")
        patterns = [{
            "id": 0,
            "description": "demographic stereotypes and social group biases",
            "n_examples": len(bias_texts),
            "sample_texts": bias_texts[:3],
        }]

    return patterns


def build_cal_prompt_prefix(patterns: List[Dict]) -> str:
    """
    CAL zero-shot ICL prefix (Section 3.3):
    "We should treat people from different [induced pattern] equally.
     When we lack information, choose 'unknown' rather than assuming based on stereotypes."
    """
    descriptions = [p["description"] for p in patterns if p.get("description")]
    if descriptions:
        combined = ", ".join(descriptions)
    else:
        combined = "social groups and demographic backgrounds"

    return (
        f"We should treat people from different {combined} equally. "
        "When we lack information, choose 'unknown' rather than assuming based on stereotypes."
    )


# ─── Dataset-specific entry points ────────────────────────────────────────────

def _train_bbq(args, mod, model, tokenizer) -> List[Dict]:
    df = mod.load_bbq_df(
        args.bbq_dir, args.meta_file,
        category=args.category,
        limit_per_category=args.limit_per_category,
    )
    bias_vectors, bias_texts = find_bias_representations_bbq(
        model, tokenizer, df, batch_size=args.batch_size
    )
    return induce_bias_patterns(
        model, tokenizer, bias_vectors, bias_texts, "bbq",
        max_patterns=args.max_patterns,
        pca_components=args.pca_components,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )


def _train_crowspairs(args, mod, model, tokenizer) -> List[Dict]:
    df = mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = mod.make_pair_records(df)
    bias_vectors, bias_texts = find_bias_representations_crowspairs(
        model, tokenizer, pair_df, batch_size=args.batch_size
    )
    return induce_bias_patterns(
        model, tokenizer, bias_vectors, bias_texts, "crowspairs",
        max_patterns=args.max_patterns,
        pca_components=args.pca_components,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )


def _train_stereoset(args, mod, model, tokenizer) -> List[Dict]:
    examples = mod.load_examples(
        args.data_path, split=args.split, bias_type=args.bias_type, limit=args.limit
    )
    bias_vectors, bias_texts = find_bias_representations_stereoset(
        model, tokenizer, examples, batch_size=args.batch_size
    )
    return induce_bias_patterns(
        model, tokenizer, bias_vectors, bias_texts, "stereoset",
        max_patterns=args.max_patterns,
        pca_components=args.pca_components,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )


# ─── Defaults, argparse, main ─────────────────────────────────────────────────

def _apply_defaults(args: argparse.Namespace) -> None:
    defaults: Dict[str, Dict[str, object]] = {
        "bbq": {
            "bbq_dir": "/scratch/craj/diy/data/BBQ/data",
            "meta_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "batch_size": 8,
            "model_tag": "bbq_all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/cal/models_bbq",
        },
        "crowspairs": {
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "batch_size": 4,
            "model_tag": "crowspairs_all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/cal/models_crowspairs",
        },
        "stereoset": {
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "batch_size": 4,
            "model_tag": "stereoset_all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/cal/models_stereoset",
        },
    }
    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAL: Bias Pattern Induction via Causal Invariance + PCA/DBSCAN + LLM Summarization"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--bbq_dir", type=str, default=None)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--max_patterns", type=int, default=5,
                        help="Maximum number of bias patterns to induce (default: 5)")
    parser.add_argument("--pca_components", type=int, default=50,
                        help="PCA output dimensions for bias representation (default: 50)")
    parser.add_argument("--dbscan_eps", type=float, default=0.3,
                        help="DBSCAN epsilon (cosine metric, default: 0.3)")
    parser.add_argument("--dbscan_min_samples", type=int, default=3,
                        help="DBSCAN min_samples (default: 3)")

    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    set_seed(42)
    args = parse_args()
    _apply_defaults(args)
    mod = _load_dataset_common(args.dataset)

    print("=" * 60)
    print(f"CAL: Bias Pattern Induction ({args.dataset})")
    print("=" * 60)

    # Load base model - CAL is inference-time only, no fine-tuning
    model, tokenizer = mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=None,
        base_model=mod.BASE_MODEL,
    )

    if args.dataset == "bbq":
        patterns = _train_bbq(args, mod, model, tokenizer)
    elif args.dataset == "crowspairs":
        patterns = _train_crowspairs(args, mod, model, tokenizer)
    else:
        patterns = _train_stereoset(args, mod, model, tokenizer)

    prompt_prefix = build_cal_prompt_prefix(patterns)
    print(f"\nCAL zero-shot prefix:\n{prompt_prefix}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "n_patterns": len(patterns),
        "patterns": patterns,
        "prompt_prefix": prompt_prefix,
        "pca_components": args.pca_components,
        "dbscan_eps": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
    }
    config_path = os.path.join(args.output_dir, f"cal_config_{args.model_tag}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved CAL config with {len(patterns)} patterns to {config_path}")


if __name__ == "__main__":
    main()
