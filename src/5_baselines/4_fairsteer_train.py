#!/usr/bin/env python3
"""Dataset-agnostic FairSteer training entrypoint (BAD classifiers + DSV)."""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("bbq", "crowspairs", "stereoset")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None):
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _safe_str(x) -> str:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def _resolve_model_name(model_name: Optional[str], model_alias: Optional[str]) -> str:
    if model_name:
        return model_name
    if model_alias in (None, "llama_8b"):
        return "meta-llama/Llama-3.1-8B-Instruct"
    raise ValueError(f"Unsupported --model alias: {model_alias}")


def _load_target_map(metadata_file: Optional[str]) -> Dict[Tuple[str, int], int]:
    if not metadata_file or not os.path.exists(metadata_file):
        return {}

    meta_df = pd.read_csv(metadata_file)
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]
    needed = {"example_id", "target_loc", "category"}
    if not needed.issubset(set(meta_df.columns)):
        return {}

    target_map: Dict[Tuple[str, int], int] = {}
    for _, row in meta_df.iterrows():
        try:
            cat = str(row["category"]).replace(".jsonl", "")
            ex_id = int(row["example_id"])
            tloc = int(row["target_loc"])
            target_map[(cat, ex_id)] = tloc
        except Exception:
            continue
    return target_map


def load_bbq_data(data_dir: str, categories: List[str], max_per_category=300, metadata_file: Optional[str] = None) -> List[Dict]:
    target_map = _load_target_map(metadata_file)
    all_data: List[Dict] = []

    for cat in categories:
        fp = os.path.join(data_dir, f"{cat}.jsonl")
        if not os.path.exists(fp):
            print(f"[warn] Missing BBQ file: {fp}")
            continue

        with open(fp, "r", encoding="utf-8") as f:
            cat_data = [json.loads(line) for line in f if line.strip()]

        cat_data = [d for d in cat_data if d.get("context_condition") == "ambig"]
        if len(cat_data) > max_per_category:
            cat_data = random.sample(cat_data, max_per_category)

        for d in cat_data:
            d["category"] = cat
            ex_id = d.get("example_id")
            if ex_id is not None:
                try:
                    d["target_loc"] = target_map.get((cat, int(ex_id)))
                except Exception:
                    d["target_loc"] = None

        all_data.extend(cat_data)

    return all_data


def identify_crows_stereotype(row: pd.Series) -> Tuple[str, str, str]:
    sent_more = _safe_str(row.get("sent_more", "")).strip()
    sent_less = _safe_str(row.get("sent_less", "")).strip()
    direction = _safe_str(row.get("stereo_antistereo", "")).strip().lower()

    if direction == "stereo":
        return sent_more, sent_less, "stereo"
    if direction == "antistereo":
        return sent_less, sent_more, "antistereo"

    return sent_more, sent_less, "unknown"


def load_crows_data(data_path: str, bias_type: Optional[str] = None) -> List[Dict]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CrowS-Pairs file not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"sent_more", "sent_less"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CrowS-Pairs data: {sorted(missing)}")

    if bias_type is not None:
        if "bias_type" not in df.columns:
            raise ValueError("--bias_type provided but dataset has no 'bias_type' column")
        df = df[df["bias_type"].astype(str) == bias_type].copy()

    out = []
    for _, row in df.iterrows():
        stereo, anti, direction = identify_crows_stereotype(row)
        if not stereo or not anti:
            continue
        out.append(
            {
                "stereotype": stereo,
                "anti": anti,
                "direction": direction,
                "bias_type": _safe_str(row.get("bias_type", "")),
            }
        )

    return out


def load_stereoset_pairs(data_path: str, split: str, bias_type: Optional[str], limit: Optional[int]) -> List[Dict]:
    shared_mod = _load_module(
        "paper_baselines_shared_for_fairsteer_train_stereo",
        BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    common_mod = shared_mod.get_dataset_common("stereoset")

    data = common_mod.load_stereoset_data(data_path)
    examples = common_mod.flatten_examples(data, split=split, bias_type=bias_type, limit=limit)
    if not examples:
        raise ValueError("No StereoSet examples after filtering")

    return [{"stereotype": e["stereotype"], "anti": e["anti"], "split": e.get("split", ""), "bias_type": e.get("bias_type", "")} for e in examples]


def get_last_token_activations(model, tokenizer, text: str, max_length: int):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return [layer_hidden[0, -1, :].detach().cpu().numpy() for layer_hidden in outputs.hidden_states[1:]]


def get_sequence_logprob(model, tokenizer, text: str, max_length: int) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = enc["input_ids"][:, 1:]
    shift_mask = enc["attention_mask"][:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * shift_mask
    return float(token_log_probs.sum().item())


def collect_activations_by_model_output_bbq(model, tokenizer, data: List[Dict], max_examples: int = 800):
    num_layers = model.config.num_hidden_layers
    biased_acts = {l: [] for l in range(num_layers)}
    unbiased_acts = {l: [] for l in range(num_layers)}

    option_a = tokenizer.encode("A", add_special_tokens=False)[0]
    option_b = tokenizer.encode("B", add_special_tokens=False)[0]
    option_c = tokenizer.encode("C", add_special_tokens=False)[0]

    examples = data[:max_examples] if len(data) > max_examples else data

    for ex in tqdm(examples, desc="Collecting activations"):
        prompt = (
            f"{ex['context']}\n"
            f"Question: {ex['question']}\n"
            f"A. {ex['ans0']}\n"
            f"B. {ex['ans1']}\n"
            f"C. {ex['ans2']}\n"
            "Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits[[option_a, option_b, option_c]], dim=0)
            pred = int(torch.argmax(probs).item())

        activations = [layer_hidden[0, -1, :].detach().cpu().numpy() for layer_hidden in outputs.hidden_states[1:]]

        label = int(ex["label"])
        target_loc = ex.get("target_loc")
        is_biased = (pred == int(target_loc)) if target_loc is not None else (pred != label)

        for l, act in enumerate(activations):
            if is_biased:
                biased_acts[l].append(act)
            else:
                unbiased_acts[l].append(act)

    return biased_acts, unbiased_acts


def collect_activations_by_model_output_pairs(model, tokenizer, data: List[Dict], max_examples: int = 800, max_length: int = 256):
    num_layers = model.config.num_hidden_layers
    biased_acts = {l: [] for l in range(num_layers)}
    unbiased_acts = {l: [] for l in range(num_layers)}

    examples = data[:max_examples] if len(data) > max_examples else data
    for ex in tqdm(examples, desc="Collecting activations"):
        stereo = ex["stereotype"]
        anti = ex["anti"]

        stereo_lp = get_sequence_logprob(model, tokenizer, stereo, max_length=max_length)
        anti_lp = get_sequence_logprob(model, tokenizer, anti, max_length=max_length)
        activations = get_last_token_activations(model, tokenizer, anti, max_length=max_length)

        is_biased = stereo_lp > anti_lp
        for l, act in enumerate(activations):
            if is_biased:
                biased_acts[l].append(act)
            else:
                unbiased_acts[l].append(act)

    return biased_acts, unbiased_acts


def train_bad_classifiers(biased_acts, unbiased_acts, seed: int = 42):
    num_layers = len(biased_acts)
    classifiers = {}
    layer_accs = []

    for layer in tqdm(range(num_layers), desc="Training BAD classifiers"):
        x_b = np.array(biased_acts[layer])
        x_u = np.array(unbiased_acts[layer])

        min_n = min(len(x_b), len(x_u))
        if min_n == 0:
            classifiers[layer] = None
            layer_accs.append(0.5)
            continue

        x_b = x_b[:min_n]
        x_u = x_u[:min_n]

        x = np.vstack([x_b, x_u])
        y = np.array([0] * len(x_b) + [1] * len(x_u))

        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]

        split = int(0.8 * len(x))
        x_train, x_val = x[:split], x[split:]
        y_train, y_val = y[:split], y[split:]

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(x_train, y_train)

        val_acc = accuracy_score(y_val, clf.predict(x_val)) if len(x_val) > 0 else 0.5
        classifiers[layer] = clf
        layer_accs.append(float(val_acc))

    return classifiers, layer_accs


def compute_dsv_bbq(model, tokenizer, data: List[Dict], layer: int, num_pairs: int = 110):
    sampled = random.sample(data, min(num_pairs, len(data)))
    diffs = []

    for ex in tqdm(sampled, desc="Computing DSV"):
        target_loc = ex.get("target_loc")
        if target_loc is None:
            continue

        label = int(ex["label"])
        target_loc = int(target_loc)
        prompt = (
            f"{ex['context']}\n"
            f"Question: {ex['question']}\n"
            f"A. {ex['ans0']}\n"
            f"B. {ex['ans1']}\n"
            f"C. {ex['ans2']}\n"
            "Answer:"
        )

        biased_prompt = prompt + " " + ["A", "B", "C"][target_loc]
        unbiased_prompt = prompt + " " + ["A", "B", "C"][label]

        with torch.no_grad():
            in_b = tokenizer(biased_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            in_u = tokenizer(unbiased_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            out_b = model(**in_b, output_hidden_states=True)
            out_u = model(**in_u, output_hidden_states=True)

        act_b = out_b.hidden_states[layer + 1][0, -1, :].detach().cpu().numpy()
        act_u = out_u.hidden_states[layer + 1][0, -1, :].detach().cpu().numpy()
        diffs.append(act_u - act_b)

    if not diffs:
        raise ValueError("No valid samples found for BBQ DSV computation; target_loc missing/alignment issue.")
    return np.mean(diffs, axis=0)


def compute_dsv_pairs(model, tokenizer, data: List[Dict], layer: int, num_pairs: int = 110, max_length: int = 256):
    sampled = random.sample(data, min(num_pairs, len(data)))
    diffs = []

    for ex in tqdm(sampled, desc="Computing DSV"):
        stereo = ex["stereotype"]
        anti = ex["anti"]

        with torch.no_grad():
            in_s = tokenizer(stereo, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            in_a = tokenizer(anti, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            out_s = model(**in_s, output_hidden_states=True)
            out_a = model(**in_a, output_hidden_states=True)

        act_s = out_s.hidden_states[layer + 1][0, -1, :].detach().cpu().numpy()
        act_a = out_a.hidden_states[layer + 1][0, -1, :].detach().cpu().numpy()
        diffs.append(act_a - act_s)

    if not diffs:
        raise ValueError("No valid samples found for pairwise DSV computation.")
    return np.mean(diffs, axis=0)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "bbq_data_dir": "/scratch/craj/diy/data/BBQ/data",
            "metadata_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/fairsteer/models",
            "num_examples_bad": 800,
            "num_pairs_dsv": 110,
        },
        "crowspairs": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/fairsteer/models_crowspairs",
            "model_tag": "crowspairs_all",
            "num_examples_bad": 800,
            "num_pairs_dsv": 110,
        },
        "stereoset": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/fairsteer/models_stereoset",
            "model_tag": "stereoset_all",
            "num_examples_bad": 800,
            "num_pairs_dsv": 110,
        },
    }

    for k, v in defaults[args.dataset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FairSteer training across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--num_examples_bad", type=int, default=None)
    parser.add_argument("--num_pairs_dsv", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--bbq_data_dir", type=str, default=None)
    parser.add_argument("--metadata_file", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_defaults(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    model_name = _resolve_model_name(args.model_name, args.model)

    print("=" * 60)
    print(f"FairSteer Training ({args.dataset})")
    print("=" * 60)

    quant_config = get_quantization_config() if args.use_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        output_hidden_states=True,
        token=args.hf_token,
    )

    if args.dataset == "bbq":
        categories = [
            "Age",
            "Disability_status",
            "Gender_identity",
            "Nationality",
            "Physical_appearance",
            "Race_ethnicity",
            "Religion",
            "SES",
            "Sexual_orientation",
        ]
        data = load_bbq_data(
            args.bbq_data_dir,
            categories,
            metadata_file=args.metadata_file,
        )
        print(f"Loaded {len(data)} BBQ examples")

        biased_acts, unbiased_acts = collect_activations_by_model_output_bbq(
            model, tokenizer, data, max_examples=args.num_examples_bad
        )
        classifiers, layer_accs = train_bad_classifiers(biased_acts, unbiased_acts, seed=args.seed)
        optimal_layer = int(np.argmax(layer_accs))
        dsv = compute_dsv_bbq(model, tokenizer, data, optimal_layer, num_pairs=args.num_pairs_dsv)

        if args.model_tag:
            out_dir = os.path.join(args.output_dir, f"model_{args.model_tag}")
            cfg_path = os.path.join(out_dir, "config.json")
        else:
            out_dir = args.output_dir
            cfg_path = os.path.join(out_dir, "config.json")

    elif args.dataset == "crowspairs":
        data = load_crows_data(args.data_path, bias_type=args.bias_type)
        if args.limit:
            data = data[: args.limit]
        print(f"Loaded {len(data)} CrowS-Pairs pairs")

        biased_acts, unbiased_acts = collect_activations_by_model_output_pairs(
            model, tokenizer, data, max_examples=args.num_examples_bad, max_length=256
        )
        classifiers, layer_accs = train_bad_classifiers(biased_acts, unbiased_acts, seed=args.seed)
        optimal_layer = int(np.argmax(layer_accs))
        dsv = compute_dsv_pairs(model, tokenizer, data, optimal_layer, num_pairs=args.num_pairs_dsv, max_length=256)

        out_dir = os.path.join(args.output_dir, f"model_{args.model_tag}")
        cfg_path = os.path.join(args.output_dir, f"config_{args.model_tag}.json")

    else:
        data = load_stereoset_pairs(args.data_path, split=args.split, bias_type=args.bias_type, limit=args.limit)
        print(f"Loaded {len(data)} StereoSet pairs")

        biased_acts, unbiased_acts = collect_activations_by_model_output_pairs(
            model, tokenizer, data, max_examples=args.num_examples_bad, max_length=256
        )
        classifiers, layer_accs = train_bad_classifiers(biased_acts, unbiased_acts, seed=args.seed)
        optimal_layer = int(np.argmax(layer_accs))
        dsv = compute_dsv_pairs(model, tokenizer, data, optimal_layer, num_pairs=args.num_pairs_dsv, max_length=256)

        out_dir = os.path.join(args.output_dir, f"model_{args.model_tag}")
        cfg_path = os.path.join(out_dir, "config.json")

    os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {
            "classifiers": classifiers,
            "optimal_layer": optimal_layer,
            "layer_accuracies": layer_accs,
        },
        os.path.join(out_dir, "bad_classifiers.pt"),
    )
    np.save(os.path.join(out_dir, "dsv.npy"), dsv)

    config = {
        "dataset": args.dataset,
        "model_name": model_name,
        "model_tag": args.model_tag,
        "optimal_layer": int(optimal_layer),
        "dsv_norm": float(np.linalg.norm(dsv)),
        "num_examples_bad": int(args.num_examples_bad),
        "num_pairs_dsv": int(args.num_pairs_dsv),
        "seed": int(args.seed),
    }
    if args.dataset == "bbq":
        config.update(
            {
                "bbq_data_dir": args.bbq_data_dir,
                "metadata_file": args.metadata_file,
                "num_examples": len(data),
            }
        )
    elif args.dataset == "crowspairs":
        config.update({"data_path": args.data_path, "bias_type": args.bias_type, "num_pairs": len(data)})
    else:
        config.update(
            {
                "data_path": args.data_path,
                "split": args.split,
                "bias_type": args.bias_type,
                "num_pairs": len(data),
                "limit": args.limit,
            }
        )

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved BAD classifiers + DSV to {out_dir}")
    print(f"Optimal layer: {optimal_layer}, accuracy={layer_accs[optimal_layer]:.4f}")


if __name__ == "__main__":
    main()
