#!/usr/bin/env python3
# from html import parser
import os
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import wandb

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl
from huggingface_hub import HfApi

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

DEBIASED_INSTANCES_ROOT = "/scratch/craj/diy/outputs/1_generations/debiased_instances"

AVAILABLE_STRATEGIES = {
    "sr": "stereotype_replacement",
    "ci": "counter_imaging",
    "ind": "individuating",
    "pt": "perspective_taking",
    "pc": "positive_contact",
}

PAIR_TO_VERSION = {
    "opinion": "opinion_version",
    "action": "action_version",
    "event": "event_version",
}

ALL_STRATEGIES = sorted(AVAILABLE_STRATEGIES.values())
ALL_VERSIONS = sorted(PAIR_TO_VERSION.keys())
VERSION_ALIASES = {
    "opinion": "opinion",
    "opinion_version": "opinion",
    "action": "action",
    "action_version": "action",
    "event": "event",
    "event_version": "event",
}

KNOWN_SCENARIOS = [
    "art_and_leisure",
    "economics",
    "education",
    "environment",
    "healthcare",
    "law_and_policy",
    "media",
    "sports",
    "technology",
    "workplace",
]

KNOWN_BIAS_DIMENSIONS = [
    "age",
    "gender",
    "nationality",
    "physical_appearance",
    "physical_disability",
    "race_ethnicity",
    "religion",
    "sexual_orientation",
    "socioeconomic_status",
]

DEBIASING_PROMPTS = { #dont talk about task
    "stereotype_replacement": {
        "zero": "The following input may contain/trigger bias or stereotypes. Perform the stereotype replacement strategy to remove any bias present, following the steps below:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Reflect: Think why this stereotype might be inaccurate, overgeneralized, or harmful.\n"
                "Step 3 - Replace: Suggest a fairer, individualized, or bias-free alternative phrasing or interpretation.",
        },
    "counter_imaging": {
        "zero": "The following input may contain/trigger bias or stereotypes. Perform the counter imaging strategy to remove any bias present, following the steps below:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Imagine: Think of an imagined person who contradicts this stereotype.\n"
                "Step 3 - Reinforce: Elaborate details about this counter-stereotypic individual to strengthen the new association.",
        },
    "individuating": {
        "zero": "The following input may contain/trigger bias or stereotypes. Perform the individuating strategy to remove any bias present, following the steps below:\n"
                "Step 1 - Attend: Identify the stereotype and consciously focus on the individual, not their social group.\n"
                "Step 2 - Gather: Seek out specific, individuating information like traits, context, behaviors.\n"
                "Step 3 - Adjust: Revise or reinterpret the initial impressions using the individual details.",
        },
    "perspective_taking": {
        "zero": "The following input may contain/trigger bias or stereotypes. Perform the perspective taking strategy to remove any bias present, following the steps below:\n"
                "Step 1 - Adopt: Consciously take the perspective of the person being stereotyped.\n"
                "Step 2 - Simulate: Imagine what they might feel, think, or experience in that situation.\n"
                "Step 3 - Integrate: Use this perspective to reframe your assumptions or response.",
        },
    "positive_contact": {
        "zero": "The following input may contain/trigger bias or stereotypes. Perform the positive contact strategy to remove any bias present, following the steps below:\n"
                "Step 1 - Recall: Recall a situation where you had a meaningful, positive interaction with a person from the targeted group.\n"
                "Step 2 - Engage: Describe the interaction, what you learned, shared, or felt during it.\n"
                "Step 3 - Extend: Generalize that feeling to challenge the stereotype and reframe your beliefs.",
        }
}
# simple prompt trigger during inference
# cot
# ft + cot
# embedding to represent a strategy (soft prompting) + fixed/learneable embedding, just insert the embedding during inference
# token insertion
# post inference correction: how to debias a model after generating a response
# ft + prompt engg
# ft + incontext learning
# correcting using debiased model
# correcting false positives


# ============================================================
# Prompt formatting — Llama 3 chat template tokens
# ============================================================
def format_prompt(sys_prompt, instruction, input_text, output_text):
    PROMPT_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "{output_text}<|eot_id|>"
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

class LossModeCollator:
    def __init__(self, tokenizer, loss_mode: str, response_markers):
        self.tokenizer = tokenizer
        self.loss_mode = loss_mode
        self.response_markers = tuple(response_markers)
        self._warned_missing_boundary = False

    def _feature_to_text(self, feature):
        if "text" in feature:
            return feature["text"]
        if "input_ids" not in feature:
            raise ValueError("Feature missing both `text` and `input_ids`.")

        ids = feature["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if "attention_mask" in feature:
            mask = feature["attention_mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.tolist()
            ids = [tid for tid, m in zip(ids, mask) if int(m) == 1]

        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _response_char_start(self, text: str):
        for marker in self.response_markers:
            idx = text.find(marker)
            if idx != -1:
                return idx + len(marker)
        return None

    def __call__(self, features):
        texts = [self._feature_to_text(f) for f in features]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_offsets_mapping=(self.loss_mode == "response_only"),
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        if self.loss_mode == "response_only":
            offsets = enc["offset_mapping"]
            missing = 0
            for i in range(input_ids.size(0)):
                start_char = self._response_char_start(texts[i])
                if start_char is None:
                    labels[i, :] = -100
                    missing += 1
                    continue

                # Mask prompt tokens by char span; keep only response tokens.
                for j in range(input_ids.size(1)):
                    if attention_mask[i, j] == 0:
                        continue
                    end_char = int(offsets[i, j, 1].item())
                    if end_char <= start_char:
                        labels[i, j] = -100

            if missing > 0 and not self._warned_missing_boundary:
                print(
                    f"[WARN] response_only: missing response boundary in {missing}/{input_ids.size(0)} "
                    "samples in a batch; those samples are fully masked."
                )
                self._warned_missing_boundary = True

            del enc["offset_mapping"]

        enc["labels"] = labels
        return enc

class EvalHistoryCallback(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return
        self.history.append(
            {
                "epoch": state.epoch,
                "eval_loss": metrics.get("eval_loss"),
            }
        )

def clean_debiased_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Trim everything after the first standalone "assistant" token (case-insensitive).
    lower = text.lower()
    idx = lower.find("assistant")
    if idx != -1:
        before = lower[idx - 1] if idx > 0 else ""
        after = lower[idx + len("assistant")] if idx + len("assistant") < len(lower) else ""
        if (not before.isalnum()) and (not after.isalnum()):
            text = text[:idx]

    text = text.strip()

    # If the text looks like JSON, keep only the first JSON object.
    first_open = text.find("{")
    if first_open != -1:
        depth = 0
        end_idx = None
        for i, ch in enumerate(text[first_open:], start=first_open):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx is not None:
            text = text[first_open : end_idx + 1]

    return text.strip()


def load_table_file(file_path: str) -> pd.DataFrame:
    if not os.path.isfile(file_path):
        raise ValueError(f"Input file not found: {file_path}")

    if file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use .jsonl or .csv")

    print(f"[INFO] Loaded {len(df)} rows from {file_path}")
    return df


def resolve_merged_path(strategy: str, version: str) -> str:
    version_suffix = PAIR_TO_VERSION[version]
    filename = f"{strategy}_{version_suffix}_merged.jsonl"
    return os.path.join(DEBIASED_INSTANCES_ROOT, filename)


def _parse_csv_arg(raw: str):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.lower() == "all":
        return "all"
    return [p.strip() for p in raw.split(",") if p.strip()]


def _normalize_strategies(strategies_raw: str):
    parsed = _parse_csv_arg(strategies_raw)
    if parsed is None:
        return ALL_STRATEGIES
    if parsed == "all":
        return ALL_STRATEGIES

    normalized = []
    for item in parsed:
        if item in AVAILABLE_STRATEGIES:
            normalized.append(AVAILABLE_STRATEGIES[item])
        elif item in ALL_STRATEGIES:
            normalized.append(item)
        else:
            raise ValueError(f"Unknown strategy `{item}`. Expected one of {ALL_STRATEGIES} or keys {list(AVAILABLE_STRATEGIES.keys())}.")
    return sorted(set(normalized))


def _normalize_versions(versions_raw: str):
    parsed = _parse_csv_arg(versions_raw)
    if parsed is None:
        return ALL_VERSIONS
    if parsed == "all":
        return ALL_VERSIONS

    normalized = []
    for item in parsed:
        if item not in VERSION_ALIASES:
            raise ValueError(f"Unknown version `{item}`. Expected one of {list(VERSION_ALIASES.keys())}.")
        normalized.append(VERSION_ALIASES[item])
    return sorted(set(normalized))


def _normalize_optional_values(raw: str):
    parsed = _parse_csv_arg(raw)
    if parsed in (None, "all"):
        return None
    return sorted(set(parsed))


def _slug(text: str, max_len: int = 96) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    slug = "-".join([p for p in slug.split("-") if p])
    return slug[:max_len].rstrip("-")


def _selection_tag(values, full_values, empty_label: str = "all") -> str:
    if not values:
        return empty_label
    values = sorted(values)
    if values == sorted(full_values):
        return empty_label
    return "+".join(values)


def _presence_tag(values, label: str):
    if not values:
        return f"{label}-all"
    return f"{label}-{len(values)}"


def _apply_inclusion_filter(df: pd.DataFrame, col: str, values, label: str):
    if not values:
        return df
    if col not in df.columns:
        raise ValueError(f"Column `{col}` not found; cannot filter by {label}.")
    before = len(df)
    df = df[df[col].isin(values)]
    print(f"[INFO] Filtered {label}: {before} -> {len(df)} rows using {len(values)} value(s)")
    return df


def _sample_unique_values(df: pd.DataFrame, col: str, n: int, rng: np.random.Generator, label: str):
    if n is None:
        return df
    if n <= 0:
        raise ValueError(f"`{label}` count must be > 0")
    if col not in df.columns:
        raise ValueError(f"Column `{col}` not found; cannot sample {label}.")

    unique_vals = sorted(df[col].dropna().unique().tolist())
    if n >= len(unique_vals):
        print(f"[INFO] Requested {label}={n}; available={len(unique_vals)}. Keeping all.")
        return df

    chosen = rng.choice(np.array(unique_vals, dtype=object), size=n, replace=False).tolist()
    before = len(df)
    df = df[df[col].isin(chosen)]
    print(f"[INFO] Sampled {label}: selected {n}/{len(unique_vals)} unique values, rows {before} -> {len(df)}")
    return df


def _versions_with_data(df: pd.DataFrame, col_map):
    available = []
    for version, (src, tgt) in col_map.items():
        if src not in df.columns or tgt not in df.columns:
            continue
        if df[src].notna().any() and df[tgt].notna().any():
            available.append(version)
    return available


def _resolve_strategy_prompt(strategy_name: str, shot: str) -> str:
    if strategy_name not in DEBIASING_PROMPTS:
        raise ValueError(f"Unknown strategy prompt `{strategy_name}`.")
    shot_map = DEBIASING_PROMPTS[strategy_name]
    if shot in shot_map:
        return shot_map[shot]
    if "zero" in shot_map:
        print(
            f"[WARN] Shot `{shot}` not configured for strategy `{strategy_name}`. "
            "Falling back to `zero`."
        )
        return shot_map["zero"]
    raise ValueError(
        f"No prompt available for strategy `{strategy_name}` at shot `{shot}`."
    )


def _validate_allowed(values, allowed, label: str):
    if not values:
        return
    allowed_set = set(allowed)
    invalid = sorted(set(values) - allowed_set)
    if invalid:
        raise ValueError(
            f"Invalid {label}: {invalid}. Allowed values: {sorted(allowed_set)}"
        )


def _preview_examples(df: pd.DataFrame, sys_prompt: str, n: int):
    if n <= 0:
        return
    total = len(df)
    take_n = min(n, total)
    print(f"\n[PREVIEW] Showing {take_n}/{total} training example(s). No training will be run.\n")
    for i, row in df.head(take_n).reset_index(drop=True).iterrows():
        print("=" * 80)
        print(f"[EXAMPLE {i + 1}]")
        print(f"pair_type: {row.get('pair_type', '')}")
        print(f"strategy: {row.get('strategy', '')}")
        print(f"version: {row.get('version', '')}")
        print("-" * 80)
        print("[SYS PROMPT]")
        print(sys_prompt)
        print("-" * 80)
        print("[INSTRUCTION]")
        print(row.get("instruction_text", ""))
        print("-" * 80)
        print("[INPUT]")
        print(row.get("input_text", ""))
        print("-" * 80)
        print("[DEBIASED RESPONSE]")
        print(row.get("output_text", ""))
    print("=" * 80)
    print("[PREVIEW] Done.\n")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune causal LMs on debiased pairs with instruction-style prompts")
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Strategy selector: one/multiple/all. Accepts strategy keys (sr,ci,ind,pt,pc) or full names, comma-separated.",
    )
    parser.add_argument(
        "--versions",
        type=str,
        default="all",
        help="Version selector: one/multiple/all. Accepts opinion,action,event (or *_version), comma-separated.",
    )

    parser.add_argument(
        "--bias_dimensions",
        type=str,
        default=None,
        help="Bias-dimension selector: one/multiple/all, comma-separated.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Scenario selector: one/multiple/all, comma-separated.",
    )
    parser.add_argument("--num_identities", type=int, default=None, help="Randomly sample this many unique identities after filtering.")
    parser.add_argument("--num_concepts", type=int, default=None, help="Randomly sample this many unique concept templates after filtering.")
    parser.add_argument("--selection_seed", type=int, default=SEED, help="Random seed for scenario/identity/concept sampling.")
    parser.add_argument(
        "--cap_sampling_seed",
        type=int,
        default=None,
        help=(
            "Seed for --max_debias_samples subsampling. "
            "Default: derived from selection_seed and sample size to avoid nested subsets across size sweeps."
        ),
    )

    parser.add_argument("--model", type=str, choices=list(AVAILABLE_MODELS.keys()), default="llama_8b")
    parser.add_argument("--shot", type=str, choices=["zero", "one", "two", "five"], default="zero")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--preview_examples",
        type=int,
        default=0,
        help="If >0, print N assembled training examples and exit before model loading/training.",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["full_sequence", "response_only"],
        default="full_sequence",
        help="Loss target mode: full sequence or response-only after assistant header.",
    )
    parser.add_argument("--max_debias_samples", type=int, default=None, help="Optional cap on debias dataset rows before train/test split.")
    parser.add_argument("--alpaca_ratio", type=float, default=0.2, help="Alpaca augmentation ratio relative to debias rows.")
    parser.add_argument("--wandb_group", type=str, default=None, help="Optional W&B group for related runs.")
    parser.add_argument("--skip_push", action="store_true", help="Skip final Hub push (useful for quick ablation runs).")
    parser.add_argument(
        "--resume_if_available",
        action="store_true",
        help="Resume from last checkpoint in the auto-generated output directory if one exists. Default is off to avoid accidental carryover.",
    )
    args = parser.parse_args()
    if args.epochs > 3:
        raise ValueError("--epochs must be <= 3")

    selected_strategies = _normalize_strategies(args.strategies)
    selected_versions = _normalize_versions(args.versions)
    selected_bias_dims = _normalize_optional_values(args.bias_dimensions)
    _validate_allowed(selected_bias_dims, KNOWN_BIAS_DIMENSIONS, "bias_dimensions")
    selected_scenarios = _normalize_optional_values(args.scenarios)
    _validate_allowed(selected_scenarios, KNOWN_SCENARIOS, "scenarios")

    max_samples_tag = f"ms-{args.max_debias_samples if args.max_debias_samples is not None else 'full'}"
    if selected_strategies == ALL_STRATEGIES:
        strategy_component = "allstrategies"
    else:
        strategy_component = " ".join(selected_strategies)

    if selected_versions == ALL_VERSIONS:
        version_component = "opinion action event allversions"
    else:
        version_component = " ".join(selected_versions)

    slug_parts = [max_samples_tag, strategy_component, version_component]
    if selected_bias_dims:
        slug_parts.extend(sorted(selected_bias_dims))
    if selected_scenarios:
        slug_parts.extend(sorted(selected_scenarios))
    if args.num_identities is not None:
        slug_parts.append(f"ni-{args.num_identities}")
    if args.num_concepts is not None:
        slug_parts.append(f"nc-{args.num_concepts}")

    scope_slug = _slug(
        "_".join(slug_parts)
    )
    print(
        f"[INFO] Data selection: strategies={selected_strategies}, versions={selected_versions}, "
        f"bias_dimensions={selected_bias_dims or 'all'}"
    )
    print(f"[INFO] Loss mode: {args.loss_mode}")

    # =========================
    # Auto-naming output models
    # =========================
    base_out = "/scratch/craj/diy/outputs/7_finetuned_models"
    args.output_dir = os.path.join(base_out, f"finetuned_{scope_slug}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Saving model to: {args.output_dir}")

    if args.wandb_group:
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group

    # =========================
    # Dynamic Hugging Face Repo Naming
    # =========================
    base_ns = "chahatraj"
    HF_REPO = f"{base_ns}/diy_{scope_slug}"

    # === System Prompt ===
    sys_prompt = (
        "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, "
        "while being safe. Ensure responses are unbiased, factual, and positive."
    )

    # === Instruction mode ===
    print("[INFO] Instruction mode: per-example strategy instruction")

    # ========================
    # Load dataset (file-based)
    # ========================
    dfs = []
    for strategy_name in selected_strategies:
        for version_name in selected_versions:
            merged_path = resolve_merged_path(strategy_name, version_name)
            tmp_df = load_table_file(merged_path)
            tmp_df["strategy"] = strategy_name
            tmp_df["version"] = version_name
            dfs.append(tmp_df)
    if not dfs:
        raise ValueError("No input files resolved from the provided strategy/version selectors.")
    df = pd.concat(dfs, ignore_index=True)

    df = _apply_inclusion_filter(df, "bias_dimension", selected_bias_dims, "bias_dimension")
    df = _apply_inclusion_filter(df, "scenario", selected_scenarios, "scenario")

    selector_rng = np.random.default_rng(args.selection_seed)
    df = _sample_unique_values(df, "identity", args.num_identities, selector_rng, "identities")
    df = _sample_unique_values(df, "concept_template", args.num_concepts, selector_rng, "concept templates")

    if len(df) == 0:
        raise ValueError("No rows left after applying selection filters.")

    if "strategy" in df.columns:
        strategy_counts = df["strategy"].value_counts().to_dict()
        print(f"[INFO] Rows by strategy after filtering: {strategy_counts}")
    if "version" in df.columns:
        version_counts = df["version"].value_counts().to_dict()
        print(f"[INFO] Rows by version after filtering: {version_counts}")

    # Shuffle before split to reduce leakage from adjacent generated rows
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    for col in ["debiased_opinion_version", "debiased_action_version", "debiased_event_version"]:
        if col in df.columns:
            before = df[col].copy()
            df[col] = df[col].apply(clean_debiased_text)
            cleaned_count = (before != df[col]).sum()
            print(f"[INFO] Cleaned {cleaned_count} rows in {col}")

    col_map = {
        "opinion": ("opinion_version", "debiased_opinion_version"),
        "action": ("action_version", "debiased_action_version"),
        "event": ("event_version", "debiased_event_version"),
    }

    detected_versions = _versions_with_data(df, col_map)
    if not detected_versions:
        raise ValueError(
            "No usable opinion/action/event pairs found. "
            "Expected source+debiased columns like "
            "`opinion_version` + `debiased_opinion_version`."
        )

    requested_versions = list(selected_versions)
    selected_versions = [v for v in requested_versions if v in detected_versions]
    missing_versions = [v for v in requested_versions if v not in detected_versions]

    if missing_versions:
        print(
            f"[WARN] Requested versions missing in data: {missing_versions}. "
            f"Detected versions with data: {detected_versions}"
        )

    if not selected_versions:
        raise ValueError(
            f"Requested versions {requested_versions} not available. "
            f"Detected versions with data: {detected_versions}"
        )

    print(f"[INFO] Effective training versions: {selected_versions}")

    # === Build formatted debiasing dataset ===
    all_rows = []
    for version_name in selected_versions:
        src, tgt = col_map[version_name]
        keep_cols = [src, tgt]
        if "strategy" in df.columns:
            keep_cols.append("strategy")
        if "version" in df.columns:
            keep_cols.append("version")
        temp = df[keep_cols].dropna(subset=[src, tgt]).copy()
        if temp.empty:
            print(f"[WARN] Version `{version_name}` has zero rows after filtering; skipping.")
            continue
        temp["input_text"] = temp[src]
        temp["output_text"] = temp[tgt]
        temp["pair_type"] = version_name
        if "strategy" not in temp.columns:
            if len(selected_strategies) == 1:
                temp["strategy"] = selected_strategies[0]
            else:
                raise ValueError(
                    "Missing `strategy` column and multiple strategies selected. "
                    "Cannot assign per-example instructions."
                )
        if "version" not in temp.columns:
            temp["version"] = version_name
        temp["instruction_text"] = temp["strategy"].map(
            lambda s: _resolve_strategy_prompt(str(s), args.shot)
        )
        temp["text"] = temp.apply(
            lambda r: format_prompt(sys_prompt, r["instruction_text"], r["input_text"], r["output_text"]),
            axis=1,
        )
        all_rows.append(
            temp[
                [
                    "text",
                    "pair_type",
                    "strategy",
                    "version",
                    "instruction_text",
                    "input_text",
                    "output_text",
                ]
            ]
        )

    if not all_rows:
        raise ValueError("No rows available for selected versions after filtering.")

    combined_df = pd.concat(all_rows, ignore_index=True)

    if args.max_debias_samples is not None:
        take_n = min(args.max_debias_samples, len(combined_df))
        if args.cap_sampling_seed is None:
            cap_seed = int(args.selection_seed + (take_n * 1009))
            cap_seed_source = "auto-derived"
        else:
            cap_seed = int(args.cap_sampling_seed)
            cap_seed_source = "user-provided"
        cap_rng = np.random.default_rng(cap_seed)
        chosen_idx = cap_rng.choice(len(combined_df), size=take_n, replace=False).tolist()
        combined_df = combined_df.iloc[chosen_idx].reset_index(drop=True)
        print(
            f"[INFO] Capped debias_dataset to {len(combined_df)} rows "
            f"(cap_seed={cap_seed}, {cap_seed_source})"
        )

    if args.preview_examples > 0:
        _preview_examples(
            df=combined_df,
            sys_prompt=sys_prompt,
            n=args.preview_examples,
        )
        print("[INFO] Exiting because --preview_examples was set.")
        return

    debias_dataset = Dataset.from_pandas(combined_df[["text", "pair_type"]])

    # === Merge Alpaca-Cleaned data for quality (only 20% of debias size) ===
    debias_n = len(debias_dataset)
    alpaca_n = int(math.ceil(args.alpaca_ratio * debias_n))
    if alpaca_n > 0:
        alpaca = load_dataset("yahma/alpaca-cleaned", split="train")

        def _alpaca_to_text(example):
            inst = (example.get("instruction") or "").strip()
            inp = (example.get("input") or "").strip()
            out = (example.get("output") or "").strip()
            return {"text": format_prompt(sys_prompt, inst, inp, out)}

        # Sample deterministically from Alpaca
        alpaca = alpaca.shuffle(seed=SEED).select(range(min(alpaca_n, len(alpaca))))
        # Format Alpaca to match your dataset schema
        alpaca = alpaca.map(_alpaca_to_text, remove_columns=alpaca.column_names)
        dataset = concatenate_datasets([debias_dataset, alpaca]).shuffle(seed=SEED)
        alpaca_len = len(alpaca)
    else:
        dataset = debias_dataset.shuffle(seed=SEED)
        alpaca_len = 0

    print(f"[INFO] debias_dataset size = {debias_n}")
    print(f"[INFO] alpaca added size    = {alpaca_len} (target was {alpaca_n})")
    print(f"[INFO] combined dataset size = {len(dataset)}")


    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # === Model & Tokenizer ===
    model_info = AVAILABLE_MODELS[args.model]
    model_name = model_info["model"]
    model_load_kwargs = {
        "cache_dir": model_info["cache_dir"],
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            **model_load_kwargs,
        )
        print("[INFO] Attention backend: flash_attention_2")
    except Exception as e:
        err = str(e).lower()
        if not any(
            k in err
            for k in ("flash_attn", "flash attention", "attn_implementation", "undefined symbol")
        ):
            raise
        print(f"[WARN] flash_attention_2 unavailable ({e}). Falling back to sdpa.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            **model_load_kwargs,
        )
        print("[INFO] Attention backend: sdpa")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_info["cache_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))

    # === LoRA config ===
    peft_config = LoraConfig(
        r=256,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=["v_proj", "q_proj", "gate_proj", "k_proj", "up_proj", "down_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        optim="adamw_bnb_8bit",
        max_length=2048,
        logging_steps=1,
        report_to="wandb",
        run_name=f"ft_epoch{args.epochs}_{scope_slug}",
        push_to_hub=False,
        use_liger_kernel=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
    )

    # response_template_ids = tokenizer("\n[/INST]\n")["input_ids"][2:]
    # if DataCollatorForCompletionOnlyLM is None:
    #     raise RuntimeError(
    #         "DataCollatorForCompletionOnlyLM is required for SFTTrainer. "
    #         "Your installed trl version is too old."
    #     )

    # data_collator = DataCollatorForCompletionOnlyLM(
    #     response_template=response_template_ids,
    #     tokenizer=tokenizer,
    # )

    # try:
    #     trainer = SFTTrainer(
    #         model=model,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         dataset_text_field="text",
    #         data_collator=data_collator,
    #         peft_config=peft_config,
    #         tokenizer=tokenizer,
    #         args=training_args,
    #     )
    # except TypeError:
    #     try:
    #         trainer = SFTTrainer(
    #             model=model,
    #             train_dataset=train_dataset,
    #             eval_dataset=eval_dataset,
    #             data_collator=data_collator,
    #             peft_config=peft_config,
    #             tokenizer=tokenizer,
    #             args=training_args,
    #         )
    #     except TypeError:
    #         trainer = SFTTrainer(
    #             model=model,
    #             train_dataset=train_dataset,
    #             eval_dataset=eval_dataset,
    #             data_collator=data_collator,
    #             peft_config=peft_config,
    #             args=training_args,
    #         )

    # def build_sft_trainer(
    #     model,
    #     train_dataset,
    #     eval_dataset,
    #     tokenizer,
    #     training_args,
    #     data_collator,
    #     peft_config,
    # ):
    #     sig = inspect.signature(SFTTrainer.__init__)
    #     params = sig.parameters

    #     kwargs = {
    #         "model": model,
    #         "args": training_args,
    #         "train_dataset": train_dataset,
    #         "eval_dataset": eval_dataset,
    #         "data_collator": data_collator,
    #     }

    #     if "dataset_text_field" in params:
    #         kwargs["dataset_text_field"] = "text"

    #     if "tokenizer" in params:
    #         kwargs["tokenizer"] = tokenizer
    #     elif "processing_class" in params:
    #         kwargs["processing_class"] = tokenizer

    #     if "peft_config" in params:
    #         kwargs["peft_config"] = peft_config

    #     return SFTTrainer(**kwargs)


    # trainer = build_sft_trainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     training_args=training_args,
    #     data_collator=data_collator,
    #     peft_config=peft_config,
    # )


    response_markers = [
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>",
        "\n[/INST]\n",  # legacy fallback for older runs
        "[/INST]\n",    # legacy fallback for older runs
        "[/INST]",      # legacy fallback for older runs
    ]

    data_collator = LossModeCollator(
        tokenizer=tokenizer,
        loss_mode=args.loss_mode,
        response_markers=response_markers,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",
    )


    trainer.add_callback(LossLoggingCallback())
    eval_history_cb = EvalHistoryCallback()
    trainer.add_callback(eval_history_cb)

    resume_checkpoint = None
    if os.path.isdir(args.output_dir):
        resume_checkpoint = get_last_checkpoint(args.output_dir)
    if resume_checkpoint:
        if args.resume_if_available:
            print(f"[INFO] Resuming training from checkpoint: {resume_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            raise ValueError(
                f"Found existing checkpoint at {resume_checkpoint}, but resume is disabled. "
                "For a fresh run, change selection/sample args to generate a new run directory, "
                "or pass --resume_if_available to continue."
            )
    else:
        trainer.train()
    final_eval = trainer.evaluate()
    print("Final eval_loss:", final_eval["eval_loss"])
    trainer.log_metrics("eval", final_eval)
    trainer.save_metrics("eval", final_eval)

    # Save eval history and log to W&B
    if eval_history_cb.history:
        hist_df = pd.DataFrame(eval_history_cb.history)
        hist_path = os.path.join(args.output_dir, "eval_history.csv")
        hist_df.to_csv(hist_path, index=False)
        table = wandb.Table(dataframe=hist_df)
        wandb.log(
            {
                "eval_history": table,
                "eval_loss_by_epoch": wandb.plot.line(
                    table, "epoch", "eval_loss", title="Eval Loss by Epoch"
                ),
            }
        )

    # Save adapter checkpoints (for reference)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Merge LoRA weights into the base model and save full model
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.skip_push:
        print("Skipping Hub push for this run (--skip_push).")
    else:
        print(f"Pushing merged full model to Hugging Face Hub → {HF_REPO}")
        HF_TOKEN = os.getenv("HF_TOKEN")
        HfApi().create_repo(repo_id=HF_REPO, token=HF_TOKEN, exist_ok=True)
        merged_model.push_to_hub(HF_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)


if __name__ == "__main__":
    main()
