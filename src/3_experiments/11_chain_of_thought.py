#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) debiasing experiment — base model.

For each debiasing strategy the three-step CoT reasoning prompt is prepended
to every input at inference time.  The base (non-fine-tuned) LLaMA model is
used throughout.  Evaluation is performed on CrowS-Pairs, StereoSet, and BBQ.

Usage examples
--------------
# CrowS-Pairs
python 11_chain_of_thought.py eval_crows \
    --strategy sr --model llama_8b \
    --output_dir .../outputs/11_cot/crowspairs \
    --results_dir .../results/11_cot/crowspairs

# StereoSet
python 11_chain_of_thought.py eval_stereoset \
    --strategy ci --model llama_8b

# BBQ
python 11_chain_of_thought.py infer_bbq \
    --strategy ind --source_file Age.jsonl --model llama_8b

# Aggregate BBQ metrics
python 11_chain_of_thought.py eval_bbq \
    --model_dir .../outputs/11_cot/bbq \
    --output_file .../results/11_cot/bbq/metrics.csv
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

# ── shared evaluation helpers ──────────────────────────────────────────────────
_EVAL_SHARED_PATH = Path(__file__).resolve().parent / "7_eval_shared.py"
_eval_spec = importlib.util.spec_from_file_location("eval_shared_for_cot", _EVAL_SHARED_PATH)
if _eval_spec is None or _eval_spec.loader is None:
    raise ImportError(f"Could not import {_EVAL_SHARED_PATH}")
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)

build_scored_from_sentence_scores = _eval_mod.build_scored_from_sentence_scores
compute_metrics_from_scored = _eval_mod.compute_metrics_from_scored
flatten_stereoset_examples = _eval_mod.flatten_stereoset_examples
load_crowspairs_df = _eval_mod.load_crowspairs_df
load_stereoset_data = _eval_mod.load_stereoset_data
make_crowspairs_eval_pairs = _eval_mod.make_crowspairs_eval_pairs
sequence_logprob_batch_from_texts = _eval_mod.sequence_logprob_batch_from_texts
stereoset_score = _eval_mod.stereoset_score

_BBQ_EVAL_PATH = Path(__file__).resolve().parent / "8_bbq_eval_shared.py"
_bbq_spec = importlib.util.spec_from_file_location("bbq_eval_shared_for_cot", _BBQ_EVAL_PATH)
if _bbq_spec is None or _bbq_spec.loader is None:
    raise ImportError(f"Could not import {_BBQ_EVAL_PATH}")
_bbq_mod = importlib.util.module_from_spec(_bbq_spec)
_bbq_spec.loader.exec_module(_bbq_mod)
evaluate_prediction_directory = _bbq_mod.evaluate_prediction_directory

# ── constants ──────────────────────────────────────────────────────────────────
SEED = 42

AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
    "llama_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/scratch/craj/cache/model_cache/llama-3.1-8b-instruct",
    },
    "llama_70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/cache/model_cache/llama-3.3-70b-instruct",
    },
}

DEFAULT_CROWS_PATH = "/scratch/craj/diy/data/crows_pairs_anonymized.csv"
DEFAULT_STEREOSET_PATH = "/scratch/craj/diy/data/stereoset/dev.json"
DEFAULT_BBQ_PROCESSED = "/scratch/craj/diy/data/processed_bbq_all.csv"
DEFAULT_BBQ_METADATA = "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv"

DEFAULT_CROWS_OUTPUT = "/scratch/craj/diy/outputs/11_chain_of_thought/crowspairs"
DEFAULT_CROWS_RESULTS = "/scratch/craj/diy/results/11_chain_of_thought/crowspairs"
DEFAULT_STEREO_OUTPUT = "/scratch/craj/diy/outputs/11_chain_of_thought/stereoset"
DEFAULT_STEREO_RESULTS = "/scratch/craj/diy/results/11_chain_of_thought/stereoset"
DEFAULT_BBQ_OUTPUT = "/scratch/craj/diy/outputs/11_chain_of_thought/bbq"
DEFAULT_BBQ_RESULTS = "/scratch/craj/diy/results/11_chain_of_thought/bbq"

VALID_BBQ_SOURCE_FILES = [
    "Age.jsonl",
    "Disability_status.jsonl",
    "Gender_identity.jsonl",
    "Nationality.jsonl",
    "Physical_appearance.jsonl",
    "Race_ethnicity.jsonl",
    "Race_x_gender.jsonl",
    "Race_x_SES.jsonl",
    "Religion.jsonl",
    "SES.jsonl",
    "Sexual_orientation.jsonl",
]

STRATEGY_ALIASES: Dict[str, str] = {
    "sr": "stereotype_replacement",
    "stereotype_replacement": "stereotype_replacement",
    "stereotype-replacement": "stereotype_replacement",
    "stereotype replacement": "stereotype_replacement",
    "ci": "counter_imaging",
    "counter_imaging": "counter_imaging",
    "counter-imaging": "counter_imaging",
    "counter imaging": "counter_imaging",
    "ind": "individuating",
    "individuating": "individuating",
    "pt": "perspective_taking",
    "perspective_taking": "perspective_taking",
    "perspective-taking": "perspective_taking",
    "perspective taking": "perspective_taking",
    "pc": "positive_contact",
    "positive_contact": "positive_contact",
    "positive-contact": "positive_contact",
    "positive contact": "positive_contact",
    "all": "all_strategies",
    "all_strategies": "all_strategies",
    "all-strategies": "all_strategies",
    "all strategies": "all_strategies",
}

STRATEGY_CANONICAL: Dict[str, str] = {
    "stereotype_replacement": "stereotype replacement",
    "counter_imaging": "counter imaging",
    "individuating": "individuating",
    "perspective_taking": "perspective taking",
    "positive_contact": "positive contact",
    "all_strategies": "all debiasing strategies",
}

# Three reasoning steps per strategy used to build the CoT prefix
STRATEGY_STEPS: Dict[str, List[str]] = {
    "stereotype_replacement": [
        "Recognize stereotype cues in the text.",
        "Reflect on why the stereotype is unfair or overgeneralized.",
        "Replace it with fair, specific, individualized wording.",
    ],
    "counter_imaging": [
        "Recognize stereotype cues in the text.",
        "Imagine a concrete counter-stereotypic case.",
        "Reframe the text around that counter example.",
    ],
    "individuating": [
        "Recognize stereotype cues in the text.",
        "Focus on individual-level details.",
        "Rewrite using individualized evidence instead of group assumptions.",
    ],
    "perspective_taking": [
        "Recognize stereotype cues in the text.",
        "Take the perspective of the targeted person or group.",
        "Rewrite with empathy and fairness.",
    ],
    "positive_contact": [
        "Recognize stereotype cues in the text.",
        "Recall positive cross-group interactions.",
        "Rewrite in a way that reinforces that positive framing.",
    ],
    "all_strategies": [
        "Recognize stereotype cues in the text.",
        "Select an appropriate debiasing strategy.",
        "Produce a fair, specific, and non-stereotypical revision.",
    ],
}


# ── helpers ────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def normalize_strategy(raw: str) -> str:
    key = str(raw).strip().lower()
    if key not in STRATEGY_ALIASES:
        raise ValueError(f"Unknown strategy `{raw}`")
    return STRATEGY_ALIASES[key]


def resolve_tag(
    model: str,
    model_path: Optional[str],
    model_tag: Optional[str],
    strategy: str,
) -> str:
    if model_tag:
        base = model_tag
    elif model_path:
        base = os.path.basename(os.path.normpath(model_path))
    else:
        base = model
    return slug(f"{base}_cot_{strategy}")


def build_cot_prefix(strategy: str) -> str:
    """Return the chain-of-thought instruction prefix for the given strategy."""
    strategy_name = STRATEGY_CANONICAL[strategy]
    steps = STRATEGY_STEPS[strategy]
    return (
        f"Use chain-of-thought style debiasing with {strategy_name}.\n"
        "Follow these steps before producing the final answer:\n"
        f"1. {steps[0]}\n"
        f"2. {steps[1]}\n"
        f"3. {steps[2]}"
    )


def apply_cot_to_content(content: str, strategy: str) -> str:
    prefix = build_cot_prefix(strategy)
    return f"{prefix}\n\nContent:\n{content}"


def load_model_and_tokenizer(
    model_key: str,
    model_path: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model `{model_key}`")

    info = AVAILABLE_MODELS[model_key]
    model_source = model_path if model_path else info["model"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=info["cache_dir"])
    except Exception as exc:
        if model_path:
            print(f"Tokenizer load from {model_source} failed ({exc}); fallback to base tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(info["model"], cache_dir=info["cache_dir"])
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                quantization_config=quant_config,
                device_map="auto",
                cache_dir=info["cache_dir"],
            )
        except Exception as exc:
            print(f"Quantized load failed ({exc}); fallback to float16.")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=info["cache_dir"],
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.float32,
            cache_dir=info["cache_dir"],
        )

    model.eval()
    return model, tokenizer, model_source


def completion_logprob_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completions: Sequence[str],
    max_length: int,
) -> List[float]:
    """
    For each completion string return the sum of log-probs of its tokens
    conditioned on ``prompt``.
    """
    prefix = prompt if prompt.endswith((" ", "\n", "\t")) else prompt + " "
    boundary = len(prefix)
    scores: List[float] = []

    for comp in completions:
        full_text = prefix + str(comp)
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        comp_positions = [
            i for i, (s, e) in enumerate(offsets) if int(e) > int(boundary)
        ]
        if not comp_positions:
            scores.append(float("-inf"))
            continue

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        lp = torch.log_softmax(logits, dim=-1)
        shift_lp = lp[:, :-1, :]

        score = 0.0
        for pos in comp_positions:
            if pos == 0:
                continue
            tok = int(input_ids[0, pos].item())
            score += float(shift_lp[0, pos - 1, tok].item())
        scores.append(score)

    return scores


def stereoset_sentence_records(examples: List[Dict], strategy: str) -> List[Dict]:
    records: List[Dict] = []
    for ex in examples:
        for s in ex.get("sentences", []):
            sid = s.get("id")
            text = s.get("sentence", "")
            if sid is None or not text:
                continue
            records.append(
                {
                    "split": ex["split"],
                    "example_id": ex["example_id"],
                    "target": ex.get("target", ""),
                    "bias_type": ex.get("bias_type", ""),
                    "sentence_id": sid,
                    "gold_label": s.get("gold_label", ""),
                    "sentence": text,
                    "scored_text": apply_cot_to_content(text, strategy),
                }
            )
    return records


def stereoset_results_rows(
    results: Dict, model_name: str, strategy: str
) -> List[Dict]:
    rows: List[Dict] = []
    for split_key in ["intrasentence", "intersentence"]:
        for domain, vals in results.get(split_key, {}).items():
            rows.append(
                {
                    "model": model_name,
                    "split": split_key,
                    "domain": domain,
                    "Count": vals.get("Count", 0),
                    "LM Score": vals.get("LM Score", 0.0),
                    "SS Score": vals.get("SS Score", 0.0),
                    "ICAT Score": vals.get("ICAT Score", 0.0),
                    "debias_method": "cot",
                    "strategy": strategy,
                }
            )
    overall_vals = results.get("overall", {})
    rows.append(
        {
            "model": model_name,
            "split": "overall",
            "domain": "overall",
            "Count": overall_vals.get("Count", 0),
            "LM Score": overall_vals.get("LM Score", 0.0),
            "SS Score": overall_vals.get("SS Score", 0.0),
            "ICAT Score": overall_vals.get("ICAT Score", 0.0),
            "debias_method": "cot",
            "strategy": strategy,
        }
    )
    return rows


def format_bbq_prompt(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:
    return f"{context}\n{question}\nA. {ans0}\nB. {ans1}\nC. {ans2}\nAnswer:"


# ── run functions ──────────────────────────────────────────────────────────────

def run_eval_crows(args: argparse.Namespace) -> None:
    strategy = normalize_strategy(args.strategy)
    tag = resolve_tag(args.model, args.model_path, args.model_tag, strategy)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    model, tokenizer, loaded_source = load_model_and_tokenizer(args.model, args.model_path)
    print(f"Loaded model for CrowS-Pairs: {loaded_source} (tag={tag})")

    raw_df = load_crowspairs_df(args.data_path, limit=args.limit)
    pair_df = make_crowspairs_eval_pairs(raw_df)

    sent_more_scored = [
        apply_cot_to_content(t, strategy) for t in pair_df["sent_more"].tolist()
    ]
    sent_less_scored = [
        apply_cot_to_content(t, strategy) for t in pair_df["sent_less"].tolist()
    ]

    print("Scoring sent_more sentences...")
    sent_more_scores = sequence_logprob_batch_from_texts(
        model, tokenizer, sent_more_scored, batch_size=args.batch_size
    )
    print("Scoring sent_less sentences...")
    sent_less_scores = sequence_logprob_batch_from_texts(
        model, tokenizer, sent_less_scored, batch_size=args.batch_size
    )

    scored_df = build_scored_from_sentence_scores(
        pair_df, sent_more_scores=sent_more_scores, sent_less_scores=sent_less_scores
    )
    scored_df["debias_method"] = "cot"
    scored_df["strategy"] = strategy

    pairs_outfile = os.path.join(args.output_dir, f"crows_pairs_scored_{tag}.csv")
    scored_df.to_csv(pairs_outfile, index=False)

    overall_metrics, per_bias_metrics = compute_metrics_from_scored(
        scored_df,
        model_name=tag,
        extra_fields={"debias_method": "cot", "strategy": strategy},
    )

    overall_outfile = os.path.join(args.results_dir, f"crows_pairs_metrics_overall_{tag}.csv")
    per_bias_outfile = os.path.join(args.results_dir, f"crows_pairs_metrics_by_bias_{tag}.csv")
    overall_metrics.to_csv(overall_outfile, index=False)
    per_bias_metrics.to_csv(per_bias_outfile, index=False)

    print(f"Saved pair scores: {pairs_outfile}")
    print(f"Saved overall metrics: {overall_outfile}")
    print(f"Saved per-bias metrics: {per_bias_outfile}")


def run_eval_stereoset(args: argparse.Namespace) -> None:
    strategy = normalize_strategy(args.strategy)
    tag = resolve_tag(args.model, args.model_path, args.model_tag, strategy)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    model, tokenizer, loaded_source = load_model_and_tokenizer(args.model, args.model_path)
    print(f"Loaded model for StereoSet: {loaded_source} (tag={tag})")

    data = load_stereoset_data(args.data_path)
    examples = flatten_stereoset_examples(
        data, split=args.split, bias_type=args.bias_type, limit=args.limit
    )
    if not examples:
        raise ValueError("No StereoSet examples found after filtering.")

    records = stereoset_sentence_records(examples, strategy)
    texts = [r["scored_text"] for r in records]
    sentence_ids = [r["sentence_id"] for r in records]
    sentence_splits = [r["split"] for r in records]

    scores = sequence_logprob_batch_from_texts(
        model, tokenizer, texts, batch_size=args.batch_size
    )
    id2score = {sid: float(sc) for sid, sc in zip(sentence_ids, scores)}

    preds_json: Dict[str, list] = {"intrasentence": [], "intersentence": []}
    for sid, sp in zip(sentence_ids, sentence_splits):
        preds_json[sp].append({"id": sid, "score": id2score[sid]})

    for rec in records:
        rec["score"] = id2score.get(rec["sentence_id"], np.nan)
        rec["debias_method"] = "cot"
        rec["strategy"] = strategy

    records_df = pd.DataFrame(records)
    results = stereoset_score(examples, id2score)
    rows_df = pd.DataFrame(
        stereoset_results_rows(results, model_name=tag, strategy=strategy)
    )

    scores_output_file = os.path.join(args.output_dir, f"stereoset_sentence_scores_{tag}.csv")
    predictions_file = os.path.join(args.output_dir, f"stereoset_predictions_{tag}.json")
    results_json_file = os.path.join(args.results_dir, f"stereoset_metrics_{tag}.json")
    results_csv_file = os.path.join(args.results_dir, f"stereoset_metrics_{tag}.csv")

    records_df.to_csv(scores_output_file, index=False)
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(preds_json, f, indent=2)
    with open(results_json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    rows_df.to_csv(results_csv_file, index=False)

    print(f"Saved sentence scores: {scores_output_file}")
    print(f"Saved predictions: {predictions_file}")
    print(f"Saved metrics json: {results_json_file}")
    print(f"Saved metrics csv: {results_csv_file}")


def run_infer_bbq(args: argparse.Namespace) -> None:
    strategy = normalize_strategy(args.strategy)
    tag = resolve_tag(args.model, args.model_path, args.model_tag, strategy)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, loaded_source = load_model_and_tokenizer(args.model, args.model_path)
    print(f"Loaded model for BBQ: {loaded_source} (tag={tag})")

    df = pd.read_csv(args.data_path)
    df["answer_info"] = df["answer_info"].apply(ast.literal_eval)
    df = df[df["source_file"] == args.source_file].reset_index(drop=True)
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()

    print(f"Running BBQ CoT inference on {len(df)} rows for {args.source_file}")

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="BBQ CoT inference"):
        options = [row["ans0"], row["ans1"], row["ans2"]]
        base_prompt = format_bbq_prompt(row["context"], row["question"], *options)
        scored_prompt = apply_cot_to_content(base_prompt, strategy)

        try:
            logprobs = completion_logprob_batch(
                model, tokenizer, scored_prompt, options, max_length=args.max_length
            )
            arr = np.array(logprobs, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                probs_arr = np.ones(len(options)) / len(options)
            else:
                m = float(np.max(finite))
                exps = np.where(np.isfinite(arr), np.exp(arr - m), 0.0)
                denom = float(exps.sum()) or 1.0
                probs_arr = exps / denom

            pred_idx = int(np.argmax(probs_arr))
            pred_letter = chr(65 + pred_idx)

            rows.append(
                {
                    "example_id": row.example_id,
                    "source_file": row.source_file,
                    "context_condition": row.context_condition,
                    "label": row.label,
                    "context": row.context,
                    "question": row.question,
                    "ans0": row.ans0,
                    "ans1": row.ans1,
                    "ans2": row.ans2,
                    "model_output": options[pred_idx],
                    "pred_letter": pred_letter,
                    "pred_index": pred_idx,
                    "option_probs": {chr(65 + k): float(p) for k, p in enumerate(probs_arr)},
                    "option_logprobs": {chr(65 + k): float(lp) for k, lp in enumerate(logprobs)},
                    "debias_method": "cot",
                    "strategy": strategy,
                }
            )
        except Exception as exc:
            print(f"Row {idx} failed: {exc}")

    out_file = os.path.join(
        args.output_dir,
        f"bbq_preds_{tag}_{args.source_file.replace('.jsonl', '')}.csv",
    )
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Saved BBQ predictions: {out_file}")


def run_eval_bbq(args: argparse.Namespace) -> None:
    meta = pd.read_csv(args.metadata_file)
    proc = pd.read_csv(args.processed_file)
    final_df = evaluate_prediction_directory(
        model_dir=args.model_dir,
        metadata=meta,
        processed=proc,
        model_name_prefix=args.model_name,
    )
    final_df["debias_method"] = "cot"
    final_df["strategy"] = ""
    for idx, row in final_df.iterrows():
        input_file = str(row.get("input_file", ""))
        if input_file in ("", "__overall__"):
            continue
        fpath = os.path.join(args.model_dir, input_file)
        if not os.path.exists(fpath):
            continue
        try:
            preds_df = pd.read_csv(fpath)
            cols = {str(c).strip().lower(): c for c in preds_df.columns}
            if "strategy" in cols and len(preds_df) > 0:
                final_df.at[idx, "strategy"] = str(preds_df.loc[0, cols["strategy"]])
        except Exception:
            continue

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    final_df.to_csv(args.output_file, index=False)
    print(f"Saved BBQ metrics: {args.output_file}")
    print(final_df.to_string(index=False))


# ── argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought debiasing experiment (base model)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_crows = sub.add_parser("eval_crows", help="Evaluate on CrowS-Pairs")
    p_crows.add_argument("--model", choices=list(AVAILABLE_MODELS), default="llama_8b")
    p_crows.add_argument("--model_path", type=str, default=None)
    p_crows.add_argument("--model_tag", type=str, default=None)
    p_crows.add_argument("--data_path", type=str, default=DEFAULT_CROWS_PATH)
    p_crows.add_argument("--output_dir", type=str, default=DEFAULT_CROWS_OUTPUT)
    p_crows.add_argument("--results_dir", type=str, default=DEFAULT_CROWS_RESULTS)
    p_crows.add_argument("--batch_size", type=int, default=4)
    p_crows.add_argument("--limit", type=int, default=None)
    p_crows.add_argument("--strategy", type=str, default="stereotype_replacement")

    p_stereo = sub.add_parser("eval_stereoset", help="Evaluate on StereoSet")
    p_stereo.add_argument("--model", choices=list(AVAILABLE_MODELS), default="llama_8b")
    p_stereo.add_argument("--model_path", type=str, default=None)
    p_stereo.add_argument("--model_tag", type=str, default=None)
    p_stereo.add_argument("--data_path", type=str, default=DEFAULT_STEREOSET_PATH)
    p_stereo.add_argument("--split", choices=["all", "intrasentence", "intersentence"], default="all")
    p_stereo.add_argument("--bias_type", type=str, default=None)
    p_stereo.add_argument("--limit", type=int, default=None)
    p_stereo.add_argument("--batch_size", type=int, default=4)
    p_stereo.add_argument("--output_dir", type=str, default=DEFAULT_STEREO_OUTPUT)
    p_stereo.add_argument("--results_dir", type=str, default=DEFAULT_STEREO_RESULTS)
    p_stereo.add_argument("--strategy", type=str, default="stereotype_replacement")

    p_bbq = sub.add_parser("infer_bbq", help="Run BBQ inference with CoT")
    p_bbq.add_argument("--model", choices=list(AVAILABLE_MODELS), default="llama_8b")
    p_bbq.add_argument("--model_path", type=str, default=None)
    p_bbq.add_argument("--model_tag", type=str, default=None)
    p_bbq.add_argument("--data_path", type=str, default=DEFAULT_BBQ_PROCESSED)
    p_bbq.add_argument("--source_file", choices=VALID_BBQ_SOURCE_FILES, required=True)
    p_bbq.add_argument("--output_dir", type=str, default=DEFAULT_BBQ_OUTPUT)
    p_bbq.add_argument("--max_length", type=int, default=1024)
    p_bbq.add_argument("--limit", type=int, default=None)
    p_bbq.add_argument("--strategy", type=str, default="stereotype_replacement")

    p_bbq_eval = sub.add_parser("eval_bbq", help="Aggregate BBQ metrics")
    p_bbq_eval.add_argument("--model_dir", type=str, required=True)
    p_bbq_eval.add_argument("--output_file", type=str, required=True)
    p_bbq_eval.add_argument("--metadata_file", type=str, default=DEFAULT_BBQ_METADATA)
    p_bbq_eval.add_argument("--processed_file", type=str, default=DEFAULT_BBQ_PROCESSED)
    p_bbq_eval.add_argument("--model_name", type=str, default=None)

    return parser


def main() -> None:
    seed_everything(SEED)
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "eval_crows":
        run_eval_crows(args)
    elif args.command == "eval_stereoset":
        run_eval_stereoset(args)
    elif args.command == "infer_bbq":
        run_infer_bbq(args)
    elif args.command == "eval_bbq":
        run_eval_bbq(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
