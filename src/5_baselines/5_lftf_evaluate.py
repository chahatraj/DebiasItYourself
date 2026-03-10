#!/usr/bin/env python3
"""
LFTF evaluation entrypoint.

Paper : "LFTF: Locating First and Then Fine-Tuning for Mitigating Gender Bias
        in Large Language Models"
        arXiv:2505.15475

Loads the LoRA adapter from 5_lftf_train.py (single-block targeted adapter) and
evaluates on BBQ (accuracy + bias score), CrowS-Pairs (% stereotype preferred),
and StereoSet (lm_score, ss_score, icat_score).
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


BASE_DIR = Path(__file__).resolve().parent
BBQ_CATEGORIES = [
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

SEED = 42
torch.manual_seed(SEED)
set_seed(SEED)

AVAILABLE_MODELS: Dict[str, str] = {
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None) -> ModuleType:
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_model_name(model_name: Optional[str], model_alias: Optional[str]) -> str:
    if model_name:
        return model_name
    if model_alias in (None, "llama_8b"):
        return AVAILABLE_MODELS["llama_8b"]
    raise ValueError(f"Unsupported --model alias: {model_alias}")


def get_quantization_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _load_base_model_and_tokenizer(model_name: str, use_4bit: bool, hf_token: str, tokenizer_source: Optional[str] = None):
    tok_src = tokenizer_source if tokenizer_source else model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_quantization_config(use_4bit),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    return model, tokenizer


def _evaluate_bbq(args: argparse.Namespace, model_name: str):
    exp_mod = _load_module(
        "bbq_eval_shared_for_lftf_eval",
        BASE_DIR.parent / "3_experiments" / "7_eval_shared.py",
        add_to_syspath=BASE_DIR.parent / "3_experiments",
    )

    tokenizer_source = None
    if args.lftf_model_dir and os.path.exists(args.lftf_model_dir):
        tokenizer_source = args.lftf_model_dir

    base_model, tokenizer = _load_base_model_and_tokenizer(
        model_name=model_name,
        use_4bit=args.use_4bit,
        hf_token=args.hf_token,
        tokenizer_source=tokenizer_source,
    )

    if args.eval_baseline or not args.lftf_model_dir:
        model = base_model
    else:
        if not os.path.exists(args.lftf_model_dir):
            raise FileNotFoundError(f"LFTF model dir not found: {args.lftf_model_dir}")
        model = PeftModel.from_pretrained(base_model, args.lftf_model_dir)

    all_results = []
    all_preds = []

    categories = [args.category] if args.category else BBQ_CATEGORIES

    prompt_template = """{context}
Question: {question}
A. {ans0}
B. {ans1}
C. {ans2}
Answer:"""

    for category in categories:
        csv_url = (
            "https://raw.githubusercontent.com/chahatraj/DebiasItYourself/main/outputs/2_base_models/"
            f"bbq/llama_8b/bbq_preds_llama_8b_{category}.csv"
        )
        df = pd.read_csv(csv_url)
        if args.max_examples:
            df = df.sample(n=min(args.max_examples, len(df)), random_state=SEED)

        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=category, leave=False):
            prompt = prompt_template.format(
                context=row["context"],
                question=row["question"],
                ans0=row["ans0"],
                ans1=row["ans1"],
                ans2=row["ans2"],
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).lower().strip()

            pred = 2
            if generated.startswith("a"):
                pred = 0
            elif generated.startswith("b"):
                pred = 1
            elif generated.startswith("c"):
                pred = 2

            preds.append(
                {
                    "example_id": row.get("example_id"),
                    "source_file": f"{category}.jsonl",
                    "context_condition": row.get("context_condition"),
                    "label": int(row.get("label")),
                    "pred_index": int(pred),
                    "question_polarity": row.get("question_polarity"),
                    "target_loc": row.get("target_loc"),
                    "ans0": row.get("ans0"),
                    "ans1": row.get("ans1"),
                    "ans2": row.get("ans2"),
                    "model_output": row.get(f"ans{pred}"),
                }
            )

        preds_df = pd.DataFrame(preds)
        metrics = exp_mod.compute_bbq_metrics_row(
            preds_df=preds_df,
            model_name=category,
            metadata=args.metadata_file,
            processed=args.processed_file,
        )
        metrics["Category"] = category
        all_results.append(metrics)
        all_preds.extend(preds)

    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results_df.to_csv(args.output_file, index=False)

    if all_preds:
        preds_df = pd.DataFrame(all_preds)
        pred_cols = [
            "example_id",
            "source_file",
            "context_condition",
            "label",
            "pred_index",
            "question_polarity",
            "target_loc",
        ]
        preds_df = preds_df[[c for c in pred_cols if c in preds_df.columns]]
        os.makedirs(os.path.dirname(args.preds_output_file), exist_ok=True)
        preds_df.to_csv(args.preds_output_file, index=False)

    print(f"Saved BBQ metrics to {args.output_file}")
    print(f"Saved BBQ predictions to {args.preds_output_file}")


def _evaluate_crowspairs(args: argparse.Namespace, model_name: str):
    crows_mod = _load_module(
        "crowspairs_eval_shared_for_lftf_eval",
        BASE_DIR.parent / "3_experiments" / "7_eval_shared.py",
        add_to_syspath=BASE_DIR.parent / "3_experiments",
    )

    tokenizer_source = None
    if not args.eval_baseline and args.lftf_model_dir and os.path.exists(args.lftf_model_dir):
        tokenizer_source = args.lftf_model_dir

    base_model, tokenizer = _load_base_model_and_tokenizer(
        model_name=model_name,
        use_4bit=args.use_4bit,
        hf_token=args.hf_token,
        tokenizer_source=tokenizer_source,
    )

    if args.eval_baseline:
        model = base_model
        eval_model_name = f"{args.model}_baseline"
    else:
        if not args.lftf_model_dir or not os.path.exists(args.lftf_model_dir):
            raise FileNotFoundError(f"LFTF model dir not found: {args.lftf_model_dir}")
        model = PeftModel.from_pretrained(base_model, args.lftf_model_dir)
        eval_model_name = f"{args.model}_lftf"

    raw_df = crows_mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = crows_mod.make_crowspairs_eval_pairs(raw_df)

    sent_more_scores = crows_mod.sequence_logprob_batch_from_texts(
        model, tokenizer, pair_df["sent_more"].tolist(), batch_size=args.batch_size
    )
    sent_less_scores = crows_mod.sequence_logprob_batch_from_texts(
        model, tokenizer, pair_df["sent_less"].tolist(), batch_size=args.batch_size
    )

    scored_df = crows_mod.build_scored_from_sentence_scores(
        pair_df,
        sent_more_scores=sent_more_scores,
        sent_less_scores=sent_less_scores,
    )
    overall_df, per_bias_df = crows_mod.compute_metrics_from_scored(scored_df, model_name=eval_model_name)

    os.makedirs(os.path.dirname(args.pairs_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.per_bias_output_file), exist_ok=True)

    scored_df.to_csv(args.pairs_output_file, index=False)
    overall_df.to_csv(args.output_file, index=False)
    per_bias_df.to_csv(args.per_bias_output_file, index=False)

    print(f"Saved CrowS-Pairs scores to {args.pairs_output_file}")
    print(f"Saved CrowS-Pairs overall metrics to {args.output_file}")
    print(f"Saved CrowS-Pairs per-bias metrics to {args.per_bias_output_file}")


def _evaluate_stereoset(args: argparse.Namespace, model_name: str):
    shared_mod = _load_module(
        "paper_baselines_shared_for_lftf_eval_stereo",
        BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    common_mod = shared_mod.get_dataset_common("stereoset")

    tokenizer_source = None
    if not args.eval_baseline and args.lftf_model_dir and os.path.exists(args.lftf_model_dir):
        tokenizer_source = args.lftf_model_dir

    base_model, tokenizer = _load_base_model_and_tokenizer(
        model_name=model_name,
        use_4bit=args.use_4bit,
        hf_token=args.hf_token,
        tokenizer_source=tokenizer_source,
    )

    if args.eval_baseline:
        model = base_model
        eval_model_name = f"{args.model}_baseline"
    else:
        if not args.lftf_model_dir or not os.path.exists(args.lftf_model_dir):
            raise FileNotFoundError(f"LFTF model dir not found: {args.lftf_model_dir}")
        model = PeftModel.from_pretrained(base_model, args.lftf_model_dir)
        eval_model_name = f"{args.model}_lftf"

    data = common_mod.load_stereoset_data(args.data_path)
    examples = common_mod.flatten_examples(data, split=args.split, bias_type=args.bias_type, limit=args.limit)
    if not examples:
        raise ValueError("No StereoSet examples to evaluate after filtering")

    records = common_mod.build_sentence_records(examples)
    sentence_texts = [x["sentence"] for x in records]
    sentence_ids = [x["sentence_id"] for x in records]
    sentence_splits = [x["split"] for x in records]

    scores = common_mod.sequence_logprob_batch(model, tokenizer, sentence_texts, args.batch_size)
    id2score = {sid: float(sc) for sid, sc in zip(sentence_ids, scores)}

    preds_json = {"intrasentence": [], "intersentence": []}
    for sid, sp in zip(sentence_ids, sentence_splits):
        preds_json[sp].append({"id": sid, "score": id2score[sid]})

    results = common_mod.stereoset_score(examples, id2score)
    for rec in records:
        rec["score"] = id2score.get(rec["sentence_id"], np.nan)

    rows = common_mod.nested_results_to_rows(results, eval_model_name)

    os.makedirs(os.path.dirname(args.predictions_file), exist_ok=True)
    with open(args.predictions_file, "w", encoding="utf-8") as f:
        json.dump(preds_json, f, indent=2)

    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.results_csv, index=False)

    os.makedirs(os.path.dirname(args.scores_output_file), exist_ok=True)
    pd.DataFrame(records).to_csv(args.scores_output_file, index=False)

    print(f"Saved StereoSet predictions to {args.predictions_file}")
    print(f"Saved StereoSet results JSON to {args.results_json}")
    print(f"Saved StereoSet results CSV to {args.results_csv}")
    print(f"Saved StereoSet sentence scores to {args.scores_output_file}")


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "output_file": "/scratch/craj/diy/results/3_baselines/lftf/bbq_eval_llama_8b_lftf_all.csv",
            "preds_output_file": "/scratch/craj/diy/outputs/3_baselines/lftf/bbq_preds_llama_8b_lftf_all.csv",
            "metadata_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "processed_file": "/scratch/craj/diy/data/processed_bbq_all.csv",
        },
        "crowspairs": {
            "model": "llama_8b",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "lftf_model_dir": "/scratch/craj/diy/outputs/3_baselines/lftf/models_crowspairs/model_crowspairs_all",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "batch_size": 4,
            "output_file": "/scratch/craj/diy/results/3_baselines/lftf/crowspairs_metrics_overall_llama_8b_lftf.csv",
            "per_bias_output_file": "/scratch/craj/diy/results/3_baselines/lftf/crowspairs_metrics_by_bias_llama_8b_lftf.csv",
            "pairs_output_file": "/scratch/craj/diy/outputs/3_baselines/lftf/crowspairs_scored_llama_8b_lftf.csv",
        },
        "stereoset": {
            "model": "llama_8b",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "lftf_model_dir": "/scratch/craj/diy/outputs/3_baselines/lftf/models_stereoset/model_stereoset_all",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "batch_size": 4,
            "results_json": "/scratch/craj/diy/results/3_baselines/lftf/stereoset_eval_llama_8b_lftf.json",
            "results_csv": "/scratch/craj/diy/results/3_baselines/lftf/stereoset_eval_llama_8b_lftf.csv",
            "predictions_file": "/scratch/craj/diy/outputs/3_baselines/lftf/stereoset_predictions_llama_8b_lftf.json",
            "scores_output_file": "/scratch/craj/diy/outputs/3_baselines/lftf/stereoset_sentence_scores_llama_8b_lftf.csv",
        },
    }

    for key, value in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LFTF evaluation across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--lftf_model_dir", type=str, default=None)
    parser.add_argument("--eval_baseline", action="store_true")

    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--preds_output_file", type=str, default=None)
    parser.add_argument("--metadata_file", type=str, default=None)
    parser.add_argument("--processed_file", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--category", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per_bias_output_file", type=str, default=None)
    parser.add_argument("--pairs_output_file", type=str, default=None)

    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--results_json", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--predictions_file", type=str, default=None)
    parser.add_argument("--scores_output_file", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_defaults(args)

    model_name = _resolve_model_name(args.model_name, args.model)

    if args.dataset == "bbq":
        _evaluate_bbq(args, model_name)
    elif args.dataset == "crowspairs":
        if args.base_model is None:
            args.base_model = model_name
        _evaluate_crowspairs(args, args.base_model)
    else:
        if args.base_model is None:
            args.base_model = model_name
        _evaluate_stereoset(args, args.base_model)


if __name__ == "__main__":
    main()
