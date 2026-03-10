#!/usr/bin/env python3
"""
FairSteer evaluation: inference-time dynamic activation steering.

Paper : "FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering"
        arXiv:2504.14492
GitHub: https://github.com/LiYichen99/FairSteer

At inference, for each generated token:
  1. Collect hidden state at `optimal_layer`.
  2. Run BAD classifier → bias probability p.
  3. If p < 0.5 (biased prediction): steer hidden state by adding α * DSV.
  4. Continue generation with the modified hidden state.
Official α = 1.0 (dynamic: applied only when classifier detects bias).
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
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("bbq", "crowspairs", "stereoset")

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


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _resolve_model_name(model_name: Optional[str], model_alias: Optional[str]) -> str:
    if model_name:
        return model_name
    if model_alias in (None, "llama_8b"):
        return "meta-llama/Llama-3.1-8B-Instruct"
    raise ValueError(f"Unsupported --model alias: {model_alias}")


class ActivationSteering:
    def __init__(self, model, layer: int, classifier, dsv, threshold: float = 0.5, alpha: float = 1.0):
        self.model = model
        self.layer = layer
        self.classifier = classifier
        self.dsv = torch.tensor(dsv, dtype=torch.float16)
        self.threshold = threshold
        self.alpha = alpha
        self.hook = None

    def steering_hook(self, module, inp, output):
        hidden_states = output[0] if isinstance(output, tuple) else output

        last_token_act = hidden_states[0, -1, :].detach().cpu().numpy()
        prob_unbiased = self.classifier.predict_proba([last_token_act])[0][1]

        if prob_unbiased < self.threshold:
            self.dsv = self.dsv.to(hidden_states.device)
            hidden_states[0, -1, :] += self.alpha * self.dsv

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def register(self):
        layer_module = self.model.model.layers[self.layer]
        self.hook = layer_module.register_forward_hook(self.steering_hook)

    def remove(self):
        if self.hook:
            self.hook.remove()


def _load_model_and_components(args: argparse.Namespace):
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
        output_hidden_states=True,
        token=args.hf_token,
    )

    bad_data = torch.load(os.path.join(args.components_dir, "bad_classifiers.pt"), weights_only=False)
    classifiers = bad_data["classifiers"]
    optimal_layer = int(bad_data["optimal_layer"])
    classifier = classifiers[optimal_layer]
    dsv = np.load(os.path.join(args.components_dir, "dsv.npy"))

    steerer = None
    if args.use_steering and classifier is not None:
        steerer = ActivationSteering(model, optimal_layer, classifier, dsv, alpha=args.alpha)
        steerer.register()

    return model, tokenizer, steerer


def _evaluate_bbq(args: argparse.Namespace):
    exp_mod = _load_module(
        "bbq_eval_shared_for_fairsteer_eval",
        BASE_DIR.parent / "3_experiments" / "7_eval_shared.py",
        add_to_syspath=BASE_DIR.parent / "3_experiments",
    )

    model, tokenizer, steerer = _load_model_and_components(args)

    option_tokens = [
        tokenizer.encode("A", add_special_tokens=False)[0],
        tokenizer.encode("B", add_special_tokens=False)[0],
        tokenizer.encode("C", add_special_tokens=False)[0],
    ]

    categories = [args.category] if args.category else BBQ_CATEGORIES
    all_rows = []
    all_preds = []

    for category in categories:
        csv_path = os.path.join(args.bbq_data_dir, f"{category}.csv")
        jsonl_path = os.path.join(args.bbq_data_dir, f"{category}.jsonl")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        elif os.path.exists(jsonl_path):
            df = pd.read_json(jsonl_path, lines=True)
        else:
            print(f"[warn] Missing BBQ data file for {category}; skipping")
            continue

        if args.max_examples and len(df) > args.max_examples:
            df = df.sample(n=args.max_examples, random_state=SEED)

        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=category, leave=False):
            prompt = (
                f"{row['context']}\n"
                f"Question: {row['question']}\n"
                f"A. {row['ans0']}\n"
                f"B. {row['ans1']}\n"
                f"C. {row['ans2']}\n"
                "Answer:"
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :]
                option_logits = logits[:, option_tokens]
                probs = F.softmax(option_logits, dim=-1)
                pred = int(probs.argmax(dim=-1).item())

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
        all_rows.append(metrics)
        all_preds.extend(preds)

    if steerer:
        steerer.remove()

    if not all_rows:
        raise ValueError("No BBQ categories evaluated.")

    results_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results_df.to_csv(args.output_file, index=False)

    if all_preds:
        pred_df = pd.DataFrame(all_preds)
        pred_cols = [
            "example_id",
            "source_file",
            "context_condition",
            "label",
            "pred_index",
            "question_polarity",
            "target_loc",
        ]
        pred_df = pred_df[[c for c in pred_cols if c in pred_df.columns]]
        os.makedirs(os.path.dirname(args.preds_output_file), exist_ok=True)
        pred_df.to_csv(args.preds_output_file, index=False)

    print(f"Saved BBQ metrics to {args.output_file}")
    print(f"Saved BBQ predictions to {args.preds_output_file}")


def _evaluate_crowspairs(args: argparse.Namespace):
    crows_mod = _load_module(
        "crowspairs_eval_shared_for_fairsteer",
        BASE_DIR.parent / "3_experiments" / "7_eval_shared.py",
        add_to_syspath=BASE_DIR.parent / "3_experiments",
    )

    model, tokenizer, steerer = _load_model_and_components(args)

    raw_df = crows_mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = crows_mod.make_crowspairs_eval_pairs(raw_df)

    sent_more_scores = crows_mod.sequence_logprob_batch_from_texts(
        model, tokenizer, pair_df["sent_more"].tolist(), batch_size=args.batch_size
    )
    sent_less_scores = crows_mod.sequence_logprob_batch_from_texts(
        model, tokenizer, pair_df["sent_less"].tolist(), batch_size=args.batch_size
    )

    if steerer:
        steerer.remove()

    scored_df = crows_mod.build_scored_from_sentence_scores(
        pair_df,
        sent_more_scores=sent_more_scores,
        sent_less_scores=sent_less_scores,
    )

    model_name = "llama_8b_fairsteer_steered" if args.use_steering else "llama_8b_fairsteer_nosteer"
    overall_df, per_bias_df = crows_mod.compute_metrics_from_scored(scored_df, model_name=model_name)

    os.makedirs(os.path.dirname(args.pairs_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.per_bias_output_file), exist_ok=True)

    scored_df.to_csv(args.pairs_output_file, index=False)
    overall_df.to_csv(args.output_file, index=False)
    per_bias_df.to_csv(args.per_bias_output_file, index=False)

    print(f"Saved CrowS-Pairs scores to {args.pairs_output_file}")
    print(f"Saved CrowS-Pairs overall metrics to {args.output_file}")
    print(f"Saved CrowS-Pairs per-bias metrics to {args.per_bias_output_file}")


def _evaluate_stereoset(args: argparse.Namespace):
    shared_mod = _load_module(
        "paper_baselines_shared_for_fairsteer_eval_stereo",
        BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    common_mod = shared_mod.get_dataset_common("stereoset")

    model, tokenizer, steerer = _load_model_and_components(args)

    data = common_mod.load_stereoset_data(args.data_path)
    examples = common_mod.flatten_examples(data, split=args.split, bias_type=args.bias_type, limit=args.limit)
    if not examples:
        raise ValueError("No StereoSet examples after filtering")

    records = common_mod.build_sentence_records(examples)
    sentence_texts = [x["sentence"] for x in records]
    sentence_ids = [x["sentence_id"] for x in records]
    sentence_splits = [x["split"] for x in records]

    scores = common_mod.sequence_logprob_batch(model, tokenizer, sentence_texts, args.batch_size)
    if steerer:
        steerer.remove()

    id2score = {sid: float(sc) for sid, sc in zip(sentence_ids, scores)}
    preds_json = {"intrasentence": [], "intersentence": []}
    for sid, sp in zip(sentence_ids, sentence_splits):
        preds_json[sp].append({"id": sid, "score": id2score[sid]})

    results = common_mod.stereoset_score(examples, id2score)
    for rec in records:
        rec["score"] = id2score.get(rec["sentence_id"], np.nan)

    model_name = "llama_8b_fairsteer_steered" if args.use_steering else "llama_8b_fairsteer_nosteer"
    rows = common_mod.nested_results_to_rows(results, model_name)

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
            "bbq_data_dir": "/scratch/craj/diy/data/BBQ/data",
            "metadata_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "processed_file": "/scratch/craj/diy/data/processed_bbq_all.csv",
            "output_file": "/scratch/craj/diy/results/3_baselines/fairsteer/bbq_eval_llama_8b_fairsteer_all.csv",
            "preds_output_file": "/scratch/craj/diy/outputs/3_baselines/fairsteer/bbq_preds_llama_8b_fairsteer_all.csv",
        },
        "crowspairs": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "batch_size": 4,
            "output_file": "/scratch/craj/diy/results/3_baselines/fairsteer/crowspairs_metrics_overall_llama_8b_fairsteer.csv",
            "per_bias_output_file": "/scratch/craj/diy/results/3_baselines/fairsteer/crowspairs_metrics_by_bias_llama_8b_fairsteer.csv",
            "pairs_output_file": "/scratch/craj/diy/outputs/3_baselines/fairsteer/crowspairs_scored_llama_8b_fairsteer.csv",
        },
        "stereoset": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "batch_size": 4,
            "results_json": "/scratch/craj/diy/results/3_baselines/fairsteer/stereoset_eval_llama_8b_fairsteer.json",
            "results_csv": "/scratch/craj/diy/results/3_baselines/fairsteer/stereoset_eval_llama_8b_fairsteer.csv",
            "predictions_file": "/scratch/craj/diy/outputs/3_baselines/fairsteer/stereoset_predictions_llama_8b_fairsteer.json",
            "scores_output_file": "/scratch/craj/diy/outputs/3_baselines/fairsteer/stereoset_sentence_scores_llama_8b_fairsteer.csv",
        },
    }

    for k, v in defaults[args.dataset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FairSteer evaluation across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--components_dir", type=str, required=True)

    parser.add_argument("--use_steering", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering vector scale factor (official default: 1.0)")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--bbq_data_dir", type=str, default=None)
    parser.add_argument("--metadata_file", type=str, default=None)
    parser.add_argument("--processed_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--preds_output_file", type=str, default=None)
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

    parser.add_argument("--download_bbq", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_defaults(args)
    args.model_name = _resolve_model_name(args.model_name, args.model)

    if args.download_bbq and args.dataset == "bbq":
        print("[warn] --download_bbq not implemented in shared script; assuming local files present.")

    if args.dataset == "bbq":
        _evaluate_bbq(args)
    elif args.dataset == "crowspairs":
        _evaluate_crowspairs(args)
    else:
        _evaluate_stereoset(args)


if __name__ == "__main__":
    main()
