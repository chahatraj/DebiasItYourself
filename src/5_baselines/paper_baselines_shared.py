#!/usr/bin/env python3
"""Shared helper utilities and dataset adapters for baseline methods.

This module directly integrates:
- src/3_experiments/9_eval_shared.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR.parent / "3_experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CATEGORIES = [
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


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None) -> ModuleType:
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EVAL_SHARED = _load_module(
    "exp9_eval_shared_for_baselines",
    EXPERIMENTS_DIR / "9_eval_shared.py",
    add_to_syspath=EXPERIMENTS_DIR,
)

_BBQ_SHARED = _EVAL_SHARED
_CROWS_SHARED = _EVAL_SHARED
_STEREO_SHARED = _EVAL_SHARED


def get_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    hf_token: str,
    adapter_path: Optional[str] = None,
    base_model: str = BASE_MODEL,
):
    tokenizer_src = adapter_path if adapter_path else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=get_quant_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )

    if adapter_path:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def build_prompted_text(text: str, prompt_prefix: Optional[str]) -> str:
    if not prompt_prefix:
        return text
    return f"{prompt_prefix}\n\n{text}".strip()


def sequence_logprob_batch_from_texts(model, tokenizer, texts: List[str], batch_size: int = 4) -> List[float]:
    logprobs: List[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * shift_mask
        seq_log_probs = token_log_probs.sum(dim=1).detach().cpu().tolist()
        logprobs.extend([float(x) for x in seq_log_probs])

    return logprobs


def sequence_logprob_batch(model, tokenizer, sentences: List[str], batch_size: int) -> List[float]:
    return sequence_logprob_batch_from_texts(model, tokenizer, sentences, batch_size=batch_size)


def tokenize_texts(tokenizer, texts: List[str], device, max_length: int):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def batch_sequence_logps(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    seq_logps = (token_logps * shift_mask).sum(dim=-1)
    return seq_logps


def is_transient_cuda_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    needles = (
        "cuda error",
        "cublas_status_not_initialized",
        "out of memory",
        "outofmemory",
        "cudnn",
    )
    return any(x in msg for x in needles)


def recover_from_cuda_error(model, optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_GENDER_SWAP = {
    " he ": " she ",
    " she ": " he ",
    " him ": " her ",
    " her ": " him ",
    " his ": " hers ",
    " hers ": " his ",
    " man ": " woman ",
    " woman ": " man ",
    " men ": " women ",
    " women ": " men ",
    " boy ": " girl ",
    " girl ": " boy ",
}


def cda_swap_sentence(text: str) -> str:
    s = f" {text.strip()} "
    for a in _GENDER_SWAP:
        s = s.replace(a, f"<<TMP_{hash(a)}>>")
    for a, b in _GENDER_SWAP.items():
        s = s.replace(f"<<TMP_{hash(a)}>>", b)
    return s.strip()


# ----------------------
# BBQ helpers
# ----------------------


def load_metadata(meta_file: str) -> pd.DataFrame:
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    meta_df = pd.read_csv(meta_file)
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]
    needed = {"example_id", "target_loc", "category"}
    missing = needed - set(meta_df.columns)
    if missing:
        raise ValueError(f"Metadata must contain columns: {sorted(needed)}; missing: {sorted(missing)}")
    meta_df["category"] = meta_df["category"].astype(str).str.replace(".jsonl", "", regex=False)
    meta_df = meta_df[["category", "example_id", "target_loc"]].drop_duplicates(["category", "example_id"])
    return meta_df


def load_bbq_df(
    bbq_dir: str,
    meta_file: str,
    category: Optional[str] = None,
    limit_per_category: Optional[int] = None,
) -> pd.DataFrame:
    meta_df = load_metadata(meta_file)

    categories = [category] if category else list(CATEGORIES)
    frames: List[pd.DataFrame] = []

    for cat in categories:
        fp = os.path.join(bbq_dir, f"{cat}.jsonl")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"BBQ category file not found: {fp}")

        rows = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        df = pd.DataFrame(rows)
        required = {
            "example_id",
            "question_polarity",
            "context_condition",
            "context",
            "question",
            "ans0",
            "ans1",
            "ans2",
            "label",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {fp}: {sorted(missing)}")

        df["Category"] = cat
        if limit_per_category is not None:
            df = df.iloc[:limit_per_category].copy()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.merge(
        meta_df,
        left_on=["Category", "example_id"],
        right_on=["category", "example_id"],
        how="left",
    )
    if "category" in out.columns:
        out = out.drop(columns=["category"])
    out = out.dropna(subset=["target_loc"]).copy()
    out["target_loc"] = out["target_loc"].astype(int)
    out["label"] = out["label"].astype(int)
    return out


def build_prompt(row: pd.Series, prompt_prefix: Optional[str] = None) -> str:
    core = (
        f"{row['context']}\n"
        f"{row['question']}\n"
        f"A. {row['ans0']}\n"
        f"B. {row['ans1']}\n"
        f"C. {row['ans2']}\n"
        "Answer:"
    )
    if not prompt_prefix:
        return core
    return f"{prompt_prefix}\n\n{core}"


def build_preference_pairs(df: pd.DataFrame) -> pd.DataFrame:
    letters = ["A", "B", "C"]
    rows = []

    for _idx, row in df.iterrows():
        correct_idx = int(row["label"])
        target_idx = int(row["target_loc"])

        if correct_idx == target_idx:
            continue

        rows.append(
            {
                "example_id": int(row["example_id"]),
                "Category": row["Category"],
                "prompt": build_prompt(row),
                "chosen": letters[correct_idx],
                "rejected": letters[target_idx],
                "context_condition": row["context_condition"],
                "question_polarity": row["question_polarity"],
            }
        )

    pref_df = pd.DataFrame(rows)
    if pref_df.empty:
        raise ValueError("No preference pairs found. Check BBQ metadata alignment.")
    return pref_df


def _predict_option_probs_batch(model, tokenizer, prompts: List[str], batch_size: int = 8):
    option_ids = [
        tokenizer.encode("A", add_special_tokens=False)[0],
        tokenizer.encode("B", add_special_tokens=False)[0],
        tokenizer.encode("C", add_special_tokens=False)[0],
    ]

    pred_idx: List[int] = []
    probs_dict: List[Dict[str, float]] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]

        option_logits = logits[:, option_ids]
        probs = F.softmax(option_logits, dim=-1).detach().cpu().numpy()
        preds = option_logits.argmax(dim=-1).detach().cpu().tolist()

        for i, p in enumerate(preds):
            pred_idx.append(int(p))
            probs_dict.append(
                {
                    "A": float(probs[i, 0]),
                    "B": float(probs[i, 1]),
                    "C": float(probs[i, 2]),
                }
            )

    return pred_idx, probs_dict


def evaluate_bbq_df(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int = 8,
    prompt_prefix: Optional[str] = None,
    model_name: str = "llama_8b",
):
    prompts = [build_prompt(row, prompt_prefix=prompt_prefix) for _idx, row in df.iterrows()]
    pred_idx, option_probs = _predict_option_probs_batch(model, tokenizer, prompts, batch_size=batch_size)

    preds_df = df.copy()
    preds_df["pred_index"] = pred_idx
    preds_df["option_probs"] = option_probs

    metrics_df = _BBQ_SHARED.compute_bbq_metrics_table(
        preds_df=preds_df,
        model_name=model_name,
        metadata=None,
        processed=None,
        include_per_category=True,
        include_overall=True,
    )
    return metrics_df, preds_df


def save_bbq_eval_outputs(metrics_df: pd.DataFrame, preds_df: pd.DataFrame, output_file: str, preds_output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(preds_output_file), exist_ok=True)

    metrics_df.to_csv(output_file, index=False)

    pred_cols = [
        "example_id",
        "Category",
        "context_condition",
        "label",
        "pred_index",
        "question_polarity",
        "target_loc",
        "option_probs",
    ]
    keep_cols = [c for c in pred_cols if c in preds_df.columns]
    preds_df[keep_cols].to_csv(preds_output_file, index=False)


# ----------------------
# CrowS-Pairs helpers
# ----------------------


def make_pair_records(df: pd.DataFrame) -> pd.DataFrame:
    out = _CROWS_SHARED.make_crowspairs_eval_pairs(df)
    out["stereotype"] = out["sent_more"]
    out["anti"] = out["sent_less"]
    out["direction_label"] = out["stereo_antistereo"]
    if out.empty:
        raise ValueError("No valid sentence pairs after preprocessing.")
    return out


def load_crowspairs_df(data_path: str, bias_type: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    return _CROWS_SHARED.load_crowspairs_df(data_path, bias_type=bias_type, limit=limit)


def evaluate_pair_records(
    pair_df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int,
    prompt_prefix: Optional[str] = None,
) -> pd.DataFrame:
    normalized = pair_df.copy()
    if "sent_more" not in normalized.columns:
        if {"stereotype", "anti"}.issubset(normalized.columns):
            normalized["sent_more"] = normalized["stereotype"]
            normalized["sent_less"] = normalized["anti"]
            if "direction_label" in normalized.columns:
                normalized["stereo_antistereo"] = normalized["direction_label"]
            elif "stereo_antistereo" not in normalized.columns:
                normalized["stereo_antistereo"] = "unknown"
            if "bias_type" not in normalized.columns:
                normalized["bias_type"] = ""
        else:
            raise ValueError("pair_df must include sent_more/sent_less or stereotype/anti columns")

    sent_more_texts = [build_prompted_text(x, prompt_prefix) for x in normalized["sent_more"].tolist()]
    sent_less_texts = [build_prompted_text(x, prompt_prefix) for x in normalized["sent_less"].tolist()]

    sent_more_scores = _CROWS_SHARED.sequence_logprob_batch_from_texts(model, tokenizer, sent_more_texts, batch_size=batch_size)
    sent_less_scores = _CROWS_SHARED.sequence_logprob_batch_from_texts(model, tokenizer, sent_less_texts, batch_size=batch_size)

    scored = _CROWS_SHARED.build_scored_from_sentence_scores(
        normalized,
        sent_more_scores=sent_more_scores,
        sent_less_scores=sent_less_scores,
    )
    scored["stereotype"] = scored["sent_more"]
    scored["anti"] = scored["sent_less"]
    scored["direction_label"] = scored["stereo_antistereo"]
    return scored


def compute_metrics_from_scored(scored: pd.DataFrame, model_name: str):
    return _CROWS_SHARED.compute_metrics_from_scored(scored, model_name=model_name)


def save_crowspairs_eval_outputs(
    scored_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    per_bias_df: pd.DataFrame,
    pairs_output_file: str,
    output_file: str,
    per_bias_output_file: str,
) -> None:
    os.makedirs(os.path.dirname(pairs_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(per_bias_output_file), exist_ok=True)

    scored_df.to_csv(pairs_output_file, index=False)
    overall_df.to_csv(output_file, index=False)
    per_bias_df.to_csv(per_bias_output_file, index=False)


# ----------------------
# StereoSet helpers
# ----------------------


def load_stereoset_data(data_path: str):
    return _STEREO_SHARED.load_stereoset_data(data_path)


def flatten_examples(data, split: str = "all", bias_type: Optional[str] = None, limit: Optional[int] = None):
    return _STEREO_SHARED.flatten_stereoset_examples(data, split=split, bias_type=bias_type, limit=limit)


def build_sentence_records(examples: List[Dict]):
    return _STEREO_SHARED.build_stereoset_sentence_records(examples)


def stereoset_score(examples: List[Dict], id2score: Dict[str, float]):
    return _STEREO_SHARED.stereoset_score(examples, id2score)


def nested_results_to_rows(results: Dict, model_name: str):
    return _STEREO_SHARED.nested_results_to_rows(results, model_name)


def load_examples(
    data_path: str,
    split: str = "all",
    bias_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    data = load_stereoset_data(data_path)
    examples = flatten_examples(data, split=split, bias_type=bias_type, limit=limit)
    if not examples:
        raise ValueError("No StereoSet examples after filtering.")
    return examples


def evaluate_examples(
    examples: List[Dict],
    model,
    tokenizer,
    batch_size: int,
    prompt_prefix: Optional[str] = None,
):
    records = build_sentence_records(examples)
    texts = [build_prompted_text(str(x["sentence"]), prompt_prefix) for x in records]
    sentence_ids = [x["sentence_id"] for x in records]
    sentence_splits = [x["split"] for x in records]

    scores = sequence_logprob_batch_from_texts(model, tokenizer, texts, batch_size=batch_size)
    id2score = {sid: float(sc) for sid, sc in zip(sentence_ids, scores)}

    preds_json = {"intrasentence": [], "intersentence": []}
    for sid, sp in zip(sentence_ids, sentence_splits):
        preds_json[sp].append({"id": sid, "score": id2score[sid]})

    for rec in records:
        rec["score"] = id2score.get(rec["sentence_id"], np.nan)

    results = stereoset_score(examples, id2score)
    return pd.DataFrame(records), results, preds_json


def save_stereoset_eval_outputs(
    records_df: pd.DataFrame,
    results: Dict,
    preds_json: Dict,
    model_name: str,
    results_csv: str,
    results_json: str,
    predictions_file: str,
    scores_output_file: str,
) -> None:
    rows = nested_results_to_rows(results, model_name)
    rows_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(preds_json, f, indent=2)

    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    rows_df.to_csv(results_csv, index=False)

    os.makedirs(os.path.dirname(scores_output_file), exist_ok=True)
    records_df.to_csv(scores_output_file, index=False)


# ----------------------
# Training helpers
# ----------------------


@dataclass
class TrainStrategy:
    name: str
    chosen_lm_weight: float = 1.0
    anti_lm_weight: float = 1.0
    pair_pref_weight: float = 1.0
    gap_mse_weight: float = 0.0
    margin: float = 0.0
    cda_weight: float = 0.0


class BBQPairDataset(Dataset):
    def __init__(self, pair_df: pd.DataFrame):
        self.df = pair_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "chosen_text": f"{row['prompt']} {row['chosen']}",
            "rejected_text": f"{row['prompt']} {row['rejected']}",
        }


class PairDataset(Dataset):
    def __init__(self, pair_df: pd.DataFrame):
        self.df = pair_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "stereotype": row["stereotype"],
            "anti": row["anti"],
            "bias_type": row.get("bias_type", ""),
        }


def _train_bbq_pairwise(
    pair_df: pd.DataFrame,
    strategy: TrainStrategy,
    output_dir: str,
    model_tag: str,
    hf_token: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_quant_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_loader = DataLoader(BBQPairDataset(pair_df), batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        count = 0
        skipped = 0
        pbar = tqdm(data_loader, desc=f"{strategy.name} epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            try:
                chosen_texts = list(batch["chosen_text"])
                rejected_texts = list(batch["rejected_text"])

                chosen_enc = tokenize_texts(tokenizer, chosen_texts, model.device, max_length=max_length)
                rejected_enc = tokenize_texts(tokenizer, rejected_texts, model.device, max_length=max_length)

                chosen_out = model(**chosen_enc, labels=chosen_enc["input_ids"])
                rejected_out = model(**rejected_enc, labels=rejected_enc["input_ids"])

                chosen_lm_loss = chosen_out.loss

                lp_chosen = batch_sequence_logps(chosen_out.logits, chosen_enc["input_ids"], chosen_enc["attention_mask"])
                lp_rejected = batch_sequence_logps(rejected_out.logits, rejected_enc["input_ids"], rejected_enc["attention_mask"])

                pref_loss = F.softplus(strategy.margin + lp_rejected - lp_chosen).mean()
                gap_mse = (lp_rejected - lp_chosen).pow(2).mean()

                total_loss = strategy.chosen_lm_weight * chosen_lm_loss
                total_loss = total_loss + strategy.pair_pref_weight * pref_loss
                total_loss = total_loss + strategy.gap_mse_weight * gap_mse

                if strategy.cda_weight > 0:
                    cda_texts = [cda_swap_sentence(t) for t in chosen_texts]
                    cda_enc = tokenize_texts(tokenizer, cda_texts, model.device, max_length=max_length)
                    cda_out = model(**cda_enc, labels=cda_enc["input_ids"])
                    total_loss = total_loss + strategy.cda_weight * cda_out.loss

                total_loss.backward()
                optimizer.step()

                total += float(total_loss.item())
                count += 1
            except RuntimeError as err:
                if not is_transient_cuda_error(err):
                    raise
                skipped += 1
                recover_from_cuda_error(model, optimizer)

            pbar.set_postfix(loss=total / max(1, count), skipped=skipped)

        if skipped:
            print(f"[warn] {strategy.name} epoch {epoch + 1}: skipped {skipped} CUDA-fragile batches.")

    model_dir = os.path.join(output_dir, f"model_{model_tag}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    cfg = {
        "strategy": strategy.name,
        "model": BASE_MODEL,
        "model_tag": model_tag,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "chosen_lm_weight": strategy.chosen_lm_weight,
        "pair_pref_weight": strategy.pair_pref_weight,
        "gap_mse_weight": strategy.gap_mse_weight,
        "margin": strategy.margin,
        "cda_weight": strategy.cda_weight,
        "train_rows": int(len(pair_df)),
    }
    with open(os.path.join(output_dir, f"config_{model_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return model_dir


def _train_pairwise(
    pair_df: pd.DataFrame,
    strategy: TrainStrategy,
    output_dir: str,
    model_tag: str,
    hf_token: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_quant_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_loader = DataLoader(PairDataset(pair_df), batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        count = 0
        skipped = 0
        pbar = tqdm(data_loader, desc=f"{strategy.name} epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            try:
                stereo_texts = list(batch["stereotype"])
                anti_texts = list(batch["anti"])

                anti_enc = tokenize_texts(tokenizer, anti_texts, model.device, max_length=max_length)
                stereo_enc = tokenize_texts(tokenizer, stereo_texts, model.device, max_length=max_length)

                anti_out = model(**anti_enc, labels=anti_enc["input_ids"])
                stereo_out = model(**stereo_enc, labels=stereo_enc["input_ids"])

                anti_lm_loss = anti_out.loss

                lp_anti = batch_sequence_logps(anti_out.logits, anti_enc["input_ids"], anti_enc["attention_mask"])
                lp_stereo = batch_sequence_logps(stereo_out.logits, stereo_enc["input_ids"], stereo_enc["attention_mask"])

                pref_loss = F.softplus(strategy.margin + lp_stereo - lp_anti).mean()
                gap_mse = (lp_stereo - lp_anti).pow(2).mean()

                total_loss = strategy.anti_lm_weight * anti_lm_loss
                total_loss = total_loss + strategy.pair_pref_weight * pref_loss
                total_loss = total_loss + strategy.gap_mse_weight * gap_mse

                if strategy.cda_weight > 0:
                    cda_texts = [cda_swap_sentence(t) for t in anti_texts]
                    cda_enc = tokenize_texts(tokenizer, cda_texts, model.device, max_length=max_length)
                    cda_out = model(**cda_enc, labels=cda_enc["input_ids"])
                    total_loss = total_loss + strategy.cda_weight * cda_out.loss

                total_loss.backward()
                optimizer.step()

                total += float(total_loss.item())
                count += 1
            except RuntimeError as err:
                if not is_transient_cuda_error(err):
                    raise
                skipped += 1
                recover_from_cuda_error(model, optimizer)

            pbar.set_postfix(loss=total / max(1, count), skipped=skipped)

        if skipped:
            print(f"[warn] {strategy.name} epoch {epoch + 1}: skipped {skipped} CUDA-fragile batches.")

    model_dir = os.path.join(output_dir, f"model_{model_tag}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    cfg = {
        "strategy": strategy.name,
        "model": BASE_MODEL,
        "model_tag": model_tag,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "anti_lm_weight": strategy.anti_lm_weight,
        "pair_pref_weight": strategy.pair_pref_weight,
        "gap_mse_weight": strategy.gap_mse_weight,
        "margin": strategy.margin,
        "cda_weight": strategy.cda_weight,
        "train_rows": int(len(pair_df)),
    }
    with open(os.path.join(output_dir, f"config_{model_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return model_dir


def train_lora_pairwise(
    train_data,
    strategy: TrainStrategy,
    output_dir: str,
    model_tag: str,
    hf_token: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_length: int = 256,
    lora_r: int = 8,
    lora_alpha: int = 16,
    dataset: Optional[str] = None,
) -> str:
    if dataset == "bbq":
        return _train_bbq_pairwise(
            train_data,
            strategy,
            output_dir,
            model_tag,
            hf_token,
            epochs,
            batch_size,
            lr,
            max_length,
            lora_r,
            lora_alpha,
        )

    if dataset == "stereoset":
        pair_df = pd.DataFrame(
            [
                {
                    "stereotype": str(ex.get("stereotype", "")).strip(),
                    "anti": str(ex.get("anti", "")).strip(),
                    "bias_type": ex.get("bias_type", ""),
                }
                for ex in train_data
                if str(ex.get("stereotype", "")).strip() and str(ex.get("anti", "")).strip()
            ]
        )
        if pair_df.empty:
            raise ValueError("No valid StereoSet stereotype/anti pairs for training")
        return _train_pairwise(
            pair_df,
            strategy,
            output_dir,
            model_tag,
            hf_token,
            epochs,
            batch_size,
            lr,
            max_length,
            lora_r,
            lora_alpha,
        )

    return _train_pairwise(
        train_data,
        strategy,
        output_dir,
        model_tag,
        hf_token,
        epochs,
        batch_size,
        lr,
        max_length,
        lora_r,
        lora_alpha,
    )


# ----------------------
# Dataset adapters
# ----------------------


def _bbq_train_lora_pairwise(train_data, strategy: TrainStrategy, **kwargs):
    return train_lora_pairwise(train_data, strategy, dataset="bbq", **kwargs)


def _crowspairs_train_lora_pairwise(train_data, strategy: TrainStrategy, **kwargs):
    return train_lora_pairwise(train_data, strategy, dataset="crowspairs", **kwargs)


def _stereoset_train_lora_pairwise(train_data, strategy: TrainStrategy, **kwargs):
    return train_lora_pairwise(train_data, strategy, dataset="stereoset", **kwargs)


_BBQ_COMMON = SimpleNamespace(
    BASE_MODEL=BASE_MODEL,
    CATEGORIES=CATEGORIES,
    TrainStrategy=TrainStrategy,
    load_model_and_tokenizer=load_model_and_tokenizer,
    load_bbq_df=load_bbq_df,
    build_preference_pairs=build_preference_pairs,
    evaluate_bbq_df=evaluate_bbq_df,
    save_eval_outputs=save_bbq_eval_outputs,
    train_lora_pairwise=_bbq_train_lora_pairwise,
)

_CROWSPAIRS_COMMON = SimpleNamespace(
    BASE_MODEL=BASE_MODEL,
    TrainStrategy=TrainStrategy,
    load_model_and_tokenizer=load_model_and_tokenizer,
    load_crowspairs_df=load_crowspairs_df,
    make_pair_records=make_pair_records,
    evaluate_pair_records=evaluate_pair_records,
    compute_metrics_from_scored=compute_metrics_from_scored,
    save_eval_outputs=save_crowspairs_eval_outputs,
    train_lora_pairwise=_crowspairs_train_lora_pairwise,
)

_STEREOSET_COMMON = SimpleNamespace(
    BASE_MODEL=BASE_MODEL,
    TrainStrategy=TrainStrategy,
    load_model_and_tokenizer=load_model_and_tokenizer,
    load_stereoset_data=load_stereoset_data,
    flatten_examples=flatten_examples,
    build_sentence_records=build_sentence_records,
    sequence_logprob_batch=sequence_logprob_batch,
    stereoset_score=stereoset_score,
    nested_results_to_rows=nested_results_to_rows,
    load_examples=load_examples,
    evaluate_examples=evaluate_examples,
    save_eval_outputs=save_stereoset_eval_outputs,
    train_lora_pairwise=_stereoset_train_lora_pairwise,
)


def get_dataset_common(dataset: str):
    if dataset == "bbq":
        return _BBQ_COMMON
    if dataset == "crowspairs":
        return _CROWSPAIRS_COMMON
    if dataset == "stereoset":
        return _STEREOSET_COMMON
    raise ValueError(f"Unsupported dataset '{dataset}'")


__all__ = [
    "BASE_MODEL",
    "CATEGORIES",
    "TrainStrategy",
    "batch_sequence_logps",
    "build_prompted_text",
    "cda_swap_sentence",
    "compute_metrics_from_scored",
    "evaluate_bbq_df",
    "evaluate_examples",
    "evaluate_pair_records",
    "flatten_examples",
    "get_dataset_common",
    "get_quant_config",
    "is_transient_cuda_error",
    "load_bbq_df",
    "load_crowspairs_df",
    "load_examples",
    "load_model_and_tokenizer",
    "load_stereoset_data",
    "make_pair_records",
    "nested_results_to_rows",
    "recover_from_cuda_error",
    "save_bbq_eval_outputs",
    "save_crowspairs_eval_outputs",
    "save_stereoset_eval_outputs",
    "sequence_logprob_batch",
    "sequence_logprob_batch_from_texts",
    "stereoset_score",
    "tokenize_texts",
    "train_lora_pairwise",
]
