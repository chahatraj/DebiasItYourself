#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

_EVAL9_PATH = Path(__file__).resolve().parent / "7_eval_shared.py"
_eval9_spec = importlib.util.spec_from_file_location("eval_shared9_for_soft_prompting", _EVAL9_PATH)
if _eval9_spec is None or _eval9_spec.loader is None:
    raise ImportError(f"Could not import {_EVAL9_PATH}")
_eval9_mod = importlib.util.module_from_spec(_eval9_spec)
_eval9_spec.loader.exec_module(_eval9_mod)

build_scored_from_sentence_scores = _eval9_mod.build_scored_from_sentence_scores
compute_metrics_from_scored = _eval9_mod.compute_metrics_from_scored
load_crowspairs_df = _eval9_mod.load_crowspairs_df
make_crowspairs_eval_pairs = _eval9_mod.make_crowspairs_eval_pairs
build_stereoset_sentence_records = _eval9_mod.build_stereoset_sentence_records
flatten_stereoset_examples = _eval9_mod.flatten_stereoset_examples
load_stereoset_data = _eval9_mod.load_stereoset_data
stereoset_score = _eval9_mod.stereoset_score


SEED = 42
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

DEBIASED_INSTANCES_ROOT = "/scratch/craj/diy/outputs/1_generations/debiased_instances"
PAIR_TO_VERSION = {
    "opinion": "opinion_version",
    "action": "action_version",
    "event": "event_version",
}

COL_MAP = {
    "opinion": ("opinion_version", "debiased_opinion_version"),
    "action": ("action_version", "debiased_action_version"),
    "event": ("event_version", "debiased_event_version"),
}

STRATEGY_ALIASES = {
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
}

ALL_STRATEGIES = [
    "stereotype_replacement",
    "counter_imaging",
    "individuating",
    "perspective_taking",
    "positive_contact",
]

ALL_VERSIONS = ["opinion", "action", "event"]

STRATEGY_DESCRIPTIONS = {
    "stereotype_replacement": "replace stereotypes with specific, fair, individualized alternatives",
    "counter_imaging": "imagine clear counter-stereotypical examples to challenge biased associations",
    "individuating": "focus on the person as an individual rather than group assumptions",
    "perspective_taking": "adopt the perspective of the targeted person and reframe assumptions",
    "positive_contact": "recall positive intergroup interaction and generalize that experience",
}

DEFAULT_CROWS_PATH = "/scratch/craj/diy/data/crows_pairs_anonymized.csv"
DEFAULT_STEREOSET_PATH = "/scratch/craj/diy/data/stereoset/dev.json"
DEFAULT_BBQ_PROCESSED = "/scratch/craj/diy/data/processed_bbq_all.csv"

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


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def parse_csv_arg(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def normalize_strategy(raw: str) -> str:
    key = str(raw).strip().lower()
    if key not in STRATEGY_ALIASES:
        raise ValueError(f"Unknown strategy `{raw}`")
    return STRATEGY_ALIASES[key]


def parse_strategies(raw: str) -> List[str]:
    if str(raw).strip().lower() == "all":
        return list(ALL_STRATEGIES)
    vals = parse_csv_arg(raw)
    out = [normalize_strategy(v) for v in vals]
    return sorted(set(out))


def parse_versions(raw: str) -> List[str]:
    if str(raw).strip().lower() == "all":
        return list(ALL_VERSIONS)
    vals = parse_csv_arg(raw)
    out: List[str] = []
    for v in vals:
        vv = v.strip().lower()
        if vv.endswith("_version"):
            vv = vv.replace("_version", "")
        if vv not in COL_MAP:
            raise ValueError(f"Unknown version `{v}`")
        out.append(vv)
    return sorted(set(out))


def resolve_merged_path(strategy: str, version: str) -> str:
    return os.path.join(DEBIASED_INSTANCES_ROOT, f"{strategy}_{PAIR_TO_VERSION[version]}_merged.jsonl")


def make_train_prompt(input_text: str) -> str:
    return (
        "Debias the content below while preserving meaning.\n\n"
        f"Content:\n{input_text.strip()}\n\n"
        "Debiased rewrite:"
    )


def load_model_and_tokenizer(
    model_key: str,
    model_source: Optional[str] = None,
    quantized: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model `{model_key}`")

    info = AVAILABLE_MODELS[model_key]
    source = model_source or info["model"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=info["cache_dir"])
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(info["model"], cache_dir=info["cache_dir"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if quantized:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            quantization_config=quant_cfg,
            device_map="auto",
            cache_dir=info["cache_dir"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=info["cache_dir"],
        )
        if not torch.cuda.is_available():
            model = model.to(DEFAULT_DEVICE)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer, source


def build_initial_soft_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    strategy_keys: Sequence[str],
    prompt_length: int,
) -> torch.Tensor:
    emb_layer = model.get_input_embeddings()
    emb_weight = emb_layer.weight.detach()
    hidden_size = emb_weight.shape[1]

    rows = []
    for strategy in strategy_keys:
        desc = STRATEGY_DESCRIPTIONS[strategy]
        ids = tokenizer(desc, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = [tokenizer.eos_token_id]

        idx = torch.tensor(ids, dtype=torch.long, device=emb_weight.device)
        vecs = emb_weight.index_select(0, idx)

        if vecs.shape[0] >= prompt_length:
            vecs = vecs[:prompt_length]
        else:
            pad_rows = prompt_length - vecs.shape[0]
            last = vecs[-1:].repeat(pad_rows, 1)
            vecs = torch.cat([vecs, last], dim=0)

        if vecs.shape != (prompt_length, hidden_size):
            raise RuntimeError("Soft prompt init shape mismatch")
        rows.append(vecs)

    return torch.stack(rows, dim=0).float().cpu()


def encode_train_example(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    target_text: str,
    max_length: int,
) -> Optional[Dict[str, List[int]]]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer eos_token_id is required")

    full_ids = prompt_ids + target_ids + [eos_id]
    if len(full_ids) > max_length:
        return None

    labels = [-100] * len(prompt_ids) + target_ids + [eos_id]
    attn = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attn,
        "labels": labels,
    }


@dataclass
class TrainRow:
    strategy: str
    input_text: str
    output_text: str


class EncodedSoftPromptDataset(Dataset):
    def __init__(self, rows: List[Dict], strategy_to_idx: Dict[str, int]):
        self.rows = rows
        self.strategy_to_idx = strategy_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
            "strategy_idx": self.strategy_to_idx[row["strategy"]],
        }


class SoftPromptWrapper(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        soft_prompt_tensor: torch.Tensor,
        strategy_keys: Sequence[str],
        trainable: bool,
    ):
        super().__init__()
        self.model = model
        self.strategy_keys = list(strategy_keys)
        self.strategy_to_idx = {k: i for i, k in enumerate(self.strategy_keys)}
        self.prompt_length = int(soft_prompt_tensor.shape[1])

        if trainable:
            self.soft_prompts = nn.Parameter(soft_prompt_tensor.clone())
        else:
            self.register_buffer("soft_prompts", soft_prompt_tensor.clone(), persistent=True)

    @property
    def hidden_size(self) -> int:
        return int(self.soft_prompts.shape[-1])

    def strategy_index(self, strategy: str) -> int:
        s = normalize_strategy(strategy)
        if s not in self.strategy_to_idx:
            raise ValueError(f"Strategy `{strategy}` not in checkpoint: {self.strategy_keys}")
        return self.strategy_to_idx[s]

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_layer = self.model.get_input_embeddings()
        token_embeds = emb_layer(input_ids)

        soft_bank = self.soft_prompts.to(token_embeds.device, dtype=token_embeds.dtype)
        soft = soft_bank.index_select(0, strategy_idx).to(token_embeds.dtype)

        inputs_embeds = torch.cat([soft, token_embeds], dim=1)
        soft_mask = torch.ones(
            (attention_mask.size(0), self.prompt_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention = torch.cat([soft_mask, attention_mask], dim=1)
        return inputs_embeds, full_attention

    def loss_on_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        strategy_idx: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds, full_attention = self._build_inputs_embeds(input_ids, attention_mask, strategy_idx)
        soft_label_pad = torch.full(
            (labels.size(0), self.prompt_length),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        full_labels = torch.cat([soft_label_pad, labels], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=full_labels,
        )
        return out.loss

    @torch.no_grad()
    def sequence_logprob_batch(
        self,
        tokenizer: AutoTokenizer,
        sentences: List[str],
        strategy: str,
        batch_size: int,
        max_length: int,
    ) -> List[float]:
        sid = self.strategy_index(strategy)
        logprobs: List[float] = []

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc["attention_mask"].to(self.model.device)
            strategy_idx = torch.full(
                (input_ids.size(0),),
                sid,
                dtype=torch.long,
                device=self.model.device,
            )

            inputs_embeds, full_attention = self._build_inputs_embeds(input_ids, attention_mask, strategy_idx)
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=full_attention)
            logits = outputs.logits

            soft_pad_ids = torch.full(
                (input_ids.size(0), self.prompt_length),
                -100,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            full_target_ids = torch.cat([soft_pad_ids, input_ids], dim=1)

            soft_mask = torch.zeros(
                (attention_mask.size(0), self.prompt_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_target_mask = torch.cat([soft_mask, attention_mask], dim=1)

            shift_logits = logits[:, :-1, :]
            shift_labels = full_target_ids[:, 1:]
            shift_mask = full_target_mask[:, 1:] * (shift_labels != -100)

            safe_labels = shift_labels.clone()
            safe_labels[safe_labels == -100] = 0

            lp = torch.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            token_lp = token_lp * shift_mask
            seq_lp = token_lp.sum(dim=1).detach().cpu().tolist()
            logprobs.extend([float(x) for x in seq_lp])

        return logprobs

    @torch.no_grad()
    def completion_logprobs(
        self,
        tokenizer: AutoTokenizer,
        prompt: str,
        completions: Sequence[str],
        strategy: str,
        max_length: int,
    ) -> List[float]:
        sid = self.strategy_index(strategy)
        out: List[float] = []

        for comp in completions:
            prefix = prompt if prompt.endswith((" ", "\n", "\t")) else prompt + " "
            full = prefix + str(comp)
            boundary = len(prefix)

            enc = tokenizer(
                full,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc["attention_mask"].to(self.model.device)

            completion_token_positions = [
                i for i, (s, e) in enumerate(offsets)
                if int(e) > int(boundary)
            ]
            if not completion_token_positions:
                out.append(float("-inf"))
                continue

            strategy_idx = torch.tensor([sid], dtype=torch.long, device=self.model.device)
            inputs_embeds, full_attention = self._build_inputs_embeds(input_ids, attention_mask, strategy_idx)
            logits = self.model(inputs_embeds=inputs_embeds, attention_mask=full_attention).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            lp = torch.log_softmax(shift_logits, dim=-1)

            score = 0.0
            for pos in completion_token_positions:
                if pos == 0:
                    continue
                # inputs_embeds has `prompt_length` soft-prompt tokens prepended,
                # so logit at (prompt_length + pos - 1) predicts input_ids[pos].
                logit_idx = self.prompt_length + pos - 1
                tok = int(input_ids[0, pos].item())
                score += float(lp[0, logit_idx, tok].item())

            out.append(score)

        return out


def collate_train_batch(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)

    in_ids = []
    attn = []
    labels = []
    strat = []

    for row in batch:
        pad = max_len - len(row["input_ids"])
        in_ids.append([pad_token_id] * pad + row["input_ids"])
        attn.append([0] * pad + row["attention_mask"])
        labels.append([-100] * pad + row["labels"])
        strat.append(row["strategy_idx"])

    return {
        "input_ids": torch.tensor(in_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "strategy_idx": torch.tensor(strat, dtype=torch.long),
    }


def load_training_rows(
    strategies: Sequence[str],
    versions: Sequence[str],
) -> List[TrainRow]:
    rows: List[TrainRow] = []

    for strategy in strategies:
        for version in versions:
            path = resolve_merged_path(strategy, version)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing merged file: {path}")

            src_col, tgt_col = COL_MAP[version]
            df = pd.read_json(path, lines=True)
            if src_col not in df.columns or tgt_col not in df.columns:
                raise ValueError(f"Missing expected columns in {path}: {src_col}, {tgt_col}")

            tmp = df[[src_col, tgt_col]].dropna().copy()
            for _, r in tmp.iterrows():
                rows.append(
                    TrainRow(
                        strategy=strategy,
                        input_text=str(r[src_col]),
                        output_text=str(r[tgt_col]),
                    )
                )

    return rows


def split_rows(rows: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not rows:
        return [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)

    split = int(len(idx) * train_ratio)
    split = max(1, min(split, len(idx) - 1))

    train_idx = idx[:split]
    eval_idx = idx[split:]

    train_rows = [rows[int(i)] for i in train_idx]
    eval_rows = [rows[int(i)] for i in eval_idx]
    return train_rows, eval_rows


def save_soft_prompt_checkpoint(
    output_dir: str,
    wrapper: SoftPromptWrapper,
    config: Dict,
    metrics: Dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    prompt_tensor = wrapper.soft_prompts.detach().float().cpu()
    torch.save(prompt_tensor, os.path.join(output_dir, "soft_prompts.pt"))

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_soft_prompt_runtime(
    soft_prompt_dir: str,
    force_model_key: Optional[str] = None,
    force_model_path: Optional[str] = None,
) -> Tuple[SoftPromptWrapper, AutoTokenizer, Dict]:
    cfg_path = os.path.join(soft_prompt_dir, "config.json")
    pt_path = os.path.join(soft_prompt_dir, "soft_prompts.pt")
    if not os.path.exists(cfg_path) or not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing config.json or soft_prompts.pt in {soft_prompt_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_key = force_model_key or cfg["model_key"]
    model_path = force_model_path or cfg.get("model_source")

    model, tokenizer, resolved_source = load_model_and_tokenizer(
        model_key=model_key,
        model_source=model_path,
        quantized=True,
    )

    soft_prompts = torch.load(pt_path, map_location="cpu")
    strategy_keys = cfg["strategy_keys"]
    wrapper = SoftPromptWrapper(
        model=model,
        soft_prompt_tensor=soft_prompts,
        strategy_keys=strategy_keys,
        trainable=False,
    )
    wrapper.eval()

    cfg["resolved_model_source"] = resolved_source
    return wrapper, tokenizer, cfg


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    strategies = parse_strategies(args.strategies)
    versions = parse_versions(args.versions)

    print(f"[INFO] strategies={strategies}")
    print(f"[INFO] versions={versions}")

    model, tokenizer, model_source = load_model_and_tokenizer(
        model_key=args.model,
        model_source=args.model_path,
        quantized=(not args.no_quantized),
    )
    print(f"[INFO] loaded model source: {model_source}")

    soft_init = build_initial_soft_prompts(
        model=model,
        tokenizer=tokenizer,
        strategy_keys=strategies,
        prompt_length=args.prompt_length,
    )

    wrapper = SoftPromptWrapper(
        model=model,
        soft_prompt_tensor=soft_init,
        strategy_keys=strategies,
        trainable=(args.soft_prompt_mode == "learnable"),
    )

    raw_rows = load_training_rows(strategies=strategies, versions=versions)
    if not raw_rows:
        raise ValueError("No rows loaded for training")

    if args.max_samples is not None:
        take = min(args.max_samples, len(raw_rows))
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(raw_rows), size=take, replace=False)
        raw_rows = [raw_rows[int(i)] for i in idx]

    print(f"[INFO] train rows before tokenization: {len(raw_rows)}")

    encoded_rows: List[Dict] = []
    dropped = 0
    for row in raw_rows:
        prompt = make_train_prompt(row.input_text)
        enc = encode_train_example(
            tokenizer=tokenizer,
            prompt_text=prompt,
            target_text=row.output_text,
            max_length=args.max_length,
        )
        if enc is None:
            dropped += 1
            continue
        enc["strategy"] = row.strategy
        encoded_rows.append(enc)

    if not encoded_rows:
        raise ValueError("All training rows dropped after tokenization/truncation")

    print(f"[INFO] encoded rows={len(encoded_rows)}, dropped={dropped}")

    train_rows, eval_rows = split_rows(encoded_rows, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[INFO] split: train={len(train_rows)}, eval={len(eval_rows)}")

    ds_train = EncodedSoftPromptDataset(train_rows, wrapper.strategy_to_idx)
    ds_eval = EncodedSoftPromptDataset(eval_rows, wrapper.strategy_to_idx)

    collator = lambda b: collate_train_batch(b, pad_token_id=tokenizer.pad_token_id)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dl_eval = DataLoader(ds_eval, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    metrics: Dict[str, object] = {
        "soft_prompt_mode": args.soft_prompt_mode,
        "train_steps": 0,
        "epochs": args.epochs,
        "train_loss_history": [],
        "eval_loss_history": [],
    }

    if args.soft_prompt_mode == "learnable":
        optimizer = torch.optim.AdamW([wrapper.soft_prompts], lr=args.lr, weight_decay=args.weight_decay)

        global_step = 0
        for epoch in range(args.epochs):
            wrapper.train()
            run_losses = []
            pbar = tqdm(dl_train, desc=f"train epoch {epoch + 1}/{args.epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(wrapper.model.device)
                attention_mask = batch["attention_mask"].to(wrapper.model.device)
                labels = batch["labels"].to(wrapper.model.device)
                strategy_idx = batch["strategy_idx"].to(wrapper.model.device)

                loss = wrapper.loss_on_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    strategy_idx=strategy_idx,
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                lv = float(loss.detach().cpu().item())
                run_losses.append(lv)
                global_step += 1
                pbar.set_postfix({"loss": f"{lv:.4f}"})

            train_loss = float(np.mean(run_losses)) if run_losses else float("nan")

            wrapper.eval()
            ev_losses = []
            with torch.no_grad():
                for batch in dl_eval:
                    input_ids = batch["input_ids"].to(wrapper.model.device)
                    attention_mask = batch["attention_mask"].to(wrapper.model.device)
                    labels = batch["labels"].to(wrapper.model.device)
                    strategy_idx = batch["strategy_idx"].to(wrapper.model.device)

                    loss = wrapper.loss_on_batch(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        strategy_idx=strategy_idx,
                    )
                    ev_losses.append(float(loss.detach().cpu().item()))

            eval_loss = float(np.mean(ev_losses)) if ev_losses else float("nan")
            print(f"[INFO] epoch {epoch + 1}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
            metrics["train_loss_history"].append(train_loss)
            metrics["eval_loss_history"].append(eval_loss)

        metrics["train_steps"] = global_step
    else:
        print("[INFO] soft_prompt_mode=fixed -> no optimization, evaluating only")
        wrapper.eval()
        ev_losses = []
        with torch.no_grad():
            for batch in tqdm(dl_eval, desc="fixed eval"):
                input_ids = batch["input_ids"].to(wrapper.model.device)
                attention_mask = batch["attention_mask"].to(wrapper.model.device)
                labels = batch["labels"].to(wrapper.model.device)
                strategy_idx = batch["strategy_idx"].to(wrapper.model.device)

                loss = wrapper.loss_on_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    strategy_idx=strategy_idx,
                )
                ev_losses.append(float(loss.detach().cpu().item()))

        eval_loss = float(np.mean(ev_losses)) if ev_losses else float("nan")
        metrics["eval_loss_history"].append(eval_loss)
        print(f"[INFO] fixed eval_loss={eval_loss:.4f}")

    cfg = {
        "model_key": args.model,
        "model_source": model_source,
        "prompt_length": args.prompt_length,
        "soft_prompt_mode": args.soft_prompt_mode,
        "strategy_keys": strategies,
        "versions": versions,
        "max_length": args.max_length,
        "seed": args.seed,
    }

    save_soft_prompt_checkpoint(args.output_dir, wrapper=wrapper, config=cfg, metrics=metrics)
    print(f"[INFO] saved soft prompt checkpoint -> {args.output_dir}")


def run_eval_crows(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    wrapper, tokenizer, _ = load_soft_prompt_runtime(
        soft_prompt_dir=args.soft_prompt_dir,
        force_model_key=args.model,
        force_model_path=args.model_path,
    )
    strategy = normalize_strategy(args.strategy)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    raw_df = load_crowspairs_df(args.data_path, limit=args.limit)
    pair_df = make_crowspairs_eval_pairs(raw_df)
    sent_more = pair_df["sent_more"].tolist()
    sent_less = pair_df["sent_less"].tolist()

    print("[INFO] scoring sent_more sentences...")
    sent_more_scores = wrapper.sequence_logprob_batch(
        tokenizer=tokenizer,
        sentences=sent_more,
        strategy=strategy,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("[INFO] scoring sent_less sentences...")
    sent_less_scores = wrapper.sequence_logprob_batch(
        tokenizer=tokenizer,
        sentences=sent_less,
        strategy=strategy,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    scored = build_scored_from_sentence_scores(
        pair_df,
        sent_more_scores=sent_more_scores,
        sent_less_scores=sent_less_scores,
    )
    scored["soft_prompt_strategy"] = strategy

    tag = sanitize_tag(args.model_tag or f"{os.path.basename(args.soft_prompt_dir)}_{strategy}")

    pairs_out = os.path.join(args.output_dir, f"crows_pairs_scored_{tag}.csv")
    scored.to_csv(pairs_out, index=False)

    overall, by_bias = compute_metrics_from_scored(
        scored,
        model_name=tag,
        extra_fields={"strategy": strategy},
    )
    overall_out = os.path.join(args.results_dir, f"crows_pairs_metrics_overall_{tag}.csv")
    by_bias_out = os.path.join(args.results_dir, f"crows_pairs_metrics_by_bias_{tag}.csv")
    overall.to_csv(overall_out, index=False)
    by_bias.to_csv(by_bias_out, index=False)

    print(f"[INFO] saved -> {pairs_out}")
    print(f"[INFO] saved -> {overall_out}")
    print(f"[INFO] saved -> {by_bias_out}")


def nested_results_to_rows(results: Dict, model_name: str, strategy: str) -> List[Dict]:
    rows: List[Dict] = []
    for split_key in ["intrasentence", "intersentence"]:
        for domain, vals in results.get(split_key, {}).items():
            rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "split": split_key,
                    "domain": domain,
                    "Count": vals.get("Count", 0),
                    "LM Score": vals.get("LM Score", 0.0),
                    "SS Score": vals.get("SS Score", 0.0),
                    "ICAT Score": vals.get("ICAT Score", 0.0),
                }
            )

    ov = results.get("overall", {})
    rows.append(
        {
            "model": model_name,
            "strategy": strategy,
            "split": "overall",
            "domain": "overall",
            "Count": ov.get("Count", 0),
            "LM Score": ov.get("LM Score", 0.0),
            "SS Score": ov.get("SS Score", 0.0),
            "ICAT Score": ov.get("ICAT Score", 0.0),
        }
    )
    return rows


def run_eval_stereoset(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    wrapper, tokenizer, _ = load_soft_prompt_runtime(
        soft_prompt_dir=args.soft_prompt_dir,
        force_model_key=args.model,
        force_model_path=args.model_path,
    )
    strategy = normalize_strategy(args.strategy)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    data = load_stereoset_data(args.data_path)
    examples = flatten_stereoset_examples(data, split=args.split, bias_type=args.bias_type, limit=args.limit)
    if not examples:
        raise ValueError("No StereoSet examples found after filtering")

    records = build_stereoset_sentence_records(examples)
    texts = [r["sentence"] for r in records]
    sent_ids = [r["sentence_id"] for r in records]
    sent_splits = [r["split"] for r in records]

    scores = wrapper.sequence_logprob_batch(
        tokenizer=tokenizer,
        sentences=texts,
        strategy=strategy,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    id2score = {sid: float(sc) for sid, sc in zip(sent_ids, scores)}

    preds_json = {"intrasentence": [], "intersentence": []}
    for sid, sp in zip(sent_ids, sent_splits):
        preds_json[sp].append({"id": sid, "score": id2score[sid]})

    for rec in records:
        rec["score"] = id2score.get(rec["sentence_id"], np.nan)
        rec["soft_prompt_strategy"] = strategy

    tag = sanitize_tag(args.model_tag or f"{os.path.basename(args.soft_prompt_dir)}_{strategy}")
    records_df = pd.DataFrame(records)

    results = stereoset_score(examples, id2score)
    rows_df = pd.DataFrame(nested_results_to_rows(results, model_name=tag, strategy=strategy))

    scores_out = os.path.join(args.output_dir, f"stereoset_sentence_scores_{tag}.csv")
    preds_out = os.path.join(args.output_dir, f"stereoset_predictions_{tag}.json")
    metrics_json = os.path.join(args.results_dir, f"stereoset_metrics_{tag}.json")
    metrics_csv = os.path.join(args.results_dir, f"stereoset_metrics_{tag}.csv")

    records_df.to_csv(scores_out, index=False)
    with open(preds_out, "w", encoding="utf-8") as f:
        json.dump(preds_json, f, indent=2)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    rows_df.to_csv(metrics_csv, index=False)

    print(f"[INFO] saved -> {scores_out}")
    print(f"[INFO] saved -> {preds_out}")
    print(f"[INFO] saved -> {metrics_json}")
    print(f"[INFO] saved -> {metrics_csv}")


def format_bbq_prompt(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:
    return f"{context}\n{question}\nA. {ans0}\nB. {ans1}\nC. {ans2}\nAnswer:"


def run_infer_bbq(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    if args.source_file not in VALID_BBQ_SOURCE_FILES:
        raise ValueError(f"Unknown source_file `{args.source_file}`")

    wrapper, tokenizer, _ = load_soft_prompt_runtime(
        soft_prompt_dir=args.soft_prompt_dir,
        force_model_key=args.model,
        force_model_path=args.model_path,
    )
    strategy = normalize_strategy(args.strategy)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df["answer_info"] = df["answer_info"].apply(ast.literal_eval)
    df = df[df["source_file"] == args.source_file].reset_index(drop=True)

    if args.limit:
        df = df.iloc[: args.limit].copy()

    print(f"[INFO] loaded {len(df)} rows for {args.source_file}")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="bbq inference"):
        options = [row["ans0"], row["ans1"], row["ans2"]]
        prompt = format_bbq_prompt(row["context"], row["question"], *options)

        logps = wrapper.completion_logprobs(
            tokenizer=tokenizer,
            prompt=prompt,
            completions=options,
            strategy=strategy,
            max_length=args.max_length,
        )

        if all(np.isneginf(x) for x in logps):
            probs = np.array([1.0 / len(options)] * len(options), dtype=np.float64)
        else:
            m = np.max(logps)
            exps = np.exp(np.array(logps, dtype=np.float64) - float(m))
            denom = float(exps.sum()) if float(exps.sum()) > 0 else 1.0
            probs = exps / denom

        pred_idx = int(np.argmax(probs))
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
                "option_probs": {chr(65 + k): float(p) for k, p in enumerate(probs.tolist())},
                "option_logprobs": {chr(65 + k): float(lp) for k, lp in enumerate(logps)},
                "soft_prompt_strategy": strategy,
                "soft_prompt_dir": args.soft_prompt_dir,
            }
        )

    tag = sanitize_tag(args.model_tag or f"{os.path.basename(args.soft_prompt_dir)}_{strategy}")
    out_csv = os.path.join(args.output_dir, f"bbq_preds_{tag}_{args.source_file.replace('.jsonl', '')}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] saved -> {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Soft strategy-prompt experiments (fixed/learnable embeddings)")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train or materialize soft strategy prompts")
    p_train.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default="llama_8b")
    p_train.add_argument("--model_path", type=str, default=None)
    p_train.add_argument("--strategies", type=str, default="all")
    p_train.add_argument("--versions", type=str, default="all")
    p_train.add_argument("--soft_prompt_mode", choices=["learnable", "fixed"], default="learnable")
    p_train.add_argument("--prompt_length", type=int, default=16)
    p_train.add_argument("--max_length", type=int, default=512)
    p_train.add_argument("--max_samples", type=int, default=None)
    p_train.add_argument("--train_ratio", type=float, default=0.9)
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=5e-3)
    p_train.add_argument("--weight_decay", type=float, default=0.0)
    p_train.add_argument("--seed", type=int, default=SEED)
    p_train.add_argument("--no_quantized", action="store_true")
    p_train.add_argument("--output_dir", type=str, required=True)

    p_crows = sub.add_parser("eval_crows", help="CrowS-Pairs eval with soft prompts")
    p_crows.add_argument("--soft_prompt_dir", type=str, required=True)
    p_crows.add_argument("--strategy", type=str, required=True)
    p_crows.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default=None)
    p_crows.add_argument("--model_path", type=str, default=None)
    p_crows.add_argument("--data_path", type=str, default=DEFAULT_CROWS_PATH)
    p_crows.add_argument("--batch_size", type=int, default=4)
    p_crows.add_argument("--max_length", type=int, default=512)
    p_crows.add_argument("--limit", type=int, default=None)
    p_crows.add_argument("--model_tag", type=str, default=None)
    p_crows.add_argument("--seed", type=int, default=SEED)
    p_crows.add_argument("--output_dir", type=str, required=True)
    p_crows.add_argument("--results_dir", type=str, required=True)

    p_st = sub.add_parser("eval_stereoset", help="StereoSet eval with soft prompts")
    p_st.add_argument("--soft_prompt_dir", type=str, required=True)
    p_st.add_argument("--strategy", type=str, required=True)
    p_st.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default=None)
    p_st.add_argument("--model_path", type=str, default=None)
    p_st.add_argument("--data_path", type=str, default=DEFAULT_STEREOSET_PATH)
    p_st.add_argument("--split", choices=["all", "intrasentence", "intersentence"], default="all")
    p_st.add_argument("--bias_type", type=str, default=None)
    p_st.add_argument("--batch_size", type=int, default=4)
    p_st.add_argument("--max_length", type=int, default=512)
    p_st.add_argument("--limit", type=int, default=None)
    p_st.add_argument("--model_tag", type=str, default=None)
    p_st.add_argument("--seed", type=int, default=SEED)
    p_st.add_argument("--output_dir", type=str, required=True)
    p_st.add_argument("--results_dir", type=str, required=True)

    p_bbq = sub.add_parser("infer_bbq", help="BBQ inference with soft prompts")
    p_bbq.add_argument("--soft_prompt_dir", type=str, required=True)
    p_bbq.add_argument("--strategy", type=str, required=True)
    p_bbq.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default=None)
    p_bbq.add_argument("--model_path", type=str, default=None)
    p_bbq.add_argument("--data_path", type=str, default=DEFAULT_BBQ_PROCESSED)
    p_bbq.add_argument("--source_file", type=str, required=True, choices=VALID_BBQ_SOURCE_FILES)
    p_bbq.add_argument("--max_length", type=int, default=512)
    p_bbq.add_argument("--limit", type=int, default=None)
    p_bbq.add_argument("--model_tag", type=str, default=None)
    p_bbq.add_argument("--seed", type=int, default=SEED)
    p_bbq.add_argument("--output_dir", type=str, required=True)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval_crows":
        if args.model is None:
            args.model = "llama_8b"
        run_eval_crows(args)
    elif args.command == "eval_stereoset":
        if args.model is None:
            args.model = "llama_8b"
        run_eval_stereoset(args)
    elif args.command == "infer_bbq":
        if args.model is None:
            args.model = "llama_8b"
        run_infer_bbq(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
