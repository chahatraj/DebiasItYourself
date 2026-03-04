#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


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

DEFAULT_OUTPUT_ROOT = "/scratch/craj/diy/outputs/10_additional_benchmarks"
DEFAULT_RESULTS_ROOT = "/scratch/craj/diy/results/10_additional_benchmarks"

UNQOVER_DS = {
    "gender": "hirundo-io/unqover-gender",
    "race": "hirundo-io/unqover-race",
    "religion": "hirundo-io/unqover-religion",
    "nationality": "hirundo-io/unqover-nationality",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def safe_mean(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def load_model_and_tokenizer(
    model_key: str,
    model_path: Optional[str] = None,
    quantized: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model `{model_key}`")

    info = AVAILABLE_MODELS[model_key]
    source = model_path or info["model"]

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

    model.eval()
    return model, tokenizer, source


def sequence_logprob_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
) -> List[float]:
    out: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        lp = torch.log_softmax(shift_logits, dim=-1)
        token_lp = lp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp * shift_mask
        seq_lp = token_lp.sum(dim=1).detach().cpu().tolist()
        out.extend([float(x) for x in seq_lp])

    return out


def completion_logprobs_for_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    options: Sequence[str],
    max_length: int,
    option_batch_size: int,
) -> List[float]:
    prefix = prompt if prompt.endswith((" ", "\n", "\t")) else prompt + " "
    boundary = len(prefix)
    options = [str(x) for x in options]

    all_scores: List[float] = []
    for start in range(0, len(options), option_batch_size):
        chunk = options[start : start + option_batch_size]
        texts = [prefix + opt for opt in chunk]

        enc = tokenizer(
            texts,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = enc.pop("offset_mapping")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        lp = torch.log_softmax(shift_logits, dim=-1)

        for i in range(input_ids.size(0)):
            offs = offsets[i].tolist()
            completion_positions = [
                j
                for j, (s, e) in enumerate(offs)
                if int(e) > int(boundary) and int(attention_mask[i, j].item()) == 1
            ]

            if not completion_positions:
                all_scores.append(float("-inf"))
                continue

            score = 0.0
            for pos in completion_positions:
                if pos == 0:
                    continue
                idx = pos - 1
                tok = int(shift_labels[i, idx].item())
                score += float(lp[i, idx, tok].item())
            all_scores.append(score)

    return all_scores


def probs_from_logps(logps: Sequence[float]) -> np.ndarray:
    arr = np.asarray(logps, dtype=np.float64)
    if np.all(np.isneginf(arr)):
        return np.ones_like(arr) / max(1, len(arr))
    m = float(np.max(arr))
    exps = np.exp(arr - m)
    denom = float(exps.sum())
    if denom <= 0.0:
        return np.ones_like(arr) / max(1, len(arr))
    return exps / denom


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def eval_bold(args: argparse.Namespace, model, tokenizer, tag: str) -> None:
    ds = load_dataset("AmazonScience/bold")["train"]

    rows = []
    for ex in ds:
        prompts = ex.get("prompts") or []
        wikis = ex.get("wikipedia") or []
        if not prompts or not wikis:
            continue
        prompt = str(prompts[0]).strip()
        continuation = str(wikis[0]).strip()
        if not prompt or not continuation:
            continue
        rows.append(
            {
                "domain": str(ex.get("domain", "")),
                "category": str(ex.get("category", "")),
                "name": str(ex.get("name", "")),
                "prompt": prompt,
                "continuation": continuation,
            }
        )

    df = pd.DataFrame(rows)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    prefixes = [x if x.endswith((" ", "\n", "\t")) else x + " " for x in df["prompt"].tolist()]
    full_texts = [p + c for p, c in zip(prefixes, df["continuation"].tolist())]
    prompt_only = prefixes

    full_lp = sequence_logprob_batch(model, tokenizer, full_texts, args.batch_size, args.max_length)
    prompt_lp = sequence_logprob_batch(model, tokenizer, prompt_only, args.batch_size, args.max_length)

    continuation_toks = [
        len(tokenizer(c, add_special_tokens=False)["input_ids"]) or 1
        for c in df["continuation"].tolist()
    ]

    df["full_logprob"] = full_lp
    df["prompt_logprob"] = prompt_lp
    df["continuation_logprob"] = df["full_logprob"] - df["prompt_logprob"]
    df["continuation_tokens"] = continuation_toks
    df["avg_logprob_per_token"] = df["continuation_logprob"] / df["continuation_tokens"]

    overall = {
        "model": tag,
        "benchmark": "bold",
        "n": int(len(df)),
        "mean_avg_logprob_per_token": safe_mean(df["avg_logprob_per_token"].tolist()),
        "ppl_proxy": float(np.exp(-safe_mean(df["avg_logprob_per_token"].tolist()))),
    }

    by_domain = (
        df.groupby("domain", as_index=False)
        .agg(
            n=("domain", "size"),
            mean_avg_logprob_per_token=("avg_logprob_per_token", "mean"),
        )
    )
    by_domain["model"] = tag
    by_domain["benchmark"] = "bold"

    out_dir = os.path.join(args.output_dir, "bold", tag)
    res_dir = os.path.join(args.results_dir, "bold", tag)
    ensure_dirs(out_dir, res_dir)

    df.to_csv(os.path.join(out_dir, f"bold_scored_{tag}.csv"), index=False)
    pd.DataFrame([overall]).to_csv(os.path.join(res_dir, f"bold_metrics_overall_{tag}.csv"), index=False)
    by_domain.to_csv(os.path.join(res_dir, f"bold_metrics_by_domain_{tag}.csv"), index=False)


def eval_honest(args: argparse.Namespace, model, tokenizer, tag: str) -> None:
    ds = load_dataset("MilaNLProc/honest", args.honest_config)["honest"]

    rows = []
    for ex in ds:
        tmpl = str(ex.get("template_masked", ""))
        prompt = tmpl.replace("[M]", "").strip()
        if prompt and not prompt.endswith((" ", "\n", "\t")):
            prompt = prompt + " "
        rows.append(
            {
                "template_masked": tmpl,
                "raw": str(ex.get("raw", "")),
                "identity": str(ex.get("identity", "")),
                "category": str(ex.get("category", "")),
                "type": str(ex.get("type", "")),
                "prompt": prompt,
            }
        )

    df = pd.DataFrame(rows)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    prompts = df["prompt"].tolist()
    completions: List[str] = []

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="HONEST next-token"):
        batch = prompts[start : start + args.batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        lengths = attention_mask.sum(dim=1) - 1
        next_ids = []
        for i in range(input_ids.size(0)):
            idx = int(lengths[i].item())
            next_ids.append(int(torch.argmax(logits[i, idx, :]).item()))

        decoded = tokenizer.batch_decode(next_ids, skip_special_tokens=True)
        for text in decoded:
            m = re.search(r"[A-Za-z][A-Za-z_\-]*", text.strip())
            completions.append(m.group(0).lower() if m else text.strip().lower())

    df["completion"] = completions

    unique_ratio = float(df["completion"].nunique() / max(1, len(df)))
    top_counts = df["completion"].value_counts()
    top_comp = str(top_counts.index[0]) if len(top_counts) else ""
    top_share = float(top_counts.iloc[0] / max(1, len(df))) if len(top_counts) else 0.0

    raw_group = df.groupby("raw")["completion"].nunique().reset_index(name="n_unique_completion")
    identity_sensitive_pct = float((raw_group["n_unique_completion"] > 1).mean()) if len(raw_group) else 0.0

    overall = {
        "model": tag,
        "benchmark": "honest",
        "config": args.honest_config,
        "n": int(len(df)),
        "unique_completion_ratio": unique_ratio,
        "top_completion": top_comp,
        "top_completion_share": top_share,
        "identity_sensitive_template_pct": identity_sensitive_pct,
    }

    by_category = (
        df.groupby("category", as_index=False)
        .agg(
            n=("category", "size"),
            unique_completion_ratio=("completion", lambda s: float(s.nunique() / max(1, len(s)))),
        )
    )
    by_category["model"] = tag
    by_category["benchmark"] = "honest"
    by_category["config"] = args.honest_config

    out_dir = os.path.join(args.output_dir, "honest", tag)
    res_dir = os.path.join(args.results_dir, "honest", tag)
    ensure_dirs(out_dir, res_dir)

    df.to_csv(os.path.join(out_dir, f"honest_completions_{tag}.csv"), index=False)
    raw_group.to_csv(os.path.join(out_dir, f"honest_template_variation_{tag}.csv"), index=False)
    pd.DataFrame([overall]).to_csv(os.path.join(res_dir, f"honest_metrics_overall_{tag}.csv"), index=False)
    by_category.to_csv(os.path.join(res_dir, f"honest_metrics_by_category_{tag}.csv"), index=False)


def _normalize_wb_docid(docid: str) -> str:
    x = str(docid)
    x = x.replace("/stereotype/", "/")
    x = x.replace("/not_stereotype/", "/")
    return x


def _wb_sentence(tokens: Sequence[str]) -> str:
    text = " ".join(str(t) for t in tokens)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
    text = text.replace(" ?", "?").replace(" !", "!")
    return text


def eval_winobias(args: argparse.Namespace, model, tokenizer, tag: str) -> None:
    rows_all = []

    for wb_type in ["type1", "type2"]:
        ds_pro = load_dataset("uclanlp/wino_bias", f"{wb_type}_pro")["test"]
        ds_anti = load_dataset("uclanlp/wino_bias", f"{wb_type}_anti")["test"]

        pro_map = {_normalize_wb_docid(x["document_id"]): x for x in ds_pro}
        anti_map = {_normalize_wb_docid(x["document_id"]): x for x in ds_anti}
        keys = sorted(set(pro_map.keys()) & set(anti_map.keys()))

        stereo_texts = []
        anti_texts = []
        key_rows = []
        for k in keys:
            pro = pro_map[k]
            anti = anti_map[k]
            s_txt = _wb_sentence(pro["tokens"])
            a_txt = _wb_sentence(anti["tokens"])
            stereo_texts.append(s_txt)
            anti_texts.append(a_txt)
            key_rows.append({"pair_key": k, "type": wb_type, "stereotype_sentence": s_txt, "anti_sentence": a_txt})

        lp_s = sequence_logprob_batch(model, tokenizer, stereo_texts, args.batch_size, args.max_length)
        lp_a = sequence_logprob_batch(model, tokenizer, anti_texts, args.batch_size, args.max_length)

        for r, s, a in zip(key_rows, lp_s, lp_a):
            r["stereo_logprob"] = float(s)
            r["anti_logprob"] = float(a)
            r["stereo_preferred"] = int(float(s) > float(a))
            rows_all.append(r)

    df = pd.DataFrame(rows_all)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    overall = {
        "model": tag,
        "benchmark": "winobias",
        "n_pairs": int(len(df)),
        "stereotype_preference_pct": float(df["stereo_preferred"].mean() * 100.0),
    }

    by_type = (
        df.groupby("type", as_index=False)
        .agg(n_pairs=("type", "size"), stereotype_preference_pct=("stereo_preferred", lambda s: float(s.mean() * 100.0)))
    )
    by_type["model"] = tag
    by_type["benchmark"] = "winobias"

    out_dir = os.path.join(args.output_dir, "winobias", tag)
    res_dir = os.path.join(args.results_dir, "winobias", tag)
    ensure_dirs(out_dir, res_dir)

    df.to_csv(os.path.join(out_dir, f"winobias_pairs_{tag}.csv"), index=False)
    pd.DataFrame([overall]).to_csv(os.path.join(res_dir, f"winobias_metrics_overall_{tag}.csv"), index=False)
    by_type.to_csv(os.path.join(res_dir, f"winobias_metrics_by_type_{tag}.csv"), index=False)


def eval_winogender(args: argparse.Namespace, model, tokenizer, tag: str) -> None:
    ds_all = load_dataset("oskarvanderwal/winogender", "all")["test"]
    ds_gotcha = load_dataset("oskarvanderwal/winogender", "gotcha")["test"]
    gotcha_ids = set(str(x["sentid"]) for x in ds_gotcha)

    rows = []
    for ex in ds_all:
        sentid = str(ex.get("sentid", ""))
        sentence = str(ex.get("sentence", ""))
        pronoun = str(ex.get("pronoun", ""))
        occupation = str(ex.get("occupation", ""))
        participant = str(ex.get("participant", ""))
        target = str(ex.get("target", ""))
        gender = str(ex.get("gender", ""))

        prompt = (
            f"Sentence: {sentence}\n"
            f"Question: In the sentence above, who does the pronoun '{pronoun}' refer to?\n"
            f"A. {occupation}\n"
            f"B. {participant}\n"
            "Answer:"
        )
        logps = completion_logprobs_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            options=["A", "B"],
            max_length=args.max_length,
            option_batch_size=args.option_batch_size,
        )
        probs = probs_from_logps(logps)
        pred_idx = int(np.argmax(probs))

        if target == occupation:
            gold_idx = 0
        elif target == participant:
            gold_idx = 1
        else:
            gold_idx = int(ex.get("label", 0))

        rows.append(
            {
                "sentid": sentid,
                "sentence": sentence,
                "gender": gender,
                "occupation": occupation,
                "participant": participant,
                "target": target,
                "pred_index": pred_idx,
                "gold_index": gold_idx,
                "correct": int(pred_idx == gold_idx),
                "gotcha": int(sentid in gotcha_ids),
                "prob_A": float(probs[0]),
                "prob_B": float(probs[1]),
            }
        )

    df = pd.DataFrame(rows)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    overall = {
        "model": tag,
        "benchmark": "winogender",
        "n": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "accuracy_gotcha": float(df.loc[df["gotcha"] == 1, "correct"].mean()) if int((df["gotcha"] == 1).sum()) > 0 else float("nan"),
    }

    by_gender = (
        df.groupby("gender", as_index=False)
        .agg(n=("gender", "size"), accuracy=("correct", "mean"))
    )
    by_gender["model"] = tag
    by_gender["benchmark"] = "winogender"

    out_dir = os.path.join(args.output_dir, "winogender", tag)
    res_dir = os.path.join(args.results_dir, "winogender", tag)
    ensure_dirs(out_dir, res_dir)

    df.to_csv(os.path.join(out_dir, f"winogender_preds_{tag}.csv"), index=False)
    pd.DataFrame([overall]).to_csv(os.path.join(res_dir, f"winogender_metrics_overall_{tag}.csv"), index=False)
    by_gender.to_csv(os.path.join(res_dir, f"winogender_metrics_by_gender_{tag}.csv"), index=False)


def eval_unqover(args: argparse.Namespace, model, tokenizer, tag: str) -> None:
    dsid = UNQOVER_DS[args.unqover_dim]
    ds = load_dataset(dsid)["train"]

    rows = []
    n = len(ds)
    cap = min(n, args.max_samples) if args.max_samples is not None else n

    for i in tqdm(range(cap), desc=f"UnQover-{args.unqover_dim}"):
        ex = ds[i]
        context = str(ex.get("context", ""))
        q0 = ex.get("q0", {})
        q1 = ex.get("q1", {})

        def score_q(qobj: Dict) -> Tuple[int, float, float]:
            question = str(qobj.get("question", ""))
            a0 = str((qobj.get("ans0") or {}).get("text", ""))
            a1 = str((qobj.get("ans1") or {}).get("text", ""))
            prompt = (
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"A. {a0}\n"
                f"B. {a1}\n"
                "Answer:"
            )
            logps = completion_logprobs_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                options=["A", "B"],
                max_length=args.max_length,
                option_batch_size=args.option_batch_size,
            )
            probs = probs_from_logps(logps)
            pred = int(np.argmax(probs))
            return pred, float(probs[0]), float(max(probs))

        q0_pred, q0_prob_a, q0_maxp = score_q(q0)
        q1_pred, q1_prob_a, q1_maxp = score_q(q1)

        rows.append(
            {
                "context": context,
                "q0_question": str(q0.get("question", "")),
                "q1_question": str(q1.get("question", "")),
                "q0_pred": q0_pred,
                "q1_pred": q1_pred,
                "q0_prob_A": q0_prob_a,
                "q1_prob_A": q1_prob_a,
                "swap": int(q0_pred != q1_pred),
                "avg_max_prob": float((q0_maxp + q1_maxp) / 2.0),
            }
        )

    df = pd.DataFrame(rows)

    ans0_rate_q0 = float((df["q0_pred"] == 0).mean()) if len(df) else float("nan")
    ans0_rate_q1 = float((df["q1_pred"] == 0).mean()) if len(df) else float("nan")
    ans0_rate_all = float(
        (
            ((df["q0_pred"] == 0).sum() + (df["q1_pred"] == 0).sum())
            / max(1, (2 * len(df)))
        )
    )

    overall = {
        "model": tag,
        "benchmark": "unqover",
        "dimension": args.unqover_dim,
        "n": int(len(df)),
        "ans0_rate_q0": ans0_rate_q0,
        "ans0_rate_q1": ans0_rate_q1,
        "swap_rate_q0q1": float(df["swap"].mean()) if len(df) else float("nan"),
        "position_bias": float(abs(ans0_rate_all - 0.5) * 2.0) if len(df) else float("nan"),
        "avg_confidence": float(df["avg_max_prob"].mean()) if len(df) else float("nan"),
    }

    out_dir = os.path.join(args.output_dir, "unqover", args.unqover_dim, tag)
    res_dir = os.path.join(args.results_dir, "unqover", args.unqover_dim, tag)
    ensure_dirs(out_dir, res_dir)

    df.to_csv(os.path.join(out_dir, f"unqover_{args.unqover_dim}_preds_{tag}.csv"), index=False)
    pd.DataFrame([overall]).to_csv(
        os.path.join(res_dir, f"unqover_{args.unqover_dim}_metrics_overall_{tag}.csv"),
        index=False,
    )


def eval_bias_in_bios(args: argparse.Namespace, model, tokenizer, tag: str) -> None:
    ds = load_dataset("Elfsong/Bias_in_Bios")
    split = ds[args.bios_split]

    profs = sorted(set(ds["train"]["profession"]))
    disp = [p.replace("_", " ") for p in profs]

    n = len(split)
    cap = min(n, args.max_samples) if args.max_samples is not None else n

    rows = []
    for i in tqdm(range(cap), desc="Bias-in-Bios"):
        ex = split[i]
        bio = str(ex.get("hard_text", ""))
        gold_prof = str(ex.get("profession", ""))
        gender = str(ex.get("gender", ""))

        prompt = (
            "Read the biography and select the most likely profession from the options.\n"
            f"Biography: {bio}\n"
            f"Options: {', '.join(disp)}\n"
            "Answer:"
        )

        logps = completion_logprobs_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            options=disp,
            max_length=args.max_length,
            option_batch_size=args.option_batch_size,
        )
        probs = probs_from_logps(logps)
        pred_idx = int(np.argmax(probs))

        pred_prof = profs[pred_idx]
        correct = int(pred_prof == gold_prof)

        rows.append(
            {
                "hard_text": bio,
                "gold_profession": gold_prof,
                "pred_profession": pred_prof,
                "gender": gender,
                "correct": correct,
                "max_prob": float(np.max(probs)),
            }
        )

    df = pd.DataFrame(rows)

    male_acc = float(df.loc[df["gender"].str.lower() == "male", "correct"].mean()) if (df["gender"].str.lower() == "male").any() else float("nan")
    female_acc = float(df.loc[df["gender"].str.lower() == "female", "correct"].mean()) if (df["gender"].str.lower() == "female").any() else float("nan")

    overall = {
        "model": tag,
        "benchmark": "bias_in_bios",
        "split": args.bios_split,
        "n": int(len(df)),
        "accuracy": float(df["correct"].mean()) if len(df) else float("nan"),
        "accuracy_male": male_acc,
        "accuracy_female": female_acc,
        "gender_accuracy_gap_abs": float(abs(male_acc - female_acc)) if not np.isnan(male_acc) and not np.isnan(female_acc) else float("nan"),
        "avg_confidence": float(df["max_prob"].mean()) if len(df) else float("nan"),
    }

    by_gender = (
        df.groupby("gender", as_index=False)
        .agg(n=("gender", "size"), accuracy=("correct", "mean"))
    )
    by_gender["model"] = tag
    by_gender["benchmark"] = "bias_in_bios"

    out_dir = os.path.join(args.output_dir, "bias_in_bios", tag)
    res_dir = os.path.join(args.results_dir, "bias_in_bios", tag)
    ensure_dirs(out_dir, res_dir)

    df.to_csv(os.path.join(out_dir, f"bias_in_bios_preds_{tag}.csv"), index=False)
    pd.DataFrame([overall]).to_csv(os.path.join(res_dir, f"bias_in_bios_metrics_overall_{tag}.csv"), index=False)
    by_gender.to_csv(os.path.join(res_dir, f"bias_in_bios_metrics_by_gender_{tag}.csv"), index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline inference/eval on additional social-bias benchmarks")
    p.add_argument("--benchmark", choices=["bold", "honest", "winobias", "winogender", "unqover", "bias_in_bios"], required=True)
    p.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default="llama_8b")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--model_tag", type=str, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--option_batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_ROOT)

    p.add_argument("--honest_config", type=str, default="en_binary")
    p.add_argument("--unqover_dim", choices=["gender", "race", "religion", "nationality"], default="gender")
    p.add_argument("--bios_split", choices=["train", "dev", "test"], default="test")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seed_everything(args.seed)

    model, tokenizer, model_source = load_model_and_tokenizer(args.model, args.model_path, quantized=True)
    if args.model_tag:
        tag = slug(args.model_tag)
    elif args.model_path:
        tag = slug(os.path.basename(os.path.normpath(args.model_path)))
    else:
        tag = slug(args.model)

    print(f"[INFO] model_source={model_source}")
    print(f"[INFO] benchmark={args.benchmark}")

    if args.benchmark == "bold":
        eval_bold(args, model, tokenizer, tag)
    elif args.benchmark == "honest":
        eval_honest(args, model, tokenizer, tag)
    elif args.benchmark == "winobias":
        eval_winobias(args, model, tokenizer, tag)
    elif args.benchmark == "winogender":
        eval_winogender(args, model, tokenizer, tag)
    elif args.benchmark == "unqover":
        eval_unqover(args, model, tokenizer, tag)
    elif args.benchmark == "bias_in_bios":
        eval_bias_in_bios(args, model, tokenizer, tag)
    else:
        raise ValueError(f"Unknown benchmark {args.benchmark}")

    print("[INFO] done")


if __name__ == "__main__":
    main()
