#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for core debiasing results.

This script uses the exact rows already plotted in the core debiasing-method
figures, locates the corresponding per-example prediction files, and computes
paired bootstrap intervals for normalized bias-error reduction versus the base
model. Positive values mean lower bias error than the base model.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "figures"
N_BOOTSTRAP = 5000
SEED = 20260507

MODEL_CONFIGS = [
    ("llama8b", "Llama 8B", FIGURES / "llama8b" / "csv" / "debiasing_method_bars_data.csv"),
    ("qwen", "Qwen 27B", FIGURES / "qwen" / "csv" / "debiasing_method_bars_data.csv"),
    ("llama70b", "Llama 70B", FIGURES / "llama70b" / "csv" / "debiasing_method_bars_data.csv"),
]

METHOD_COLORS = {
    "DIY IT": "#5EB8E8",
    "DIY Two Pass (No IT)": "#55C783",
    "DIY Two Pass (IT)": "#F2A65A",
}

METHOD_MARKERS = {
    "DIY IT": "o",
    "DIY Two Pass (No IT)": "D",
    "DIY Two Pass (IT)": "s",
}

DATASET_ORDER = [
    "CrowS-Pairs",
    "StereoSet",
    "BBQ Ambig.",
    "BBQ Disambig.",
    "WinoBias",
    "WinoGender",
]

BBQ_UNKNOWN_STRINGS = {
    "unknown",
    "cannot be determined",
    "can't be determined",
    "not answerable",
    "not known",
    "not enough info",
    "not enough information",
    "cannot answer",
    "can't answer",
    "undetermined",
}


@dataclass
class MetricData:
    dataset_key: str
    metric: str
    ideal_value: float
    path: str
    units: dict[str, object]
    metric_fn: Callable[[list[object]], float]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | float | int | None, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def bool_int(value: str | int | bool | None) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    return int(float(text))


def normalize_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    while text.endswith("."):
        text = text[:-1].strip()
    return text


def row_method(row: dict[str, str]) -> str:
    return row.get("method_label") or row.get("method") or row.get("method_key") or ""


def error_from_metric(value: float, ideal_value: float) -> float:
    return abs(value - ideal_value)


def mean_pct(values: list[object]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals) * 100.0) if vals else float("nan")


def winobias_gap(values: list[object]) -> float:
    pro = [float(v["correct"]) for v in values if v["condition"] == "pro"]  # type: ignore[index]
    anti = [float(v["correct"]) for v in values if v["condition"] == "anti"]  # type: ignore[index]
    if not pro or not anti:
        return float("nan")
    return abs(float(np.mean(pro)) - float(np.mean(anti)))


def bbq_context_bias(values: list[object]) -> float:
    non_unknown = [v for v in values if not bool(v["unknown"])]  # type: ignore[index]
    if not non_unknown:
        return float("nan")
    target_rate = float(np.mean([float(v["target_selected"]) for v in non_unknown]))  # type: ignore[index]
    return target_rate * 2.0 - 1.0


def bbq_ambig_score(values: list[object]) -> float:
    if not values:
        return float("nan")
    acc = float(np.mean([float(v["correct"]) for v in values]))  # type: ignore[index]
    return (1.0 - acc) * bbq_context_bias(values) * 100.0


def bbq_disambig_score(values: list[object]) -> float:
    return bbq_context_bias(values) * 100.0


def load_bbq_target_locations() -> dict[tuple[str, str], int]:
    path = ROOT / "data" / "BBQ" / "analysis_scripts" / "additional_metadata.csv"
    targets: dict[tuple[str, str], int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            target_loc = as_float(row.get("target_loc"))
            if math.isnan(target_loc):
                continue
            category = (row.get("category") or "").replace(".jsonl", "").strip()
            key = (category, str(row.get("example_id", "")).strip())
            targets.setdefault(key, int(target_loc))
    return targets


BBQ_TARGETS = load_bbq_target_locations()


def bbq_pred_category(row: dict[str, str]) -> str:
    pred = row.get("pred_index") or row.get("pred_label")
    pred_idx = int(as_float(pred, -1))
    if pred_idx < 0:
        return ""
    selected_answer = normalize_text(row.get(f"ans{pred_idx}") or row.get("model_output"))
    if selected_answer in BBQ_UNKNOWN_STRINGS:
        return "unknown"
    return selected_answer


def build_row_file_index() -> dict[str, list[Path]]:
    prefixes = (
        "bbq_preds",
        "crows_pairs_scored",
        "stereoset_sentence_scores",
        "winobias_predictions",
        "winogender_preds",
    )
    index: dict[str, list[Path]] = {}
    for root in [ROOT / "outputs", ROOT / "archive"]:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".csv") and filename.startswith(prefixes):
                    path = Path(dirpath) / filename
                    index.setdefault(filename, []).append(path)
    return {k: sorted(v) for k, v in index.items()}


ROW_FILE_INDEX = build_row_file_index()


def candidate_output_rel(source: str) -> str | None:
    if source.startswith("results/new_results/"):
        return source.replace("results/new_results/", "outputs/new_outputs/", 1)
    if source.startswith("results/10_additional_benchmarks/"):
        return source.replace("results/10_additional_benchmarks/", "outputs/10_additional_benchmarks/", 1)
    if source.startswith("results/"):
        return source.replace("results/", "outputs/", 1)
    return None


def metric_to_row_filename(filename: str, dataset_key: str) -> str | None:
    if dataset_key == "crowspairs":
        return filename.replace("crows_pairs_metrics_overall_", "crows_pairs_scored_").replace(
            "crowspairs_metrics_overall_", "crows_pairs_scored_"
        )
    if dataset_key == "stereoset":
        return filename.replace("stereoset_metrics_", "stereoset_sentence_scores_")
    if dataset_key == "winobias":
        return filename.replace("winobias_metrics_overall_", "winobias_predictions_")
    if dataset_key == "winogender":
        return filename.replace("winogender_metrics_overall_", "winogender_preds_")
    return None


def locate_row_files(metric_source: str, dataset_key: str, model_key: str) -> tuple[list[Path], str]:
    if metric_source.startswith("recomputed_from_predictions:"):
        rel = metric_source.split(":", 1)[1]
        pred_dir = ROOT / rel
        files = sorted(pred_dir.glob("bbq_preds*.csv")) if pred_dir.is_dir() else []
        return files, "recomputed_prediction_dir"

    out_rel = candidate_output_rel(metric_source)
    if out_rel:
        out_path = ROOT / out_rel
        row_name = metric_to_row_filename(out_path.name, dataset_key)
        if dataset_key == "bbq":
            files = sorted(out_path.parent.glob("bbq_preds*.csv"))
            if files:
                return files, "same_output_directory"
            files = sorted(out_path.parent.glob("*/bbq_preds*.csv"))
            if files:
                return files, "child_output_directory"
        elif row_name:
            direct = out_path.with_name(row_name)
            if direct.exists():
                return [direct], "direct_metric_transform"
            indexed = [p for p in ROW_FILE_INDEX.get(row_name, []) if p.name == row_name]
            if indexed:
                return indexed[:1], "basename_index"

    if dataset_key == "bbq" and model_key == "llama8b" and "baseline_evalshared" in metric_source:
        files = sorted((ROOT / "outputs/2_base_models/bbq/llama_8b").glob("bbq_preds*.csv"))
        if files:
            return files, "llama8b_base_bbq_special_case"

    row_name = metric_to_row_filename(Path(metric_source).name, dataset_key)
    if row_name and row_name in ROW_FILE_INDEX:
        return ROW_FILE_INDEX[row_name][:1], "basename_index_fallback"

    return [], "not_found"


def load_metric_units(
    model_key: str,
    dataset_key: str,
    metric: str,
    ideal_value: float,
    metric_source: str,
) -> tuple[MetricData | None, dict[str, object]]:
    files, locator = locate_row_files(metric_source, dataset_key, model_key)
    audit = {
        "metric_source_file": metric_source,
        "locator": locator,
        "row_files": ";".join(str(p.relative_to(ROOT)) for p in files),
    }
    if not files:
        audit["reason"] = "row_file_not_found"
        return None, audit

    try:
        if dataset_key == "crowspairs":
            units: dict[str, object] = {}
            for row in read_csv(files[0]):
                if str(row.get("neutral", "0")).strip() in {"1", "true", "True"}:
                    continue
                units[str(row.get("pair_id", len(units)))] = bool_int(row.get("stereo_preferred"))
            return MetricData(dataset_key, metric, ideal_value, str(files[0].relative_to(ROOT)), units, mean_pct), audit

        if dataset_key == "stereoset":
            grouped: dict[str, dict[str, float]] = {}
            for row in read_csv(files[0]):
                label = row.get("gold_label")
                if label not in {"stereotype", "anti-stereotype"}:
                    continue
                grouped.setdefault(str(row["example_id"]), {})[label] = as_float(row.get("score"))
            units = {
                key: float(scores["stereotype"] > scores["anti-stereotype"])
                for key, scores in grouped.items()
                if "stereotype" in scores and "anti-stereotype" in scores
            }
            return MetricData(dataset_key, metric, ideal_value, str(files[0].relative_to(ROOT)), units, mean_pct), audit

        if dataset_key == "bbq":
            want_condition = "ambig" if metric == "Bias_score_ambig" else "disambig"
            units = {}
            for file in files:
                for row in read_csv(file):
                    condition = str(row.get("context_condition", "")).strip().lower()
                    if condition != want_condition:
                        continue
                    category = (row.get("source_file") or row.get("category") or "").replace(".jsonl", "").strip()
                    example_id = str(row.get("example_id", "")).strip()
                    target_loc = BBQ_TARGETS.get((category, example_id))
                    if target_loc is None:
                        continue
                    pred_idx = int(as_float(row.get("pred_index") or row.get("pred_label"), -1))
                    label = int(as_float(row.get("label"), -1))
                    if pred_idx < 0 or label < 0:
                        continue
                    key = "|".join([category, example_id, condition, row.get("question", "")])
                    units[key] = {
                        "correct": float(pred_idx == label),
                        "target_selected": float(pred_idx == target_loc),
                        "unknown": bbq_pred_category(row) == "unknown",
                    }
            metric_fn = bbq_ambig_score if metric == "Bias_score_ambig" else bbq_disambig_score
            return MetricData(dataset_key, metric, ideal_value, ";".join(str(p.relative_to(ROOT)) for p in files), units, metric_fn), audit

        if dataset_key == "winobias":
            units = {}
            for row in read_csv(files[0]):
                condition = str(row.get("condition", "")).strip().lower()
                if condition not in {"pro", "anti"}:
                    continue
                units[str(row.get("document_id", len(units)))] = {
                    "condition": condition,
                    "correct": float(bool_int(row.get("correct"))),
                }
            return MetricData(dataset_key, metric, ideal_value, str(files[0].relative_to(ROOT)), units, winobias_gap), audit

        if dataset_key == "winogender":
            pairs: dict[str, dict[str, int]] = {}
            for row in read_csv(files[0]):
                gender = str(row.get("gender", "")).strip().lower()
                if gender not in {"male", "female"}:
                    continue
                parts = str(row.get("sentid", "")).split(".")
                key = ".".join(parts[:3]) if len(parts) >= 4 else str(row.get("sentid", len(pairs)))
                pairs.setdefault(key, {})[gender] = bool_int(row.get("pred_is_occupation"))
            units = {
                key: float(pair["male"] != pair["female"])
                for key, pair in pairs.items()
                if "male" in pair and "female" in pair
            }
            return MetricData(dataset_key, metric, ideal_value, str(files[0].relative_to(ROOT)), units, lambda values: float(np.mean([float(v) for v in values])) if values else float("nan")), audit

    except Exception as exc:  # noqa: BLE001 - keep audit alive for messy experiment files.
        audit["reason"] = f"load_error:{type(exc).__name__}:{exc}"
        return None, audit

    audit["reason"] = "unsupported_dataset"
    return None, audit


def metric_on_indices(data: MetricData, values: list[object], indices: np.ndarray) -> float:
    sampled = [values[int(i)] for i in indices]
    return data.metric_fn(sampled)


def sampled_errors(data: MetricData, values: list[object], indices: np.ndarray) -> np.ndarray:
    """Compute normalized bias errors for a batch of bootstrap samples."""
    if data.dataset_key in {"crowspairs", "stereoset"}:
        arr = np.asarray([float(v) for v in values], dtype=float)
        metric = arr[indices].mean(axis=1) * 100.0
        return np.abs(metric - data.ideal_value)

    if data.dataset_key == "winogender":
        arr = np.asarray([float(v) for v in values], dtype=float)
        metric = arr[indices].mean(axis=1)
        return np.abs(metric - data.ideal_value)

    if data.dataset_key == "winobias":
        correct = np.asarray([float(v["correct"]) for v in values], dtype=float)  # type: ignore[index]
        is_pro = np.asarray([v["condition"] == "pro" for v in values], dtype=bool)  # type: ignore[index]
        sampled_correct = correct[indices]
        sampled_pro = is_pro[indices]
        pro_count = sampled_pro.sum(axis=1)
        anti_count = (~sampled_pro).sum(axis=1)
        pro_mean = np.divide(
            (sampled_correct * sampled_pro).sum(axis=1),
            pro_count,
            out=np.full(indices.shape[0], np.nan),
            where=pro_count > 0,
        )
        anti_mean = np.divide(
            (sampled_correct * (~sampled_pro)).sum(axis=1),
            anti_count,
            out=np.full(indices.shape[0], np.nan),
            where=anti_count > 0,
        )
        metric = np.abs(pro_mean - anti_mean)
        return np.abs(metric - data.ideal_value)

    if data.dataset_key == "bbq":
        correct = np.asarray([float(v["correct"]) for v in values], dtype=float)  # type: ignore[index]
        target = np.asarray([float(v["target_selected"]) for v in values], dtype=float)  # type: ignore[index]
        unknown = np.asarray([bool(v["unknown"]) for v in values], dtype=bool)  # type: ignore[index]
        sampled_correct = correct[indices]
        sampled_target = target[indices]
        sampled_non_unknown = ~unknown[indices]
        non_unknown_count = sampled_non_unknown.sum(axis=1)
        target_rate = np.divide(
            (sampled_target * sampled_non_unknown).sum(axis=1),
            non_unknown_count,
            out=np.full(indices.shape[0], np.nan),
            where=non_unknown_count > 0,
        )
        context_bias = target_rate * 2.0 - 1.0
        if data.metric == "Bias_score_ambig":
            acc = sampled_correct.mean(axis=1)
            metric = (1.0 - acc) * context_bias * 100.0
        else:
            metric = context_bias * 100.0
        return np.abs(metric - data.ideal_value)

    effects = np.empty(indices.shape[0], dtype=float)
    for i, idx in enumerate(indices):
        effects[i] = error_from_metric(metric_on_indices(data, values, idx), data.ideal_value)
    return effects


def bootstrap_comparison(
    ref: MetricData,
    method: MetricData,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    keys = sorted(set(ref.units).intersection(method.units))
    ref_values = [ref.units[k] for k in keys]
    method_values = [method.units[k] for k in keys]
    n = len(keys)
    if n == 0:
        raise ValueError("no_overlapping_units")

    ref_metric = ref.metric_fn(ref_values)
    method_metric = method.metric_fn(method_values)
    ref_error = error_from_metric(ref_metric, ref.ideal_value)
    method_error = error_from_metric(method_metric, method.ideal_value)
    point = ref_error - method_error

    effects = np.empty(N_BOOTSTRAP, dtype=float)
    chunk_size = max(100, min(500, 2_000_000 // max(n, 1)))
    start = 0
    while start < N_BOOTSTRAP:
        stop = min(start + chunk_size, N_BOOTSTRAP)
        idx = rng.integers(0, n, size=(stop - start, n))
        effects[start:stop] = sampled_errors(ref, ref_values, idx) - sampled_errors(method, method_values, idx)
        start = stop

    lo, hi = np.nanpercentile(effects, [2.5, 97.5])
    return {
        "reference_metric_recomputed": ref_metric,
        "method_metric_recomputed": method_metric,
        "reference_bias_error_recomputed": ref_error,
        "method_bias_error_recomputed": method_error,
        "bias_error_reduction": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_reference_units": len(ref.units),
        "n_method_units": len(method.units),
        "n_aligned_units": n,
    }


def process_model(model_key: str, model_label: str, csv_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_csv(csv_path)
    panels: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        panels.setdefault((row["panel_label"], row["metric"]), []).append(row)

    result_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str, str, float, str], MetricData | None] = {}
    rng = np.random.default_rng(SEED + sum(ord(ch) for ch in model_key))

    for (panel_label, metric), panel_rows in panels.items():
        base_rows = [r for r in panel_rows if row_method(r) == "Base Model Inference"]
        if not base_rows:
            audit_rows.append({"model": model_label, "panel_label": panel_label, "metric": metric, "reason": "base_row_missing"})
            continue
        base = base_rows[0]
        dataset_key = base["dataset_key"]
        ideal = as_float(base["ideal_value"])
        source = base["metric_source_file"]
        key = (model_key, dataset_key, metric, ideal, source)
        if key not in cache:
            data, audit = load_metric_units(model_key, dataset_key, metric, ideal, source)
            cache[key] = data
            audit_rows.append({"model": model_label, "panel_label": panel_label, "method": "Base Model Inference", "metric": metric, **audit})
        ref_data = cache[key]
        if ref_data is None:
            continue

        for method_row in panel_rows:
            method = row_method(method_row)
            if method == "Base Model Inference":
                continue
            source = method_row["metric_source_file"]
            key = (model_key, dataset_key, metric, ideal, source)
            if key not in cache:
                data, audit = load_metric_units(model_key, dataset_key, metric, ideal, source)
                cache[key] = data
                audit_rows.append({"model": model_label, "panel_label": panel_label, "method": method, "metric": metric, **audit})
            method_data = cache[key]
            if method_data is None:
                continue
            try:
                stats = bootstrap_comparison(ref_data, method_data, rng)
            except Exception as exc:  # noqa: BLE001
                audit_rows.append(
                    {
                        "model": model_label,
                        "panel_label": panel_label,
                        "method": method,
                        "metric": metric,
                        "metric_source_file": source,
                        "reason": f"bootstrap_error:{type(exc).__name__}:{exc}",
                    }
                )
                continue

            figure_ref_error = as_float(base.get("normalized_bias_error_plotted"))
            figure_method_error = as_float(method_row.get("normalized_bias_error_plotted"))
            figure_reduction = figure_ref_error - figure_method_error
            result_rows.append(
                {
                    "model_key": model_key,
                    "model": model_label,
                    "dataset_key": dataset_key,
                    "panel_label": panel_label,
                    "method": method,
                    "metric": metric,
                    "ideal_value": ideal,
                    "native_value_from_figure": as_float(method_row.get("native_value")),
                    "bias_error_from_figure": figure_method_error,
                    "reference_bias_error_from_figure": figure_ref_error,
                    "figure_bias_error_reduction": figure_reduction,
                    "bootstrap_bias_error_reduction": stats["bias_error_reduction"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "reference_metric_recomputed": stats["reference_metric_recomputed"],
                    "method_metric_recomputed": stats["method_metric_recomputed"],
                    "reference_bias_error_recomputed": stats["reference_bias_error_recomputed"],
                    "method_bias_error_recomputed": stats["method_bias_error_recomputed"],
                    "point_delta_vs_figure": float(stats["bias_error_reduction"]) - figure_reduction,
                    "n_reference_units": stats["n_reference_units"],
                    "n_method_units": stats["n_method_units"],
                    "n_aligned_units": stats["n_aligned_units"],
                    "n_bootstrap": N_BOOTSTRAP,
                    "seed": SEED,
                    "reference_row_files": ref_data.path,
                    "method_row_files": method_data.path,
                    "reference_metric_source_file": base["metric_source_file"],
                    "method_metric_source_file": source,
                }
            )

    return result_rows, audit_rows


def model_sort_key(row: dict[str, object]) -> tuple[int, int]:
    dataset = str(row["panel_label"])
    method = str(row["method"])
    return (
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER),
        list(METHOD_COLORS).index(method) if method in METHOD_COLORS else len(METHOD_COLORS),
    )


def plot_model(rows: list[dict[str, object]], model_label: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=model_sort_key)
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No paired bootstrap intervals available for {model_label}", ha="center", va="center", fontsize=13)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    labels = [f"{r['panel_label']} · {r['method']}" for r in rows]
    y = np.arange(len(rows))
    fig_h = max(5.5, 0.42 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(9.8, fig_h))

    for i, row in enumerate(rows):
        point = float(row["bootstrap_bias_error_reduction"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        method = str(row["method"])
        color = METHOD_COLORS.get(method, "#888888")
        marker = METHOD_MARKERS.get(method, "o")
        ax.errorbar(
            point,
            y[i],
            xerr=[[point - lo], [hi - point]],
            fmt=marker,
            markersize=8.5,
            color=color,
            ecolor=color,
            elinewidth=2.5,
            capsize=4,
            markeredgecolor="#30323A",
            markeredgewidth=1.2,
            zorder=3,
        )

    ax.axvline(0, color="#4E5563", linewidth=1.3, linestyle=(0, (4, 4)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlabel("Bias error reduction vs. base model (positive is better)", fontsize=13)
    ax.set_title(f"{model_label}: paired bootstrap confidence intervals", fontsize=16, pad=16, weight="semibold")
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color="#E7E8ED", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#B7BBC5")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[m],
            color="none",
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor="#30323A",
            markeredgewidth=1.2,
            markersize=8.5,
            label=m,
        )
        for m in METHOD_COLORS
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = [label for _, label, _ in MODEL_CONFIGS]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.4), sharex=True)
    for ax, model_label in zip(axes, models):
        model_rows = sorted([r for r in rows if r["model"] == model_label], key=model_sort_key)
        y = np.arange(len(model_rows))
        labels = []
        for r in model_rows:
            method_label = str(r["method"]).replace(" (", "\n(")
            labels.append(f"{r['panel_label']}\n{method_label}")
        for i, row in enumerate(model_rows):
            point = float(row["bootstrap_bias_error_reduction"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            method = str(row["method"])
            color = METHOD_COLORS.get(method, "#888888")
            marker = METHOD_MARKERS.get(method, "o")
            ax.errorbar(
                point,
                y[i],
                xerr=[[point - lo], [hi - point]],
                fmt=marker,
                markersize=7.5,
                color=color,
                ecolor=color,
                elinewidth=2.2,
                capsize=3.5,
                markeredgecolor="#30323A",
                markeredgewidth=1.1,
                zorder=3,
            )
        ax.axvline(0, color="#4E5563", linewidth=1.2, linestyle=(0, (4, 4)), zorder=1)
        ax.set_title(model_label, fontsize=15, weight="semibold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.2)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E7E8ED", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#B7BBC5")
    fig.supxlabel("Bias error reduction vs. base model (positive is better)", fontsize=13)
    fig.suptitle("Paired bootstrap confidence intervals across models", fontsize=17, weight="semibold", y=0.99)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[m],
            color="none",
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor="#30323A",
            markeredgewidth=1.1,
            markersize=8,
            label=m,
        )
        for m in METHOD_COLORS
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.91])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_rows: list[dict[str, object]] = []
    all_audit: list[dict[str, object]] = []

    for model_key, model_label, csv_path in MODEL_CONFIGS:
        rows, audit_rows = process_model(model_key, model_label, csv_path)
        all_rows.extend(rows)
        all_audit.extend(audit_rows)

        out_dir = FIGURES / model_key
        write_csv(
            out_dir / "csv" / "bootstrap_confidence_intervals_data.csv",
            rows,
            [
                "model_key",
                "model",
                "dataset_key",
                "panel_label",
                "method",
                "metric",
                "ideal_value",
                "native_value_from_figure",
                "bias_error_from_figure",
                "reference_bias_error_from_figure",
                "figure_bias_error_reduction",
                "bootstrap_bias_error_reduction",
                "ci_low",
                "ci_high",
                "reference_metric_recomputed",
                "method_metric_recomputed",
                "reference_bias_error_recomputed",
                "method_bias_error_recomputed",
                "point_delta_vs_figure",
                "n_reference_units",
                "n_method_units",
                "n_aligned_units",
                "n_bootstrap",
                "seed",
                "reference_row_files",
                "method_row_files",
                "reference_metric_source_file",
                "method_metric_source_file",
            ],
        )
        audit_fields = sorted({key for row in audit_rows for key in row})
        write_csv(out_dir / "csv" / "bootstrap_confidence_intervals_audit.csv", audit_rows, audit_fields)
        plot_model(rows, model_label, out_dir / "pdf" / "bootstrap_confidence_intervals.pdf")

    combined_dir = FIGURES / "bootstrap_ci"
    write_csv(combined_dir / "csv" / "all_models_bootstrap_confidence_intervals_data.csv", all_rows, [
        "model_key",
        "model",
        "dataset_key",
        "panel_label",
        "method",
        "metric",
        "ideal_value",
        "native_value_from_figure",
        "bias_error_from_figure",
        "reference_bias_error_from_figure",
        "figure_bias_error_reduction",
        "bootstrap_bias_error_reduction",
        "ci_low",
        "ci_high",
        "reference_metric_recomputed",
        "method_metric_recomputed",
        "reference_bias_error_recomputed",
        "method_bias_error_recomputed",
        "point_delta_vs_figure",
        "n_reference_units",
        "n_method_units",
        "n_aligned_units",
        "n_bootstrap",
        "seed",
        "reference_row_files",
        "method_row_files",
        "reference_metric_source_file",
        "method_metric_source_file",
    ])
    audit_fields = sorted({key for row in all_audit for key in row})
    write_csv(combined_dir / "csv" / "all_models_bootstrap_confidence_intervals_audit.csv", all_audit, audit_fields)
    plot_combined(all_rows, combined_dir / "pdf" / "all_models_bootstrap_confidence_intervals.pdf")

    print(f"Wrote {len(all_rows)} bootstrap comparisons")
    print(f"Wrote {len(all_audit)} audit rows")


if __name__ == "__main__":
    main()
