#!/usr/bin/env python3
"""Replot existing paper figures with bootstrap confidence intervals.

The original figure PDFs are left untouched. This script writes *_with_ci.pdf
variants and *_with_ci_data.csv files under each model folder.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_bootstrap_confidence_intervals as boot


ROOT = boot.ROOT
FIGURES = ROOT / "figures"
N_BOOTSTRAP = 200
boot.N_BOOTSTRAP = N_BOOTSTRAP
SEED = boot.SEED + 901

MODEL_DIRS = [
    ("llama8b", "Llama 8B"),
    ("qwen", "Qwen 27B"),
    ("llama70b", "Llama 70B"),
]

PANELS = boot.DATASET_ORDER
CORE_METHOD_ORDER = [
    "Base Model Inference",
    "Base Model",
    "ICL",
    "DIY IT",
    "DIY Two Pass (No IT)",
    "DIY Two Pass (IT)",
]
CORE_COLORS = {
    "Base Model Inference": "#C7CCD8",
    "Base Model": "#C7CCD8",
    "ICL": "#C4B5FD",
    "DIY IT": "#5EB8E8",
    "DIY Two Pass (No IT)": "#55C783",
    "DIY Two Pass (IT)": "#F2A65A",
}
CORE_HATCHES = {
    "Base Model Inference": "",
    "Base Model": "",
    "ICL": "xx",
    "DIY IT": "///",
    "DIY Two Pass (No IT)": "\\\\\\",
    "DIY Two Pass (IT)": "...",
}
INTERVENTION_COLORS = {
    "Stereotype replacement": "#7DD3FC",
    "Individuation": "#5EEAD4",
    "Perspective-taking": "#A7F3D0",
    "Counter-stereotypic imaging": "#FDBA74",
    "Positive contact": "#C4B5FD",
}
INTERVENTION_HATCHES = {
    "Stereotype replacement": "///",
    "Individuation": "\\\\\\",
    "Perspective-taking": "xxx",
    "Counter-stereotypic imaging": "...",
    "Positive contact": "---",
}

METRIC_DATA_CACHE: dict[tuple[str, str, str, float, str], tuple[boot.MetricData | None, dict[str, object]]] = {}
BIAS_CI_CACHE: dict[tuple[str, str, str, float, str, float], dict[str, object]] = {}
REDUCTION_CI_CACHE: dict[tuple[str, str, str, str, float, str, str, float], dict[str, object]] = {}
REASONING_CI_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    preferred = [
        "model_key",
        "model",
        "figure",
        "dataset_key",
        "panel_label",
        "method_key",
        "method",
        "method_label",
        "intervention_key",
        "intervention_label",
        "metric",
        "point",
        "ci_low",
        "ci_high",
        "n_units",
        "n_bootstrap",
        "metric_source_file",
        "row_files",
        "status",
        "reason",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str | int | float | None, default: float = np.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def method_label(row: dict[str, str]) -> str:
    value = row.get("method_label") or row.get("method") or row.get("method_key") or ""
    return value.replace("\n", " ")


def value_col(row: dict[str, str]) -> str:
    for col in ["normalized_bias_error_plotted", "normalized_bias_error", "bias_error_reduction_vs_base", "accuracy_percent_plotted"]:
        if col in row:
            return col
    return "point"


def ci_for_bias_row(model_key: str, row: dict[str, str]) -> dict[str, object]:
    dataset_key = row["dataset_key"]
    metric = row["metric"]
    ideal = safe_float(row.get("ideal_value"), 0.0)
    source = row.get("metric_source_file") or row.get("source_file") or ""
    point = safe_float(row.get("normalized_bias_error_plotted") or row.get("normalized_bias_error"))
    cache_key = (model_key, dataset_key, metric, ideal, source, point)
    if cache_key in BIAS_CI_CACHE:
        return dict(BIAS_CI_CACHE[cache_key])
    data_key = (model_key, dataset_key, metric, ideal, source)
    if data_key not in METRIC_DATA_CACHE:
        METRIC_DATA_CACHE[data_key] = boot.load_metric_units(model_key, dataset_key, metric, ideal, source)
    data, audit = METRIC_DATA_CACHE[data_key]
    out: dict[str, object] = {
        "point": point,
        "metric_source_file": source,
        "row_files": audit.get("row_files", ""),
        "status": "ok" if data is not None else "missing",
        "reason": audit.get("reason", ""),
        "n_bootstrap": N_BOOTSTRAP,
    }
    if data is None:
        BIAS_CI_CACHE[cache_key] = dict(out)
        return out
    values = list(data.units.values())
    if not values:
        out.update({"status": "missing", "reason": "no_metric_units"})
        BIAS_CI_CACHE[cache_key] = dict(out)
        return out
    metric_value = data.metric_fn(values)
    point_recomputed = boot.error_from_metric(metric_value, data.ideal_value)
    n = len(values)
    rng = np.random.default_rng(SEED + abs(sum(ord(c) for c in source)) % 100000)
    effects = np.empty(N_BOOTSTRAP, dtype=float)
    chunk_size = max(100, min(600, 2_000_000 // max(n, 1)))
    start = 0
    while start < N_BOOTSTRAP:
        stop = min(start + chunk_size, N_BOOTSTRAP)
        idx = rng.integers(0, n, size=(stop - start, n))
        effects[start:stop] = boot.sampled_errors(data, values, idx)
        start = stop
    lo, hi = np.nanpercentile(effects, [2.5, 97.5])
    out.update(
        {
            "point": point,
            "point_recomputed": point_recomputed,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_units": n,
        }
    )
    BIAS_CI_CACHE[cache_key] = dict(out)
    return out


def ci_for_bias_reduction(model_key: str, row: dict[str, str], base_row: dict[str, str]) -> dict[str, object]:
    source = row.get("metric_source_file") or ""
    point = safe_float(row.get("bias_error_reduction_vs_base"))
    cache_key = (model_key, base_row.get("metric_source_file", ""), source, row.get("panel_label", ""), point)
    if cache_key in REDUCTION_CI_CACHE:
        return dict(REDUCTION_CI_CACHE[cache_key])

    base_ci = ci_for_bias_row(model_key, base_row)
    method_ci = ci_for_bias_row(model_key, row)
    out: dict[str, object] = {
        "point": point,
        "metric_source_file": source,
        "reference_metric_source_file": base_row.get("metric_source_file", ""),
        "row_files": method_ci.get("row_files", ""),
        "reference_row_files": base_ci.get("row_files", ""),
        "status": "ok" if base_ci.get("status") == "ok" and method_ci.get("status") == "ok" else "missing",
        "reason": method_ci.get("reason") or base_ci.get("reason", ""),
        "n_bootstrap": N_BOOTSTRAP,
    }
    if out["status"] != "ok":
        REDUCTION_CI_CACHE[cache_key] = dict(out)
        return out

    base_point = safe_float(base_ci.get("point"))
    method_point = safe_float(method_ci.get("point"))
    base_low = safe_float(base_ci.get("ci_low"))
    base_high = safe_float(base_ci.get("ci_high"))
    method_low = safe_float(method_ci.get("ci_low"))
    method_high = safe_float(method_ci.get("ci_high"))
    point_recomputed = base_point - method_point
    out.update(
        {
            "point_recomputed": point_recomputed,
            "ci_low": base_low - method_high,
            "ci_high": base_high - method_low,
            "n_units": min(int(base_ci.get("n_units", 0)), int(method_ci.get("n_units", 0))),
            "ci_note": "bootstrap row CIs propagated as base_low-method_high/base_high-method_low",
        }
    )
    REDUCTION_CI_CACHE[cache_key] = dict(out)
    return out


def locate_reasoning_predictions(metric_source: str, dataset_key: str) -> Path | None:
    if metric_source.startswith("results/new_results/"):
        base = ROOT / metric_source.replace("results/new_results/", "outputs/new_outputs/", 1)
    elif metric_source.startswith("results/"):
        base = ROOT / metric_source.replace("results/", "outputs/", 1)
    else:
        base = ROOT / metric_source
    filename = base.name
    if "_metrics_overall_" in filename:
        candidate = base.with_name(filename.replace("_metrics_overall_", "_predictions_"))
        if candidate.exists():
            return candidate
    parent = base.parent
    if parent.exists():
        matches = sorted(parent.glob(f"{dataset_key}_predictions*.csv"))
        if matches:
            return matches[0]
        matches = sorted(parent.glob("*predictions*.csv"))
        if matches:
            return matches[0]
    return None


def ci_for_reasoning_row(row: dict[str, str]) -> dict[str, object]:
    source = row.get("source_file") or row.get("metric_source_file") or ""
    cache_key = (row["dataset_key"], source)
    if cache_key in REASONING_CI_CACHE:
        return dict(REASONING_CI_CACHE[cache_key])
    point = safe_float(row.get("accuracy_percent_plotted") or safe_float(row.get("accuracy")) * 100.0)
    pred = locate_reasoning_predictions(source, row["dataset_key"])
    out: dict[str, object] = {
        "point": point,
        "metric_source_file": source,
        "row_files": "" if pred is None else str(pred.relative_to(ROOT)),
        "status": "ok" if pred is not None else "missing",
        "reason": "" if pred is not None else "prediction_file_not_found",
        "n_bootstrap": N_BOOTSTRAP,
    }
    if pred is None:
        REASONING_CI_CACHE[cache_key] = dict(out)
        return out
    rows = read_csv(pred)
    vals = []
    for r in rows:
        if "correct" in r:
            vals.append(float(boot.bool_int(r["correct"])))
        elif "is_correct" in r:
            vals.append(float(boot.bool_int(r["is_correct"])))
    if not vals:
        out.update({"status": "missing", "reason": "correct_column_not_found"})
        REASONING_CI_CACHE[cache_key] = dict(out)
        return out
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(SEED + abs(sum(ord(c) for c in source)) % 100000)
    effects = np.empty(N_BOOTSTRAP, dtype=float)
    chunk_size = max(100, min(800, 2_000_000 // max(n, 1)))
    start = 0
    while start < N_BOOTSTRAP:
        stop = min(start + chunk_size, N_BOOTSTRAP)
        idx = rng.integers(0, n, size=(stop - start, n))
        effects[start:stop] = arr[idx].mean(axis=1) * 100.0
        start = stop
    lo, hi = np.nanpercentile(effects, [2.5, 97.5])
    out.update({"ci_low": float(lo), "ci_high": float(hi), "n_units": n, "point_recomputed": float(arr.mean() * 100.0)})
    REASONING_CI_CACHE[cache_key] = dict(out)
    return out


def add_ci_to_rows(model_key: str, model_label: str, figure: str, rows: list[dict[str, str]], kind: str) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    base_by_panel = {
        (r.get("panel_label", ""), r.get("metric", "")): r
        for r in rows
        if method_label(r) in {"Base Model Inference", "Base Model"}
    }
    for idx, row in enumerate(rows, 1):
        if idx == 1 or idx % 20 == 0 or idx == len(rows):
            print(f"{model_label}: {figure} {idx}/{len(rows)}", flush=True)
        out: dict[str, object] = {"model_key": model_key, "model": model_label, "figure": figure, **row}
        if kind == "bias":
            out.update(ci_for_bias_row(model_key, row))
        elif kind == "reduction":
            if method_label(row) in {"Base Model Inference", "Base Model"}:
                out.update({"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "status": "base", "n_bootstrap": N_BOOTSTRAP})
            else:
                base = base_by_panel.get((row.get("panel_label", ""), row.get("metric", "")))
                if base is None:
                    out.update({"point": safe_float(row.get("bias_error_reduction_vs_base")), "status": "missing", "reason": "base_row_missing"})
                else:
                    out.update(ci_for_bias_reduction(model_key, row, base))
        elif kind == "reasoning":
            out.update(ci_for_reasoning_row(row))
        enriched.append(out)
    return enriched


def set_style() -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11.0,
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def yerr(point: float, row: dict[str, object]) -> np.ndarray | None:
    lo = safe_float(row.get("ci_low"))
    hi = safe_float(row.get("ci_high"))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return np.asarray([[max(0.0, point - lo)], [max(0.0, hi - point)]])


def plot_debiasing(model_key: str, model_label: str, rows: list[dict[str, object]], out: Path) -> None:
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), sharey=False)
    axes = axes.ravel()
    methods = [m for m in ["Base Model Inference", "DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"] if any(method_label(r) == m for r in rows)]  # type: ignore[arg-type]
    x = np.arange(len(methods))
    for ax, panel in zip(axes, PANELS):
        panel_rows = {method_label(r): r for r in rows if r.get("panel_label") == panel}
        vals = [safe_float(panel_rows[m].get("point")) if m in panel_rows else np.nan for m in methods]
        bars = ax.bar(
            x,
            vals,
            color=[CORE_COLORS.get(m, "#CCCCCC") for m in methods],
            edgecolor="#1F2937",
            linewidth=0.8,
            hatch=[CORE_HATCHES.get(m, "") for m in methods],
            zorder=3,
        )
        for i, m in enumerate(methods):
            if m not in panel_rows or not np.isfinite(vals[i]):
                continue
            err = yerr(vals[i], panel_rows[m])
            if err is not None:
                ax.errorbar(i, vals[i], yerr=err, fmt="none", ecolor="#1F2937", elinewidth=1.3, capsize=3.5, zorder=5)
        ax.set_title(panel, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Base Model Inference", "Base").replace(" (", "\n(") for m in methods])
        ax.set_ylabel("Bias error")
        ax.grid(axis="y", color="#E5E7EB", linestyle=":", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [Patch(facecolor=CORE_COLORS[m], edgecolor="#1F2937", hatch=CORE_HATCHES[m], label=m.replace("Base Model Inference", "Base Model")) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(f"{model_label}: debiasing performance with 95% bootstrap CIs", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_reasoning(model_key: str, model_label: str, rows: list[dict[str, object]], out: Path) -> None:
    set_style()
    panels = sorted({str(r["benchmark_label"]) for r in rows})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.8), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    order = ["Base Model Inference", "ICL", "DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"]
    for ax, panel in zip(axes, panels):
        panel_rows = {method_label(r): r for r in rows if r.get("benchmark_label") == panel}
        methods = [m for m in order if m in panel_rows]
        x = np.arange(len(methods))
        vals = [safe_float(panel_rows[m].get("point")) for m in methods]
        ax.bar(x, vals, color=[CORE_COLORS.get(m, "#CCCCCC") for m in methods], edgecolor="#1F2937", linewidth=0.8, hatch=[CORE_HATCHES.get(m, "") for m in methods], zorder=3)
        for i, m in enumerate(methods):
            err = yerr(vals[i], panel_rows[m])
            if err is not None:
                ax.errorbar(i, vals[i], yerr=err, fmt="none", ecolor="#1F2937", elinewidth=1.3, capsize=3.5, zorder=5)
        ax.set_title(panel, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Base Model Inference", "Base").replace(" (", "\n(") for m in methods])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#E5E7EB", linestyle=":", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"{model_label}: reasoning performance with 95% bootstrap CIs", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_baseline(model_key: str, model_label: str, rows: list[dict[str, object]], out: Path) -> None:
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 12.2), sharey=False)
    axes = axes.ravel()
    method_keys = []
    for r in rows:
        key = str(r.get("method_key", ""))
        if key not in method_keys:
            method_keys.append(key)
    method_labels = {str(r.get("method_key", "")): str(r.get("method_label", "")) for r in rows}
    y_positions = np.arange(len(method_keys)) * 1.45
    group_colors = {"base": "#A9B7D9", "icl": "#C4B5FD", "baseline": "#8EC5FF", "diy_tune": "#5EB8E8", "diy_twopass": "#55C783", "diy_combo": "#F2A65A"}
    group_markers = {"base": "o", "icl": "D", "baseline": "o", "diy_tune": "D", "diy_twopass": "D", "diy_combo": "D"}
    for ax, panel in zip(axes, PANELS):
        panel_rows = {str(r.get("method_key", "")): r for r in rows if r.get("panel_label") == panel}
        vals = []
        for y, key in zip(y_positions, method_keys):
            r = panel_rows.get(key)
            if r is None:
                vals.append(np.nan)
                continue
            val = safe_float(r.get("point"))
            vals.append(val)
            group = str(r.get("group", "baseline"))
            color = group_colors.get(group, "#8EC5FF")
            ax.hlines(y, 0, val, color="#A8B3C5", linewidth=2.0, zorder=1)
            err = yerr(val, r)
            if err is not None:
                ax.errorbar(val, y, xerr=err, fmt="none", ecolor="#1F2937", elinewidth=1.0, capsize=3.0, zorder=4)
            ax.scatter(val, y, s=82 if group == "baseline" else 105, marker=group_markers.get(group, "o"), facecolor=color, edgecolor="#17202A", linewidth=1.35, zorder=5)
        finite = [v for v in vals if np.isfinite(v)]
        xmax = max(finite) if finite else 1.0
        ax.set_xlim(0, xmax * 1.42)
        ax.set_title(panel, fontweight="bold")
        ax.set_xlabel("Bias error")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([method_labels.get(k, k) for k in method_keys], fontsize=9 if model_key == "llama8b" else 9)
        ax.set_ylim(y_positions[-1] + 1, y_positions[0] - 1)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{model_label}: baselines with 95% bootstrap CIs", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.06, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_intervention(model_key: str, model_label: str, rows: list[dict[str, object]], out: Path) -> None:
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), sharey=False)
    axes = axes.ravel()
    methods = ["ICL", "DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"]
    interventions = [i for i in INTERVENTION_COLORS if any(r.get("intervention_label") == i for r in rows)]
    x = np.arange(len(methods))
    width = 0.13
    offsets = np.linspace(-2 * width, 2 * width, len(interventions))
    for ax, panel in zip(axes, PANELS):
        for offset, intervention in zip(offsets, interventions):
            vals = []
            errs = []
            for method in methods:
                matches = [r for r in rows if r.get("panel_label") == panel and method_label(r) == method and r.get("intervention_label") == intervention]
                if not matches:
                    vals.append(np.nan)
                    errs.append(None)
                    continue
                r = matches[0]
                val = safe_float(r.get("point"))
                vals.append(val)
                errs.append(yerr(val, r))
            ax.bar(x + offset, vals, width=width, color=INTERVENTION_COLORS[intervention], edgecolor="#111827", linewidth=0.7, hatch=INTERVENTION_HATCHES[intervention], zorder=3)
            for i, (val, err) in enumerate(zip(vals, errs)):
                if err is not None and np.isfinite(val):
                    ax.errorbar(x[i] + offset, val, yerr=err, fmt="none", ecolor="#111827", elinewidth=0.9, capsize=2.2, zorder=5)
        ax.axhline(0, color="#4B5563", linewidth=1.0)
        ax.set_title(panel, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (", "\n(") for m in methods])
        ax.set_ylabel("Bias error reduction")
        ax.grid(axis="y", color="#E5E7EB", linestyle=":", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [Patch(facecolor=INTERVENTION_COLORS[i], edgecolor="#111827", hatch=INTERVENTION_HATCHES[i], label=i) for i in interventions]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(f"{model_label}: intervention ablation with 95% bootstrap CIs", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def macro_ci(rows: list[dict[str, object]], method_key: str, source: str) -> tuple[float, float] | None:
    vals = [r for r in rows if str(r.get("method_key", "")) == method_key and np.isfinite(safe_float(r.get("ci_low"))) and np.isfinite(safe_float(r.get("ci_high")))]
    if not vals:
        return None
    lows = np.asarray([safe_float(r["ci_low"]) for r in vals], dtype=float)
    highs = np.asarray([safe_float(r["ci_high"]) for r in vals], dtype=float)
    return float(np.mean(lows)), float(np.mean(highs))


def plot_pareto(model_key: str, model_label: str, pareto_rows: list[dict[str, str]], bias_ci_rows: list[dict[str, object]], reasoning_ci_rows: list[dict[str, object]], out: Path) -> None:
    set_style()
    complete_rows = [
        row
        for row in pareto_rows
        if np.isfinite(safe_float(row.get("mean_bias_error")))
        and np.isfinite(safe_float(row.get("mean_reasoning_accuracy")))
    ]
    bias_only_rows = [
        row
        for row in pareto_rows
        if np.isfinite(safe_float(row.get("mean_bias_error")))
        and not np.isfinite(safe_float(row.get("mean_reasoning_accuracy")))
        and row.get("group", "baseline") == "baseline"
    ]
    if bias_only_rows:
        fig, (ax, ax_pending) = plt.subplots(
            2,
            1,
            figsize=(8.2, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [4.4, 1.0], "hspace": 0.08},
        )
        ax_pending.set_facecolor("#F7F9FC")
    else:
        fig, ax = plt.subplots(figsize=(8.2, 6.0))
        ax_pending = None
    colors = {"base": "#A9B7D9", "icl": "#C4B5FD", "baseline": "#8EC5FF", "diy_tune": "#5EB8E8", "diy_twopass": "#55C783", "diy_combo": "#F2A65A"}
    markers = {"base": "o", "icl": "D", "baseline": "o", "diy_tune": "D", "diy_twopass": "D", "diy_combo": "D"}
    for row in complete_rows:
        x = safe_float(row.get("mean_bias_error"))
        y = safe_float(row.get("mean_reasoning_accuracy"))
        group = row.get("group", "baseline")
        method_key = row.get("method_key", "")
        xci = macro_ci(bias_ci_rows, method_key, "bias")
        yci = macro_ci(reasoning_ci_rows, method_key, "reasoning")
        if xci is not None:
            ax.errorbar(x, y, xerr=np.asarray([[max(0, x - xci[0])], [max(0, xci[1] - x)]]), fmt="none", ecolor="#374151", elinewidth=0.9, capsize=2.5, alpha=0.8, zorder=2)
        if yci is not None:
            ax.errorbar(x, y, yerr=np.asarray([[max(0, y - yci[0])], [max(0, yci[1] - y)]]), fmt="none", ecolor="#374151", elinewidth=0.9, capsize=2.5, alpha=0.8, zorder=2)
        ax.scatter(x, y, s=95 if group == "baseline" else 125, marker=markers.get(group, "o"), facecolor=colors.get(group, "#8EC5FF"), edgecolor="#111827", linewidth=1.2, zorder=4)
        ax.text(x + 0.12, y + 0.08, row.get("method_label", method_key), fontsize=8.5)
    if ax_pending is not None:
        short_labels = {
            "RSB": "RSB",
            "SelfDebias": "SelfDebias",
        }
        for idx, row in enumerate(sorted(bias_only_rows, key=lambda r: safe_float(r.get("mean_bias_error")))):
            x = safe_float(row.get("mean_bias_error"))
            method_key = row.get("method_key", "")
            xci = macro_ci(bias_ci_rows, method_key, "bias")
            if xci is not None:
                ax_pending.errorbar(x, 0, xerr=np.asarray([[max(0, x - xci[0])], [max(0, xci[1] - x)]]), fmt="none", ecolor="#6B7280", elinewidth=0.8, capsize=2.2, alpha=0.75, zorder=2)
            ax_pending.scatter(x, 0, s=62, marker="o", facecolor="white", edgecolor="#5B6472", linewidth=1.0, alpha=0.9, zorder=4)
            label = short_labels.get(row.get("method_label", method_key), row.get("method_label", method_key))
            ax_pending.text(x, 0.18 + 0.17 * (idx % 2), label, ha="center", va="bottom", rotation=35, fontsize=7.0, color="#4B5563", linespacing=0.9)
        ax_pending.axhline(0, color="#AEB7C2", linewidth=0.9)
        ax_pending.set_ylim(-0.22, 1.1)
        ax_pending.set_yticks([])
        ax_pending.text(0.0, 0.08, "bias-only baselines\n(reasoning pending)", transform=ax_pending.transAxes, ha="left", va="bottom", fontsize=8.0, color="#4B5563", fontweight="semibold")
        ax_pending.spines["top"].set_visible(False)
        ax_pending.spines["right"].set_visible(False)
        ax_pending.spines["left"].set_visible(False)
        ax_pending.spines["bottom"].set_color("#AEB7C2")
    ax.set_xlabel("Mean bias error (lower is better)")
    ax.set_ylabel("Mean reasoning accuracy (%)")
    ax.set_title(f"{model_label}: bias-utility Pareto plot with bootstrap CIs", fontsize=15, fontweight="bold")
    ax.grid(color="#E5E7EB", linestyle=":", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for model_key, model_label in MODEL_DIRS:
        model_dir = FIGURES / model_key
        csv_dir = model_dir / "csv"
        pdf_dir = model_dir / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        debias_rows = add_ci_to_rows(model_key, model_label, "debiasing_method_bars", read_csv(csv_dir / "debiasing_method_bars_data.csv"), "bias")
        write_csv(csv_dir / "debiasing_method_bars_with_ci_data.csv", debias_rows)
        plot_debiasing(model_key, model_label, debias_rows, pdf_dir / "debiasing_method_bars_with_ci.pdf")

        baseline_rows = add_ci_to_rows(model_key, model_label, "baseline_comparison_lollipop", read_csv(csv_dir / "baseline_comparison_lollipop_data.csv"), "bias")
        write_csv(csv_dir / "baseline_comparison_lollipop_with_ci_data.csv", baseline_rows)
        plot_baseline(model_key, model_label, baseline_rows, pdf_dir / "baseline_comparison_lollipop_with_ci.pdf")

        intervention_rows = add_ci_to_rows(model_key, model_label, "intervention_ablation", read_csv(csv_dir / "intervention_ablation_data.csv"), "reduction")
        write_csv(csv_dir / "intervention_ablation_with_ci_data.csv", intervention_rows)
        plot_intervention(model_key, model_label, intervention_rows, pdf_dir / "intervention_ablation_with_ci.pdf")

        reasoning_rows = add_ci_to_rows(model_key, model_label, "reasoning_performance", read_csv(csv_dir / "reasoning_performance_data.csv"), "reasoning")
        write_csv(csv_dir / "reasoning_performance_with_ci_data.csv", reasoning_rows)
        plot_reasoning(model_key, model_label, reasoning_rows, pdf_dir / "reasoning_performance_with_ci.pdf")

        pareto_rows = read_csv(csv_dir / "bias_reasoning_pareto_data.csv")
        # Pareto points can come from either the baseline-comparison table
        # (existing baselines) or the core debiasing table (partial DIY rows for
        # models whose full baseline-comparison rows have not finished yet).
        # Use both CI sources so every plotted point gets uncertainty whenever
        # its underlying examples are available.
        plot_pareto(model_key, model_label, pareto_rows, [*baseline_rows, *debias_rows], reasoning_rows, pdf_dir / "bias_reasoning_pareto_with_ci.pdf")
        print(f"{model_label}: wrote CI overlays for debiasing, baselines, interventions, reasoning, and pareto")


if __name__ == "__main__":
    main()
