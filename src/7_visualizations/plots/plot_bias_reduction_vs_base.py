#!/usr/bin/env python3
"""Bias-reduction-vs-base plots for the final figure staging area.

The existing debiasing plots show absolute bias error:
    abs(native_metric - unbiased_target)

This script shows the complementary delta view:
    base_bias_error - method_bias_error

Positive values mean the method reduced bias relative to the base model.
Negative values mean the method increased bias relative to the base model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "figures"
FINAL_FIGURES = ROOT / "final_figures"
FT_ICL_ZERO_ROOT = (
    ROOT
    / "results/new_results/m4_ft_icl_zero/m4_ft_icl_zero_allmodels_20260514_143756"
)

PANELS = [
    ("crowspairs", "CrowS-Pairs", "stereotype_preference_pct", 50.0),
    ("stereoset", "StereoSet", "SS Score", 50.0),
    ("bbq", "BBQ Ambig.", "Bias_score_ambig", 0.0),
    ("bbq", "BBQ Disambig.", "Bias_score_disambig", 0.0),
    ("winobias", "WinoBias", "abs_pro_anti_gap", 0.0),
    (
        "winogender",
        "WinoGender",
        "male_female_pair_disagreement_rate",
        0.0,
    ),
]

MODEL_INFO = {
    "llama8b": {
        "title": "Llama 8B",
        "ft_icl_dir": "llama8b",
        "ft_icl_prefix": "llama8b",
    },
    "qwen": {
        "title": "Qwen",
        "ft_icl_dir": "qwen35_27b",
        "ft_icl_prefix": "qwen35_27b",
    },
    "llama70b": {
        "title": "Llama 70B",
        "ft_icl_dir": "llama70b",
        "ft_icl_prefix": "llama70b",
    },
}

METHODS = [
    ("ICL", "DIY ICL\n1-pass"),
    ("DIY Two Pass (No IT)", "DIY ICL\n2-pass"),
    ("DIY IT", "DIY IT\n1-pass"),
    ("DIY IT + ICL", "DIY IT + ICL\n1-pass"),
    ("DIY Two Pass (IT)", "DIY IT + ICL\n2-pass"),
]

COLORS = {
    "ICL": "#BFA7FF",
    "DIY Two Pass (No IT)": "#75D6A0",
    "DIY IT": "#7CC8EA",
    "DIY IT + ICL": "#F4A7C5",
    "DIY Two Pass (IT)": "#F6C75B",
}

HATCHES = {
    "ICL": "xx",
    "DIY Two Pass (No IT)": "\\\\\\",
    "DIY IT": "///",
    "DIY IT + ICL": "++",
    "DIY Two Pass (IT)": "...",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | int | float | None) -> float:
    if value is None:
        return float("nan")
    return float(str(value).strip())


def bias_error(native_value: float, ideal: float) -> float:
    return abs(native_value - ideal)


def method_rows_from_existing_csv(model_key: str) -> dict[tuple[str, str, str], float]:
    path = FIGURES / model_key / "csv/debiasing_method_bars_data.csv"
    rows = read_csv(path)
    values: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["panel_label"], row.get("method") or row.get("method_label"), row["metric"])
        values[key] = fnum(row["normalized_bias_error_plotted"])
    return values


def ft_icl_zero_metric_path(model_key: str, dataset_key: str) -> Path:
    info = MODEL_INFO[model_key]
    model_dir = str(info["ft_icl_dir"])
    prefix = str(info["ft_icl_prefix"])
    root = FT_ICL_ZERO_ROOT / model_dir
    tag = "all_allversions"
    run = f"m4_fticl_zero_{prefix}_{tag}"

    if dataset_key == "bbq":
        return root / f"bbq/bbq_metrics_{run}_bbq.csv"
    if dataset_key == "crowspairs":
        return root / f"evalshared/crows_pairs_metrics_overall_{run}_crowspairs.csv"
    if dataset_key == "stereoset":
        return root / f"evalshared/stereoset_metrics_{run}_stereoset.csv"
    if dataset_key == "winobias":
        return (
            root
            / f"evalshared/winobias/{run}_winobias/winobias_metrics_overall_{run}_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            root
            / f"evalshared/winogender/{run}_winogender/winogender_metrics_overall_{run}_winogender.csv"
        )
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def read_native_metric(path: Path, dataset_key: str, metric: str) -> float:
    rows = read_csv(path)
    if dataset_key == "stereoset":
        rows = [
            r
            for r in rows
            if r.get("split") == "overall" and r.get("domain") == "overall"
        ]
    elif dataset_key == "bbq":
        rows = [
            r
            for r in rows
            if r.get("input_file") == "__overall__" or r.get("Model", "").startswith("m4_fticl")
        ]
    if not rows:
        raise RuntimeError(f"No metric rows found in {path}")
    if metric not in rows[-1]:
        raise KeyError(f"{metric} missing from {path}")
    return fnum(rows[-1][metric])


def ft_icl_zero_bias_errors(model_key: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for dataset_key, panel_label, metric, ideal in PANELS:
        path = ft_icl_zero_metric_path(model_key, dataset_key)
        native = read_native_metric(path, dataset_key, metric)
        values[(panel_label, metric)] = bias_error(native, ideal)
    return values


def build_records(model_key: str) -> list[dict[str, object]]:
    existing = method_rows_from_existing_csv(model_key)
    ft_icl = ft_icl_zero_bias_errors(model_key)
    records: list[dict[str, object]] = []

    for _dataset_key, panel_label, metric, _ideal in PANELS:
        base_error = existing[(panel_label, "Base Model Inference", metric)]
        for method_key, label in METHODS:
            if method_key == "DIY IT + ICL":
                method_error = ft_icl[(panel_label, metric)]
            else:
                method_error = existing[(panel_label, method_key, metric)]
            records.append(
                {
                    "model": model_key,
                    "panel_label": panel_label,
                    "metric": metric,
                    "method_key": method_key,
                    "method_label": label,
                    "base_bias_error": base_error,
                    "method_bias_error": method_error,
                    "bias_reduction_vs_base": base_error - method_error,
                }
            )
    return records


def nice_value(value: float) -> str:
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def plot_model(model_key: str) -> Path:
    records = build_records(model_key)
    info = MODEL_INFO[model_key]
    outdir = FINAL_FIGURES / model_key / "pdf"
    outdir.mkdir(parents=True, exist_ok=True)

    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.4), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(METHODS))
    width = 0.72

    all_deltas = [float(r["bias_reduction_vs_base"]) for r in records]
    ymin = min(0.0, min(all_deltas)) - 1.2
    ymax = max(0.0, max(all_deltas)) + 1.6

    for ax, (_dataset_key, panel_label, _metric, _ideal) in zip(axes, PANELS):
        panel_records = [r for r in records if r["panel_label"] == panel_label]
        deltas = [float(r["bias_reduction_vs_base"]) for r in panel_records]
        method_keys = [str(r["method_key"]) for r in panel_records]
        bars = ax.bar(
            x,
            deltas,
            width=width,
            color=[COLORS[m] for m in method_keys],
            edgecolor="#263241",
            linewidth=1.05,
            zorder=3,
        )
        for bar, method_key in zip(bars, method_keys):
            bar.set_hatch(HATCHES[method_key])
            value = bar.get_height()
            va = "bottom" if value >= 0 else "top"
            offset = 0.30 if value >= 0 else -0.30
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                nice_value(value),
                ha="center",
                va=va,
                fontsize=8.7,
                color="#263241",
            )

        ax.axhline(0, color="#1F2937", linewidth=1.1, zorder=2)
        ax.set_title(panel_label, fontweight="bold", pad=8, color="#111827")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in METHODS], rotation=0, ha="center")
        ax.set_ylim(ymin, ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#A7B0BF")
        ax.spines["bottom"].set_color("#A7B0BF")
        ax.tick_params(colors="#344052")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7, linestyle="-", alpha=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Bias reduction vs. base", fontweight="semibold")
    axes[3].set_ylabel("Bias reduction vs. base", fontweight="semibold")

    handles = [
        Patch(
            facecolor=COLORS[method_key],
            edgecolor="#263241",
            linewidth=1.05,
            hatch=HATCHES[method_key],
            label=label.replace("\n", " "),
        )
        for method_key, label in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#CBD5E1",
    )
    fig.suptitle(
        f"{info['title']}: bias reduction relative to the base model",
        y=1.02,
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.5,
        0.016,
        "Delta = base bias error - method bias error. Positive bars reduce bias; negative bars increase bias relative to base.",
        ha="center",
        fontsize=11.2,
        color="#4B5563",
    )
    fig.subplots_adjust(top=0.83, bottom=0.15, left=0.07, right=0.985, wspace=0.22, hspace=0.42)

    outpath = outdir / "bias_reduction_vs_base.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    for model_key in MODEL_INFO:
        path = plot_model(model_key)
        print(path)


if __name__ == "__main__":
    main()
