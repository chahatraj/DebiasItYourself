#!/usr/bin/env python3
"""Reasoning benchmark performance for core DIY settings.

This figure reports raw accuracy on reasoning/utility benchmarks. Unlike the
bias figures, scores are not normalized: higher accuracy is better.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.patches import Patch

import plot_debiasing_method_bars as core


ROOT = core.ROOT
INPUT = core.INPUT
OUTDIR = core.OUTDIR
ICL_REASONING_ROOT = (
    ROOT
    / "results/new_results/m4_base_icl_zero/m4_base_icl_zero_reasoning_allmodels_20260514_011540/llama8b"
)

BENCHMARKS = [
    ("arc_challenge", "ARC-Challenge"),
    ("arc_easy", "ARC-Easy"),
    ("balanced_copa", "Balanced COPA"),
]

METHODS = [
    ("base", "Base Model\nInference"),
    ("icl", "ICL"),
    ("instruction_tune", "DIY IT"),
    ("twopass", "DIY Two Pass\n(No IT)"),
    ("tune_twopass", "DIY Two Pass\n(IT)"),
]

COLORS = {
    "base": "#B8BFD6",
    "icl": "#C4B5FD",
    "instruction_tune": "#FF8FAB",
    "twopass": "#5EEAD4",
    "tune_twopass": "#FCD34D",
}

HATCHES = {
    "base": "",
    "icl": "xx",
    "instruction_tune": "///",
    "twopass": "\\\\\\",
    "tune_twopass": "...",
}


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="") as f:
        return list(csv.DictReader(f))


def score(row: dict[str, str]) -> float:
    return float(row["score"]) * 100.0


def icl_reasoning_path(dataset_key: str):
    return (
        ICL_REASONING_ROOT
        / "reasoning"
        / dataset_key
        / f"m4_baseicl_zero_llama8b_allstrat_{dataset_key}"
        / f"{dataset_key}_metrics_overall_m4_baseicl_zero_llama8b_allstrat_{dataset_key}.csv"
    )


def read_accuracy(path) -> float:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["accuracy"])


def select_reasoning_row(
    rows: list[dict[str, str]], dataset_key: str, method_key: str
) -> dict[str, str] | None:
    dataset_rows = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key
        and r["type"] == "method"
        and r["score_label"] == "accuracy"
        and core.valid_score(r)
    ]

    if method_key == "base":
        model = f"reasoning_check__base__{dataset_key}"
        candidates = [
            r for r in dataset_rows if r["name"] == "reasoning_check" and r["model"] == model
        ]
        return candidates[0] if candidates else None

    if method_key == "instruction_tune":
        model = (
            "reasoning_check__finetuned_ms-500-allstrategies-"
            f"opinion-action-event-allversions__{dataset_key}"
        )
        candidates = [
            r for r in dataset_rows if r["name"] == "reasoning_check" and r["model"] == model
        ]
        return candidates[0] if candidates else None

    if method_key == "twopass":
        model = f"m6_two_pass__base__all_strategies__{dataset_key}"
        candidates = [
            r for r in dataset_rows if r["name"] == "m6_self_debiasing" and r["model"] == model
        ]
        return candidates[0] if candidates else None

    if method_key == "tune_twopass":
        model = (
            "m6_two_pass__finetuned_ms-500-allstrategies-"
            f"opinion-action-event-allversions__all_strategies__{dataset_key}"
        )
        candidates = [
            r for r in dataset_rows if r["name"] == "m6_self_debiasing" and r["model"] == model
        ]
        return candidates[0] if candidates else None

    raise ValueError(f"Unknown method key: {method_key}")


def collect_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for dataset_key, benchmark_label in BENCHMARKS:
        for method_key, method_label in METHODS:
            if method_key == "icl":
                path = icl_reasoning_path(dataset_key)
                if not path.exists():
                    continue
                accuracy = read_accuracy(path)
                records.append(
                    {
                        "dataset_key": dataset_key,
                        "benchmark_label": benchmark_label,
                        "method_key": method_key,
                        "method_label": method_label.replace("\n", " "),
                        "metric": "accuracy",
                        "accuracy": f"{accuracy:.8g}",
                        "accuracy_percent_plotted": f"{accuracy * 100.0:.6g}",
                        "name": "m4_base_icl_zero",
                        "model": f"m4_baseicl_zero_llama8b_allstrat_{dataset_key}",
                        "strategy": "all_strategies",
                        "settings": "zero-shot ICL",
                        "source_file": str(path.relative_to(ROOT)),
                    }
                )
                continue
            row = select_reasoning_row(rows, dataset_key, method_key)
            if row is None:
                continue
            records.append(
                {
                    "dataset_key": dataset_key,
                    "benchmark_label": benchmark_label,
                    "method_key": method_key,
                    "method_label": method_label.replace("\n", " "),
                    "metric": "accuracy",
                    "accuracy": f"{float(row['score']):.8g}",
                    "accuracy_percent_plotted": f"{score(row):.6g}",
                    "name": row["name"],
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "settings": row["settings"],
                    "source_file": row["source_file"],
                }
            )
    return records


def write_csv(records: list[dict[str, str]]) -> None:
    path = OUTDIR / "csv/reasoning_performance_data.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot(records: list[dict[str, str]]) -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11.3,
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.8,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 10.7,
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    values = {
        (r["dataset_key"], r["method_key"]): float(r["accuracy_percent_plotted"])
        for r in records
    }

    fig, ax = plt.subplots(figsize=(10.6, 5.15))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    x = np.arange(len(BENCHMARKS))
    width = 0.15
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(METHODS))

    for offset, (method_key, method_label) in zip(offsets, METHODS):
        ys = [values.get((dataset_key, method_key), np.nan) for dataset_key, _ in BENCHMARKS]
        bars = ax.bar(
            x + offset,
            ys,
            width=width,
            label=method_label.replace("\n", " "),
            color=COLORS[method_key],
            edgecolor="#20252A",
            linewidth=1.05,
            zorder=3,
        )
        for bar in bars:
            bar.set_hatch(HATCHES[method_key])
        for bar, value in zip(bars, ys):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.85,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=10.4,
                fontweight="bold",
                color="#111827",
                rotation=0,
            )

    ax.set_title(
        "Reasoning benchmark performance",
        pad=10,
        fontsize=16.5,
        fontweight="bold",
        color="#111827",
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12.8, fontweight="semibold", color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in BENCHMARKS], fontweight="bold")
    ax.set_ylim(55, 86)
    ax.set_yticks(np.arange(55, 87, 5))
    ax.tick_params(axis="x", length=0, pad=9, colors="#1F2937")
    ax.tick_params(axis="y", colors="#374151")
    ax.grid(axis="y", color="#D7DCE5", linewidth=0.85, linestyle="--", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")

    handles = [
        Patch(
            facecolor=COLORS[method_key],
            edgecolor="#20252A",
            hatch=HATCHES[method_key],
            linewidth=1.05,
            label=method_label.replace("\n", " "),
        )
        for method_key, method_label in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        edgecolor="#B9C2CF",
        handlelength=1.35,
        columnspacing=1.0,
        borderpad=0.55,
    )

    fig.text(
        0.01,
        0.026,
        "Metric: raw benchmark accuracy; higher is better. ICL and DIY settings use all bias-reducing interventions.",
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0.015, 0.082, 1, 0.875))
    fig.savefig(OUTDIR / "pdf/reasoning_performance.pdf", bbox_inches="tight")


def main() -> None:
    rows = load_rows()
    records = collect_records(rows)
    if not records:
        raise RuntimeError("No reasoning records found.")
    write_csv(records)
    plot(records)


if __name__ == "__main__":
    main()
