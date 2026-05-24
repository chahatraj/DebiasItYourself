#!/usr/bin/env python3
"""Bias-utility Pareto plot for DIY and baseline mitigation methods.

The x-axis is mean normalized bias error across the six bias panels used in the
main debiasing figures. The y-axis is mean reasoning accuracy across
ARC-Challenge, ARC-Easy, and Balanced COPA. Lower x and higher y are better.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.lines import Line2D

import plot_debiasing_method_bars as core


ROOT = core.ROOT
OUTDIR = core.OUTDIR
BIAS_INPUT = OUTDIR / "csv/baseline_comparison_lollipop_data.csv"
REASONING_INPUT = ROOT / "results/new_results/all_results_dataset_slides.test.csv"

REASONING_DATASETS = ["arc_challenge", "arc_easy", "balanced_copa"]
REASONING_LABELS = {
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "balanced_copa": "Balanced COPA",
}

BASELINE_LABELS = {
    "bba": "BBA",
    "cal": "CAL",
    "fairsteer": "FairSteer",
    "biasedit": "BiasEdit",
    "lftf": "LFTF",
    "dpo": "DPO",
    "peft": "PEFT",
    "debias_llms": "DebiasLLMs",
    "debias_nlg": "DebiasNLG",
    "reduce_social_bias": "RSB",
    "self_debiasing_reprompting": "SelfDebias",
}

METHOD_ORDER = [
    ("base", "Base Model", "base"),
    ("icl", "ICL", "icl"),
    *[(key, label, "baseline") for key, label in BASELINE_LABELS.items()],
    ("diy_instruction_tune", "DIY IT", "diy_tune"),
    ("diy_twopass", "DIY Two Pass (No IT)", "diy_twopass"),
    ("diy_tune_twopass", "DIY Two Pass (IT)", "diy_combo"),
]

STYLES = {
    "base": {
        "color": "#B8BFD6",
        "edge": "#20252A",
        "marker": "s",
        "size": 96,
        "zorder": 5,
    },
    "baseline": {
        "color": "#D9DEE9",
        "edge": "#374151",
        "marker": "o",
        "size": 92,
        "zorder": 3,
    },
    "icl": {
        "color": "#C4B5FD",
        "edge": "#20252A",
        "marker": "D",
        "size": 92,
        "zorder": 7,
    },
    "diy_tune": {
        "color": "#FF8FAB",
        "edge": "#20252A",
        "marker": "D",
        "size": 92,
        "zorder": 7,
    },
    "diy_twopass": {
        "color": "#5EEAD4",
        "edge": "#20252A",
        "marker": "D",
        "size": 92,
        "zorder": 8,
    },
    "diy_combo": {
        "color": "#FCD34D",
        "edge": "#20252A",
        "marker": "D",
        "size": 92,
        "zorder": 7,
    },
}

LABEL_OFFSETS = {
    "BBA": (0.24, 0.28),
    "CAL": (0.24, -0.42),
    "FairSteer": (0.00, 1.65),
    "BiasEdit": (0.24, -0.28),
    "LFTF": (0.24, -0.38),
    "DPO": (0.24, 0.24),
    "PEFT": (0.24, -0.34),
    "DebiasLLMs": (0.24, -0.34),
    "DebiasNLG": (0.24, 0.28),
    "RSB": (0.24, 0.42),
    "SelfDebias": (0.25, -0.42),
    "ICL": (0.24, 0.42),
    "DIY IT": (0.34, 0.26),
    "DIY Two Pass (No IT)": (-0.34, 2.45),
    "DIY Two Pass (IT)": (-1.05, 0.95),
}

SHORT_LABELS = {
    "RSB": "RSB",
    "SelfDebias": "SelfDebias",
    "ICL": "ICL",
    "DIY IT": "DIY\nIT",
    "DIY Two Pass (No IT)": "DIY Two Pass\n(No IT)",
    "DIY Two Pass (IT)": "DIY Two Pass\n(IT)",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_bias() -> dict[str, dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    groups: dict[str, str] = {}
    panels: dict[str, set[str]] = defaultdict(set)

    for row in read_csv(BIAS_INPUT):
        method_key = row["method_key"]
        labels[method_key] = row["method_label"]
        groups[method_key] = row["group"]
        grouped[method_key].append(float(row["normalized_bias_error_plotted"]))
        panels[method_key].add(row["panel_label"])

    bias = {}
    for method_key, values in grouped.items():
        bias[method_key] = {
            "method_label": labels[method_key],
            "group": groups[method_key],
            "mean_bias_error": float(np.mean(values)),
            "n_bias_panels": len(panels[method_key]),
            "bias_panels": "; ".join(sorted(panels[method_key])),
        }
    return bias


def select_reasoning_score(
    rows: list[dict[str, str]], dataset_key: str, method_key: str
) -> tuple[float, str] | None:
    if method_key == "icl":
        path = (
            ROOT
            / "results/new_results/m4_base_icl_zero/m4_base_icl_zero_reasoning_allmodels_20260514_011540/llama8b"
            / "reasoning"
            / dataset_key
            / f"m4_baseicl_zero_llama8b_allstrat_{dataset_key}"
            / f"{dataset_key}_metrics_overall_m4_baseicl_zero_llama8b_allstrat_{dataset_key}.csv"
        )
        if not path.exists():
            return None
        with path.open(newline="") as f:
            metric_rows = list(csv.DictReader(f))
        return float(metric_rows[-1]["accuracy"]) * 100.0, str(path.relative_to(ROOT))

    if method_key == "base":
        model = f"reasoning_check__base__{dataset_key}"
        candidates = [
            r
            for r in rows
            if r["dataset_key"] == dataset_key
            and r["type"] == "method"
            and r["name"] == "reasoning_check"
            and r["model"] == model
            and core.valid_score(r)
        ]
    elif method_key == "diy_instruction_tune":
        model = (
            "reasoning_check__finetuned_ms-500-allstrategies-"
            f"opinion-action-event-allversions__{dataset_key}"
        )
        candidates = [
            r
            for r in rows
            if r["dataset_key"] == dataset_key
            and r["type"] == "method"
            and r["name"] == "reasoning_check"
            and r["model"] == model
            and core.valid_score(r)
        ]
    elif method_key == "diy_twopass":
        model = f"m6_two_pass__base__all_strategies__{dataset_key}"
        candidates = [
            r
            for r in rows
            if r["dataset_key"] == dataset_key
            and r["type"] == "method"
            and r["name"] == "m6_self_debiasing"
            and r["model"] == model
            and core.valid_score(r)
        ]
    elif method_key == "diy_tune_twopass":
        model = (
            "m6_two_pass__finetuned_ms-500-allstrategies-"
            f"opinion-action-event-allversions__all_strategies__{dataset_key}"
        )
        candidates = [
            r
            for r in rows
            if r["dataset_key"] == dataset_key
            and r["type"] == "method"
            and r["name"] == "m6_self_debiasing"
            and r["model"] == model
            and core.valid_score(r)
        ]
    else:
        candidates = [
            r
            for r in rows
            if r["dataset_key"] == dataset_key
            and r["type"] == "baseline"
            and r["name"] == method_key
            and core.valid_score(r)
        ]

    if not candidates:
        return None
    row = sorted(candidates, key=lambda r: r["source_file"])[0]
    return float(row["score"]) * 100.0, row["source_file"]


def collect_records() -> list[dict[str, str]]:
    bias = collect_bias()
    reasoning_rows = read_csv(REASONING_INPUT)
    records = []

    for method_key, method_label, group in METHOD_ORDER:
        if method_key not in bias:
            continue

        reasoning_values = []
        reasoning_sources = []
        for dataset_key in REASONING_DATASETS:
            selected = select_reasoning_score(reasoning_rows, dataset_key, method_key)
            if selected is None:
                continue
            value, source = selected
            reasoning_values.append(value)
            reasoning_sources.append(f"{REASONING_LABELS[dataset_key]}={source}")

        plotted = bias[method_key]["n_bias_panels"] == 6 and len(reasoning_values) == 3
        records.append(
            {
                "method_key": method_key,
                "method_label": method_label,
                "group": group,
                "mean_bias_error": f"{bias[method_key]['mean_bias_error']:.6g}",
                "mean_reasoning_accuracy": (
                    f"{float(np.mean(reasoning_values)):.6g}" if reasoning_values else ""
                ),
                "n_bias_panels": str(bias[method_key]["n_bias_panels"]),
                "n_reasoning_benchmarks": str(len(reasoning_values)),
                "plotted": str(plotted),
                "bias_panels": str(bias[method_key]["bias_panels"]),
                "reasoning_sources": " | ".join(reasoning_sources),
            }
        )

    return records


def is_pareto(points: list[dict[str, str]]) -> dict[str, bool]:
    result = {}
    for point in points:
        x = float(point["mean_bias_error"])
        y = float(point["mean_reasoning_accuracy"])
        dominated = False
        for other in points:
            if point is other:
                continue
            ox = float(other["mean_bias_error"])
            oy = float(other["mean_reasoning_accuracy"])
            if ox <= x and oy >= y and (ox < x or oy > y):
                dominated = True
                break
        result[point["method_key"]] = not dominated
    return result


def write_csv(records: list[dict[str, str]]) -> None:
    plotted = [r for r in records if r["plotted"] == "True"]
    frontier = is_pareto(plotted)
    out = []
    for row in records:
        row = dict(row)
        row["pareto_frontier"] = str(frontier.get(row["method_key"], False))
        out.append(row)

    path = OUTDIR / "csv/bias_reasoning_pareto_data.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)


def plot(records: list[dict[str, str]]) -> None:
    points = [
        r
        for r in records
        if r["mean_reasoning_accuracy"]
        and r["group"] in {"baseline", "icl", "diy_tune", "diy_twopass", "diy_combo"}
    ]

    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.8,
            "axes.titlesize": 10,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    sorted_points = sorted(points, key=lambda r: float(r["mean_bias_error"]))
    if len(sorted_points) >= 2:
        ax.plot(
            [float(r["mean_bias_error"]) for r in sorted_points],
            [float(r["mean_reasoning_accuracy"]) for r in sorted_points],
            color="#111827",
            linestyle=(0, (1.4, 2.4)),
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )

    for row in points:
        group = row["group"]
        style = STYLES[group if group in STYLES else "baseline"]
        x = float(row["mean_bias_error"])
        y = float(row["mean_reasoning_accuracy"])
        ax.scatter(
            x,
            y,
            s=style["size"],
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor=style["edge"],
            linewidth=1.05,
            alpha=0.98 if group.startswith("diy") or group == "base" else 0.72,
            zorder=style["zorder"],
        )
        dx, dy = LABEL_OFFSETS.get(row["method_label"], (0.16, 0.25))
        label = SHORT_LABELS.get(row["method_label"], row["method_label"])
        ax.text(
            x + dx,
            y + dy,
            label,
            ha="left",
            va="center",
            fontsize=6.7,
            fontweight="medium",
            color="#374151",
            linespacing=0.9,
            zorder=10,
        )

    xs = [float(r["mean_bias_error"]) for r in points]
    ys = [float(r["mean_reasoning_accuracy"]) for r in points]
    ax.set_xlim(max(0, min(xs) - 0.55), max(xs) + 0.75)
    ax.set_ylim(min(ys) - 2.0, max(ys) + 1.5)
    ax.set_xlabel("Mean bias error", fontweight="semibold")
    ax.set_ylabel("Reasoning accuracy (%)", fontweight="semibold")
    ax.set_title("")
    ax.grid(color="#E4E8F0", linestyle=":", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.tick_params(colors="#374151")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STYLES["baseline"]["color"],
            markeredgecolor="#374151",
            markeredgewidth=1.0,
            markersize=5.5,
            label="Baselines",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=STYLES["icl"]["color"],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.4,
            label="ICL",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=STYLES["diy_tune"]["color"],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.4,
            label="DIY IT",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=STYLES["diy_twopass"]["color"],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.4,
            label="DIY Two Pass (No IT)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=STYLES["diy_combo"]["color"],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.4,
            label="DIY Two Pass (IT)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#B9C2CF",
        borderpad=0.42,
        columnspacing=0.9,
        handlelength=1.35,
    )

    fig.text(
        0.5,
        0.982,
        "Bias and reasoning performance",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="semibold",
        color="#111827",
    )

    fig.text(
        0.01,
        0.036,
        "Bias error averages six normalized bias metrics; reasoning averages ARC-Challenge, ARC-Easy, and Balanced COPA.\n"
        "Lower bias and higher reasoning accuracy are better. All baseline points and all three DIY variants are shown.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#374151",
        linespacing=1.2,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.75)
    fig.savefig(OUTDIR / "pdf/bias_reasoning_pareto.pdf", bbox_inches="tight")


def main() -> None:
    records = collect_records()
    if not records:
        raise RuntimeError("No Pareto records collected.")
    write_csv(records)
    plot(records)


if __name__ == "__main__":
    main()
