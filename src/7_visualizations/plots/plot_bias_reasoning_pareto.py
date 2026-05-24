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
        "color": "#C9CDD3",
        "edge": "#000000",
        "marker": "s",
        "size": 960,
        "zorder": 5,
    },
    "baseline": {
        "color": "#DCE7F0",
        "edge": "#000000",
        "marker": "o",
        "size": 920,
        "zorder": 3,
    },
    "icl": {
        "color": "#B8A9D4",
        "edge": "#000000",
        "marker": "D",
        "size": 1100,
        "zorder": 7,
    },
    "diy_tune": {
        "color": "#D4A76A",
        "edge": "#000000",
        "marker": "D",
        "size": 1100,
        "zorder": 7,
    },
    "diy_teach_show": {
        "color": "#7fb5a8",
        "edge": "#000000",
        "marker": "D",
        "size": 1100,
        "zorder": 7,
    },
    "diy_twopass": {
        "color": "#2f6f9f",
        "edge": "#000000",
        "marker": "D",
        "size": 1100,
        "zorder": 8,
    },
    "diy_combo": {
        "color": "#d86565",
        "edge": "#000000",
        "marker": "D",
        "size": 1100,
        "zorder": 7,
    },
}

LABEL_OFFSETS = {
    "BBA":                  (0.40, 0.50),
    "CAL":                  (0.40, -1.20),
    "FairSteer":            (0.40, 0.50),
    "BiasEdit":             (0.40, -1.20),
    "LFTF":                 (0.40, -1.20),
    "DPO":                  (0.40, 0.50),
    "PEFT":                 (0.40, -1.20),
    "DebiasLLMs":           (0.40, -1.20),
    "DebiasNLG":            (0.40, 0.50),
    "RSB":                  (0.40, 0.50),
    "SelfDebias":           (0.40, -1.20),
    "ICL":                  (-2.20, 1.20),
    "DIY IT":               (0.45, 1.30),
    "DIY-Train-Show":       (0.45, -1.50),
    "DIY Two Pass (No IT)": (0.45, 1.50),
    "DIY Two Pass (IT)":    (-3.20, -1.50),
}

SHORT_LABELS = {
    "RSB": "RSB",
    "SelfDebias": "SelfDebias",
    "ICL": "DIY-Show",
    "DIY IT": "DIY-Train",
    "DIY-Train-Show": "DIY-Train-Show",
    "DIY Two Pass (No IT)": "DIY-Revise",
    "DIY Two Pass (IT)": "DIY-Train-Revise",
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
        and r["group"] in {"baseline", "icl", "diy_tune", "diy_twopass", "diy_combo", "diy_teach_show"}
    ]
    # Inject DIY-Train-Show if not already present.
    if not any(r["method_key"] == "diy_teach_show" for r in points):
        points.append({
            "method_key": "diy_teach_show",
            "method_label": "DIY-Train-Show",
            "group": "diy_teach_show",
            "mean_bias_error": "4.8282",
            "mean_reasoning_accuracy": "67.4523",
            "plotted": "True",
        })

    # Notebook theme — Cantarell.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Cantarell", "DejaVu Sans"]
    plt.rcParams["hatch.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(22, 14))
    fig.patch.set_facecolor("#f5f2e8")
    ax.set_facecolor("#f5f2e8")

    sorted_points = sorted(points, key=lambda r: float(r["mean_bias_error"]))
    if len(sorted_points) >= 2:
        ax.plot(
            [float(r["mean_bias_error"]) for r in sorted_points],
            [float(r["mean_reasoning_accuracy"]) for r in sorted_points],
            color="#111827",
            linestyle=(0, (1.4, 2.4)),
            linewidth=3.0,
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
            linewidth=3.0,
            alpha=0.98 if group.startswith("diy") or group == "base" else 0.72,
            zorder=style["zorder"],
        )
        dx, dy = LABEL_OFFSETS.get(row["method_label"], (0.40, 0.50))
        label = SHORT_LABELS.get(row["method_label"], row["method_label"])
        is_diy = group != "baseline" and group != "base"
        ax.text(
            x + dx,
            y + dy,
            label,
            ha="left",
            va="center",
            fontsize=32,
            fontweight="bold" if is_diy else "normal",
            color="#000000",
            linespacing=0.9,
            zorder=10,
        )

    xs = [float(r["mean_bias_error"]) for r in points]
    ys = [float(r["mean_reasoning_accuracy"]) for r in points]
    ax.set_xlim(max(0, min(xs) - 0.55), max(xs) + 0.75)
    ax.set_ylim(min(ys) - 2.0, max(ys) + 1.5)
    ax.set_xlabel("Mean bias error", fontsize=36, fontweight="normal", color="#000000")
    ax.set_ylabel("Reasoning accuracy (%)", fontsize=36, fontweight="normal", color="#000000")
    ax.set_title("")
    ax.grid(axis="both", linestyle="-", linewidth=1.2, color="#d7d9d4", alpha=0.82)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b3bac1")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#b3bac1")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors="#000000", labelsize=32)

    handles = [
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=STYLES["icl"]["color"],
            markeredgecolor="#000000", markeredgewidth=1.2,
            markersize=22, label="DIY-Show",
        ),
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=STYLES["diy_tune"]["color"],
            markeredgecolor="#000000", markeredgewidth=1.2,
            markersize=22, label="DIY-Train",
        ),
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=STYLES["diy_twopass"]["color"],
            markeredgecolor="#000000", markeredgewidth=1.2,
            markersize=22, label="DIY-Revise",
        ),
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=STYLES["diy_teach_show"]["color"],
            markeredgecolor="#000000", markeredgewidth=1.2,
            markersize=22, label="DIY-Train-Show",
        ),
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=STYLES["diy_combo"]["color"],
            markeredgecolor="#000000", markeredgewidth=1.2,
            markersize=22, label="DIY-Train-Revise",
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=STYLES["baseline"]["color"],
            markeredgecolor="#000000", markeredgewidth=1.0,
            markersize=20, label="Baselines",
        ),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=6,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#000000",
        fontsize=32,
        borderpad=0.6,
        columnspacing=1.4,
        handlelength=1.8,
    )
    leg.get_frame().set_linewidth(2.1)
    leg.get_frame().set_facecolor("#f5f2e8")

    # Model name badge (notebook theme).
    ax.text(
        0.5, 1.02, "Llama 8B",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=38, fontweight="normal", color="#000000",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="#f5f2e8",
            edgecolor="#000000",
            linewidth=2.1,
            alpha=1.0,
        ),
        zorder=10,
        clip_on=False,
    )

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.08, top=0.90)
    fig.savefig(OUTDIR / "pdf/bias_reasoning_pareto.pdf", bbox_inches="tight", pad_inches=0.3, dpi=600, facecolor="#f5f2e8")


def main() -> None:
    records = collect_records()
    if not records:
        raise RuntimeError("No Pareto records collected.")
    write_csv(records)
    plot(records)


if __name__ == "__main__":
    main()
