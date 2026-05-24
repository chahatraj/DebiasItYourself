#!/usr/bin/env python3
"""Fine-grained diagnostic figures for the DIY results.

The script consumes the audit CSVs produced by the main plotting scripts and
writes derived data plus paper-style PDF/PNG figures under
figures/llama8b/finegrained/multi_axis/. The goal is broad coverage of the result axes we already
have: interventions, methods, shots, finetuned checkpoint versions, datasets,
baselines, and reasoning/utility.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PLOT_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PLOT_DIR))

from plot_style import use_nimbus_sans  # noqa: E402
import plot_debiasing_method_bars as core  # noqa: E402


OUT = ROOT / "figures/llama8b/finegrained/multi_axis"
CSV_OUT = OUT / "csv"
PDF_OUT = OUT / "pdf"
SOURCE_CSV = ROOT / "figures/llama8b/csv"

PANEL_ORDER = [
    "CrowS-Pairs",
    "StereoSet",
    "BBQ Ambig.",
    "BBQ Disambig.",
    "WinoBias",
    "WinoGender",
]

REASONING_ORDER = [
    ("arc_challenge", "ARC-Challenge"),
    ("arc_easy", "ARC-Easy"),
    ("balanced_copa", "Balanced COPA"),
]

PRIMARY_STRATEGY_SLUGS = {
    "all_strategies_all_versions",
    "stereotype_replacement_all_versions",
    "individuation_all_versions",
    "perspective_taking_all_versions",
    "counter_stereotypic_imaging_all_versions",
    "positive_contact_all_versions",
}

METHOD_ORDER = [
    "DIY IT",
    "DIY Two Pass (No IT), 0-shot",
    "DIY Two Pass (No IT), 1-shot",
    "DIY Two Pass (No IT), 2-shot",
    "DIY Two Pass (IT), 0-shot",
    "DIY Two Pass (IT), 1-shot",
    "DIY Two Pass (IT), 2-shot",
]

METHOD_FAMILIES = ["DIY IT", "Two Pass (No IT)", "Two Pass (IT)"]

METHOD_COLORS = {
    "Base Model Inference": "#BBC2CE",
    "Base Model": "#BBC2CE",
    "DIY IT": "#7CC7F2",
    "Two Pass (No IT)": "#50C77B",
    "DIY Two Pass (No IT), 0-shot": "#B6EAC7",
    "DIY Two Pass (No IT), 1-shot": "#84D9A5",
    "DIY Two Pass (No IT), 2-shot": "#50C77B",
    "Two Pass (IT)": "#FF9E45",
    "DIY Two Pass (IT), 0-shot": "#FFD8A7",
    "DIY Two Pass (IT), 1-shot": "#FFB86B",
    "DIY Two Pass (IT), 2-shot": "#FF9E45",
}

STRATEGY_COLORS = {
    "All interventions": "#8EA7E9",
    "Stereotype replacement": "#7DD3FC",
    "Individuation": "#5EEAD4",
    "Perspective-taking": "#A7F3D0",
    "Counter-stereotypic imaging": "#FDBA74",
    "Positive contact": "#C4B5FD",
}

CHECKPOINT_COLORS = {
    "all-version": "#FF9E45",
    "opinion": "#7CC7F2",
    "action": "#50C77B",
    "event": "#C4B5FD",
    "matched all-version": "#FF9E45",
}


def ensure_dirs() -> None:
    for path in (CSV_OUT, PDF_OUT):
        path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.8,
            "axes.titlesize": 12.8,
            "axes.labelsize": 11.4,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.3,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(PDF_OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fnum(value: str | float | int | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def nice_value(value: float) -> str:
    if abs(value) < 1:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def method_family(method: str) -> str:
    if method == "DIY IT":
        return "DIY IT"
    if "No IT" in method:
        return "Two Pass (No IT)"
    if "Two Pass (IT)" in method:
        return "Two Pass (IT)"
    if method.startswith("Base"):
        return "Base"
    return method


def shot_value(method: str, shot: str | None) -> int | None:
    if shot not in (None, ""):
        return int(float(shot))
    match = re.search(r"(\d)-shot", method)
    return int(match.group(1)) if match else None


def compact_method(method: str) -> str:
    return (
        method.replace("DIY Two Pass (No IT), ", "No IT ")
        .replace("DIY Two Pass (IT), ", "IT ")
        .replace("-shot", "")
    )


def compact_config_label(row: dict[str, object], include_checkpoint: bool = True) -> str:
    strategy = str(row["strategy_label"])
    method = str(row["method"])
    checkpoint = str(row.get("ft_checkpoint", ""))
    parts = [strategy, compact_method(method)]
    if include_checkpoint and "IT" in method and checkpoint:
        parts.append(checkpoint)
    return " | ".join(parts)


def load_sources() -> dict[str, list[dict[str, str]]]:
    return {
        "configs": read_csv(SOURCE_CSV / "debiasing_method_bars_by_shot_configs_data.csv"),
        "interventions": read_csv(SOURCE_CSV / "intervention_ablation_data.csv"),
        "baselines": read_csv(SOURCE_CSV / "baseline_comparison_lollipop_data.csv"),
        "pareto": read_csv(SOURCE_CSV / "bias_reasoning_pareto_data.csv"),
        "reasoning": read_csv(SOURCE_CSV / "reasoning_performance_data.csv"),
        "raw": read_csv(core.INPUT),
    }


def config_means(config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in config_rows:
        if row["method"].startswith("Base"):
            continue
        key = (row["figure_slug"], row["method"], row["shot"])
        groups[key].append(row)

    out: list[dict[str, object]] = []
    for (slug, method, shot), rows in groups.items():
        if len(rows) < len(PANEL_ORDER):
            continue
        errors = [fnum(r["normalized_bias_error_plotted"]) for r in rows]
        reductions = []
        for r in rows:
            base = base_error_for(config_rows, r["panel_label"])
            reductions.append(base - fnum(r["normalized_bias_error_plotted"]))
        first = rows[0]
        out.append(
            {
                "figure_slug": slug,
                "figure_title": first["figure_title"],
                "strategy_key": first["strategy_key"],
                "strategy_label": first["strategy_label"],
                "ft_checkpoint": first["ft_checkpoint"],
                "method": method,
                "method_family": method_family(method),
                "shot": "" if shot == "" else str(int(float(shot))),
                "mean_bias_error": float(np.mean(errors)),
                "median_bias_error": float(np.median(errors)),
                "mean_reduction_vs_base": float(np.mean(reductions)),
                "n_panels": len(rows),
            }
        )
    out.sort(key=lambda r: (float(r["mean_bias_error"]), str(r["figure_slug"]), str(r["method"])))
    return out


def base_error_for(config_rows: list[dict[str, str]], panel_label: str) -> float:
    for row in config_rows:
        if row["method"].startswith("Base") and row["panel_label"] == panel_label:
            return fnum(row["normalized_bias_error_plotted"])
    raise KeyError(panel_label)


def config_panel_records(config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in config_rows:
        if row["method"].startswith("Base"):
            continue
        err = fnum(row["normalized_bias_error_plotted"])
        out.append(
            {
                "figure_slug": row["figure_slug"],
                "strategy_key": row["strategy_key"],
                "strategy_label": row["strategy_label"],
                "ft_checkpoint": row["ft_checkpoint"],
                "dataset_key": row["dataset_key"],
                "panel_label": row["panel_label"],
                "method": row["method"],
                "method_family": method_family(row["method"]),
                "shot": "" if row["shot"] == "" else str(int(float(row["shot"]))),
                "bias_error": err,
                "bias_reduction_vs_base": base_error_for(config_rows, row["panel_label"]) - err,
                "source_model": row["model"],
                "metric_source_file": row["metric_source_file"],
            }
        )
    return out


def summarize_interventions(intervention_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in intervention_rows
        if r["dataset_key"] == "__mean__" and r["method"] != "Base Model"
    ]
    return [
        {
            "method": r["method"],
            "intervention_key": r["intervention_key"],
            "intervention_label": r["intervention_label"],
            "mean_bias_error": fnum(r["normalized_bias_error"]),
            "mean_reduction_vs_base": fnum(r["bias_error_reduction_vs_base"]),
        }
        for r in rows
    ]


def plot_strategy_method_mean(summary_rows: list[dict[str, object]]) -> None:
    methods = ["DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"]
    strategies = [
        "Stereotype replacement",
        "Individuation",
        "Perspective-taking",
        "Counter-stereotypic imaging",
        "Positive contact",
    ]
    lookup = {
        (str(r["method"]), str(r["intervention_label"])): float(r["mean_reduction_vs_base"])
        for r in summary_rows
    }

    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    x = np.arange(len(methods))
    width = 0.14
    offsets = np.linspace(-2 * width, 2 * width, len(strategies))

    for offset, strategy in zip(offsets, strategies):
        vals = [lookup.get((m, strategy), np.nan) for m in methods]
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=STRATEGY_COLORS[strategy],
            edgecolor="#252A2E",
            linewidth=0.9,
            label=strategy,
            zorder=3,
        )
        for bar, value in zip(bars, vals):
            if math.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.12,
                nice_value(value),
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="#1F2937",
            )

    ax.set_title("Mean bias-error reduction by intervention and method", fontweight="bold", pad=12)
    ax.set_ylabel("Reduction vs. base model", fontweight="semibold", color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels(["DIY IT", "DIY Two Pass\n(No IT)", "DIY Two Pass\n(IT)"], fontweight="bold")
    ax.axhline(0, color="#48515A", linewidth=0.85)
    ax.grid(axis="y", linestyle="--", color="#DCE2EA", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="#B9C2CF",
    )
    fig.text(
        0.01,
        0.015,
        "Bars average the six normalized bias metrics. Higher means larger absolute bias-error reduction from the base model.",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, "strategy_method_mean_reduction")


def plot_strategy_dataset_reductions(intervention_rows: list[dict[str, str]]) -> None:
    rows = [
        r
        for r in intervention_rows
        if r["dataset_key"] != "__mean__"
        and r["method"] != "Base Model"
        and r["intervention_label"] != "Base Model"
    ]
    strategies = [
        "Stereotype replacement",
        "Individuation",
        "Perspective-taking",
        "Counter-stereotypic imaging",
        "Positive contact",
    ]
    methods = ["DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"]
    markers = {"DIY IT": "o", "DIY Two Pass (No IT)": "s", "DIY Two Pass (IT)": "D"}
    method_colors = {
        "DIY IT": METHOD_COLORS["DIY IT"],
        "DIY Two Pass (No IT)": METHOD_COLORS["Two Pass (No IT)"],
        "DIY Two Pass (IT)": METHOD_COLORS["Two Pass (IT)"],
    }
    lookup = {
        (r["panel_label"], r["intervention_label"], r["method"]): fnum(
            r["bias_error_reduction_vs_base"]
        )
        for r in rows
    }

    fig, axes = plt.subplots(2, 3, figsize=(12.7, 7.2), sharex=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    y = np.arange(len(strategies))

    for ax, panel in zip(axes, PANEL_ORDER):
        ax.set_facecolor("#FCFCFD")
        ax.axvline(0, color="#48515A", linewidth=0.85, zorder=1)
        for method in methods:
            vals = [lookup.get((panel, s, method), np.nan) for s in strategies]
            ax.scatter(
                vals,
                y,
                s=62,
                marker=markers[method],
                color=method_colors[method],
                edgecolor="#252A2E",
                linewidth=0.9,
                label=method,
                zorder=3,
            )
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_yticks(y)
        ax.set_yticklabels(strategies if panel in {"CrowS-Pairs", "BBQ Disambig."} else [])
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", color="#DCE2EA", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
        ax.tick_params(colors="#374151")

    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[m],
            color="none",
            markerfacecolor=method_colors[m],
            markeredgecolor="#252A2E",
            markersize=8,
            label=m,
        )
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="#B9C2CF",
    )
    fig.suptitle("Per-dataset intervention gains", y=0.995, fontsize=16.2, fontweight="bold")
    fig.supxlabel("Bias-error reduction vs. base model", y=0.055, fontsize=11.5, fontweight="semibold")
    fig.text(
        0.012,
        0.018,
        "Each point is one intervention-method pair on one benchmark metric; points left of zero increase bias error.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), w_pad=1.2, h_pad=1.4)
    savefig(fig, "strategy_dataset_reduction_points")


def aggregate_mean(
    rows: list[dict[str, object]],
    key_fields: list[str],
    value_field: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    exemplars: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(float(row[value_field]))
        exemplars[key] = row
    out = []
    for key, values in groups.items():
        rec = {field: key[i] for i, field in enumerate(key_fields)}
        rec[f"mean_{value_field}"] = float(np.mean(values))
        rec[f"median_{value_field}"] = float(np.median(values))
        rec["n"] = len(values)
        out.append(rec)
    return out


def primary_strategy_panel_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        r
        for r in panel_rows
        if str(r["figure_slug"]) in PRIMARY_STRATEGY_SLUGS
        and str(r["method_family"]) in {"Two Pass (No IT)", "Two Pass (IT)"}
    ]


def plot_shot_trends_by_strategy(panel_rows: list[dict[str, object]]) -> None:
    rows = [
        r
        for r in primary_strategy_panel_rows(panel_rows)
        if str(r["shot"]) in {"0", "1", "2"}
    ]
    mean_rows = aggregate_mean(
        rows,
        ["strategy_label", "method_family", "shot"],
        "bias_error",
    )
    lookup = {
        (str(r["strategy_label"]), str(r["method_family"]), int(r["shot"])): float(
            r["mean_bias_error"]
        )
        for r in mean_rows
    }
    strategies = [
        "All interventions",
        "Stereotype replacement",
        "Individuation",
        "Perspective-taking",
        "Counter-stereotypic imaging",
        "Positive contact",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)
    fig.patch.set_facecolor("white")
    shots = np.array([0, 1, 2])
    for ax, family in zip(axes, ["Two Pass (No IT)", "Two Pass (IT)"]):
        ax.set_facecolor("#FCFCFD")
        for strategy in strategies:
            vals = [lookup.get((strategy, family, s), np.nan) for s in shots]
            ax.plot(
                shots,
                vals,
                marker="o",
                markersize=7.2,
                linewidth=2.15,
                color=STRATEGY_COLORS[strategy],
                markeredgecolor="#252A2E",
                markeredgewidth=0.8,
                label=strategy,
            )
        ax.set_title(family, fontweight="bold", pad=9)
        ax.set_xticks(shots)
        ax.set_xlabel("Number of demonstrations", fontweight="semibold")
        ax.grid(axis="y", linestyle="--", color="#DCE2EA", linewidth=0.85, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
    axes[0].set_ylabel("Mean bias error", fontweight="semibold", color="#374151")
    handles = [
        Line2D(
            [0],
            [0],
            color=STRATEGY_COLORS[strategy],
            marker="o",
            markeredgecolor="#252A2E",
            linewidth=2.15,
            markersize=7.0,
            label=strategy,
        )
        for strategy in strategies
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="#B9C2CF",
    )
    fig.suptitle("Shot sensitivity by intervention", y=1.02, fontsize=16.2, fontweight="bold")
    fig.text(
        0.012,
        0.02,
        "Each line averages the six bias metrics for a fixed intervention. Lower is better.",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86), w_pad=1.4)
    savefig(fig, "shot_trends_by_strategy_mean")


def plot_shot_trends_by_dataset(panel_rows: list[dict[str, object]]) -> None:
    rows = [
        r
        for r in panel_rows
        if str(r["figure_slug"]) == "all_strategies_all_versions"
        and str(r["method_family"]) in {"Two Pass (No IT)", "Two Pass (IT)"}
        and str(r["shot"]) in {"0", "1", "2"}
    ]
    lookup = {
        (str(r["panel_label"]), str(r["method_family"]), int(r["shot"])): float(r["bias_error"])
        for r in rows
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6), sharex=True)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    shots = np.array([0, 1, 2])
    families = ["Two Pass (No IT)", "Two Pass (IT)"]
    for ax, panel in zip(axes, PANEL_ORDER):
        ax.set_facecolor("#FCFCFD")
        for family in families:
            vals = [lookup.get((panel, family, s), np.nan) for s in shots]
            ax.plot(
                shots,
                vals,
                marker="o",
                markersize=7.0,
                linewidth=2.2,
                color=METHOD_COLORS[family],
                markeredgecolor="#252A2E",
                markeredgewidth=0.85,
                label=family,
            )
            for s, value in zip(shots, vals):
                if not math.isnan(value):
                    ax.text(s, value, f" {nice_value(value)}", fontsize=8.4, va="center", color="#1F2937")
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_xticks(shots)
        ax.grid(axis="y", linestyle="--", color="#DCE2EA", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[f],
            marker="o",
            markeredgecolor="#252A2E",
            linewidth=2.2,
            label=f,
        )
        for f in families
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=True,
        fancybox=True,
        edgecolor="#B9C2CF",
    )
    fig.suptitle("Dataset-level shot trends for all interventions", y=1.0, fontsize=16.2, fontweight="bold")
    fig.supxlabel("Number of demonstrations", y=0.06, fontsize=11.4, fontweight="semibold")
    fig.supylabel("Bias error", x=0.025, fontsize=11.4, fontweight="semibold")
    fig.text(
        0.012,
        0.02,
        "All-intervention prompt setting with the all-version IT checkpoint. Lower is better.",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.9), w_pad=1.2, h_pad=1.3)
    savefig(fig, "shot_trends_by_dataset_all_interventions")


def plot_checkpoint_version_trajectory(mean_rows: list[dict[str, object]]) -> None:
    rows = [
        r
        for r in mean_rows
        if str(r["strategy_label"]) == "All interventions"
        and str(r["method_family"]) in {"DIY IT", "Two Pass (IT)"}
    ]
    checkpoints = ["all-version", "opinion", "action", "event"]
    x_labels = ["DIY IT", "TP 0", "TP 1", "TP 2"]
    x = np.arange(len(x_labels))
    lookup = {}
    for r in rows:
        if str(r["method_family"]) == "DIY IT":
            xpos = 0
        else:
            xpos = int(str(r["shot"])) + 1
        lookup[(str(r["ft_checkpoint"]), xpos)] = float(r["mean_bias_error"])

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        vals = [lookup.get((checkpoint, i), np.nan) for i in range(len(x_labels))]
        ax.plot(
            x,
            vals,
            marker="o",
            markersize=7.8,
            linewidth=2.35,
            color=CHECKPOINT_COLORS[checkpoint],
            markeredgecolor="#252A2E",
            markeredgewidth=0.9,
            label=checkpoint,
        )
        for xpos, value in zip(x, vals):
            if not math.isnan(value):
                jitter = (checkpoint_idx - 1.5) * 0.035 if xpos == 0 else 0.08
                ax.text(
                    xpos,
                    value + jitter,
                    nice_value(value),
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    color="#1F2937",
                )
    ax.set_title("All-intervention checkpoint/version trajectory", fontweight="bold", pad=11)
    ax.set_ylabel("Mean bias error", fontweight="semibold", color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.grid(axis="y", linestyle="--", color="#DCE2EA", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.legend(
        title="IT checkpoint",
        loc="upper right",
        frameon=True,
        fancybox=True,
        edgecolor="#B9C2CF",
    )
    fig.text(
        0.012,
        0.02,
        "Mean is over the six normalized bias metrics. TP 0/1/2 are two-pass shots with instruction-tuned models.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "checkpoint_version_trajectory_all_interventions")


def plot_checkpoint_dataset_best(panel_rows: list[dict[str, object]]) -> None:
    rows = [
        r
        for r in panel_rows
        if str(r["strategy_label"]) == "All interventions"
        and str(r["method_family"]) in {"DIY IT", "Two Pass (IT)"}
    ]
    checkpoints = ["all-version", "opinion", "action", "event"]
    best: dict[tuple[str, str], dict[str, object]] = {}
    for r in rows:
        key = (str(r["panel_label"]), str(r["ft_checkpoint"]))
        if key not in best or float(r["bias_error"]) < float(best[key]["bias_error"]):
            best[key] = r

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    x = np.arange(len(checkpoints))
    for ax, panel in zip(axes, PANEL_ORDER):
        ax.set_facecolor("#FCFCFD")
        vals = [float(best[(panel, c)]["bias_error"]) for c in checkpoints]
        bars = ax.bar(
            x,
            vals,
            color=[CHECKPOINT_COLORS[c] for c in checkpoints],
            edgecolor="#252A2E",
            linewidth=0.95,
            zorder=3,
        )
        for bar, checkpoint, value in zip(bars, checkpoints, vals):
            chosen = best[(panel, checkpoint)]
            method = "IT" if chosen["method_family"] == "DIY IT" else f"TP {chosen['shot']}"
            ymax = max(vals) if max(vals) else 1.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.055 * ymax,
                f"{nice_value(value)}\n{method}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="#1F2937",
            )
        ymax = max(vals) if max(vals) else 1.0
        ax.set_ylim(0, ymax * 1.32)
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["all", "opinion", "action", "event"], rotation=0)
        ax.grid(axis="y", linestyle="--", color="#DCE2EA", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
    fig.suptitle("Best checkpoint version per dataset", y=0.995, fontsize=16.2, fontweight="bold")
    fig.supxlabel("All-intervention IT checkpoint", y=0.06, fontsize=11.3, fontweight="semibold")
    fig.supylabel("Best bias error", x=0.025, fontsize=11.3, fontweight="semibold")
    fig.text(
        0.012,
        0.02,
        "For each benchmark and checkpoint, the plotted bar is the best of DIY IT and two-pass IT shots 0/1/2. Lower is better.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.91), w_pad=1.2, h_pad=2.1)
    savefig(fig, "checkpoint_dataset_best_setting")


def plot_global_leaderboard(mean_rows: list[dict[str, object]]) -> None:
    rows = [r for r in mean_rows if str(r["figure_slug"]) in PRIMARY_STRATEGY_SLUGS]
    top = sorted(rows, key=lambda r: float(r["mean_bias_error"]))[:14]
    labels = [compact_config_label(r) for r in top]
    vals = [float(r["mean_bias_error"]) for r in top]
    colors = [METHOD_COLORS.get(str(r["method"]), METHOD_COLORS.get(str(r["method_family"]), "#A9B7D9")) for r in top]
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    bars = ax.barh(y, vals, color=colors, edgecolor="#252A2E", linewidth=0.9, zorder=3)
    for bar, value in zip(bars, vals):
        ax.text(value + 0.035, bar.get_y() + bar.get_height() / 2, nice_value(value), va="center", fontsize=9.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean bias error", fontweight="semibold", color="#374151")
    ax.set_title("Best overall DIY configurations", fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", color="#DCE2EA", linewidth=0.8, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.set_xlim(0, max(vals) * 1.14)
    fig.text(
        0.012,
        0.02,
        "Ranking uses the mean of the six normalized bias metrics. Lower is better. Color encodes method family.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    savefig(fig, "global_configuration_leaderboard")


def plot_dataset_leaderboards(panel_rows: list[dict[str, object]]) -> None:
    rows = [r for r in panel_rows if str(r["figure_slug"]) in PRIMARY_STRATEGY_SLUGS]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.3), sharex=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for ax, panel in zip(axes, PANEL_ORDER):
        top = sorted([r for r in rows if str(r["panel_label"]) == panel], key=lambda r: float(r["bias_error"]))[:6]
        labels = [compact_config_label(r, include_checkpoint=False) for r in top]
        vals = [float(r["bias_error"]) for r in top]
        y = np.arange(len(top))
        colors = [METHOD_COLORS.get(str(r["method"]), METHOD_COLORS.get(str(r["method_family"]), "#A9B7D9")) for r in top]
        ax.set_facecolor("#FCFCFD")
        bars = ax.barh(y, vals, color=colors, edgecolor="#252A2E", linewidth=0.85, zorder=3)
        for bar, value in zip(bars, vals):
            ax.text(value + 0.015 * max(vals), bar.get_y() + bar.get_height() / 2, nice_value(value), va="center", fontsize=8.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.7)
        ax.invert_yaxis()
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle="--", color="#DCE2EA", linewidth=0.75, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")

    fig.suptitle("Top configurations by dataset", y=0.995, fontsize=16.2, fontweight="bold")
    fig.supxlabel("Bias error", y=0.055, fontsize=11.3, fontweight="semibold")
    fig.text(
        0.012,
        0.019,
        "Each panel lists the six lowest-error DIY configurations among primary strategy settings. Lower is better.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=1.4, h_pad=1.3)
    savefig(fig, "dataset_configuration_leaderboards")


def plot_configuration_distribution(panel_rows: list[dict[str, object]]) -> None:
    rows = [r for r in panel_rows if str(r["figure_slug"]) in PRIMARY_STRATEGY_SLUGS]
    families = METHOD_FAMILIES
    fig, axes = plt.subplots(2, 3, figsize=(12.7, 6.9), sharex=True)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    rng = np.random.default_rng(7)

    for ax, panel in zip(axes, PANEL_ORDER):
        ax.set_facecolor("#FCFCFD")
        data = [
            [float(r["bias_error"]) for r in rows if str(r["panel_label"]) == panel and str(r["method_family"]) == family]
            for family in families
        ]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.2},
            boxprops={"edgecolor": "#252A2E", "linewidth": 0.9},
            whiskerprops={"color": "#252A2E", "linewidth": 0.8},
            capprops={"color": "#252A2E", "linewidth": 0.8},
        )
        for patch, family in zip(bp["boxes"], families):
            patch.set_facecolor(METHOD_COLORS[family])
            patch.set_alpha(0.72)
        for idx, values in enumerate(data, start=1):
            all_values = [v for group in data for v in group]
            base = base_error_for([dict(r) for r in read_csv(SOURCE_CSV / "debiasing_method_bars_by_shot_configs_data.csv")], panel)
            cap = max(base * 1.22, float(np.percentile(all_values, 88)) * 1.18)
            cap = min(max(all_values) * 1.05, cap)
            visible = [min(v, cap * 0.985) for v in values]
            jitter = rng.normal(0, 0.04, len(values))
            ax.scatter(
                np.full(len(values), idx) + jitter,
                visible,
                s=22,
                color="#FFFFFF",
                edgecolor="#252A2E",
                linewidth=0.55,
                alpha=0.9,
                zorder=3,
            )
            hidden = [v for v in values if v > cap]
            if hidden:
                ax.scatter(
                    [idx],
                    [cap * 0.985],
                    s=52,
                    marker="^",
                    color="#FFFFFF",
                    edgecolor="#252A2E",
                    linewidth=0.7,
                    zorder=4,
                )
                ax.text(
                    idx + 0.08,
                    cap * 0.965,
                    f"{len(hidden)} high",
                    fontsize=7.5,
                    va="top",
                    color="#374151",
                )
        ax.axhline(base, color="#7A8798", linestyle=":", linewidth=1.2)
        ax.set_ylim(bottom=0, top=cap * 1.04)
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(1, len(families) + 1))
        ax.set_xticklabels(["DIY IT", "No IT", "IT"], fontweight="bold")
        ax.grid(axis="y", linestyle="--", color="#DCE2EA", linewidth=0.75, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")

    fig.suptitle("Configuration spread by method family", y=0.995, fontsize=16.2, fontweight="bold")
    fig.supylabel("Bias error", x=0.025, fontsize=11.3, fontweight="semibold")
    fig.text(
        0.012,
        0.02,
        "Boxes summarize all primary strategy/checkpoint/shot configurations in each method family. Dotted line is the base model.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.94), w_pad=1.2, h_pad=1.25)
    savefig(fig, "configuration_distribution_by_dataset")


def baseline_rank_rows(baseline_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for panel in PANEL_ORDER:
        rows = [r for r in baseline_rows if r["panel_label"] == panel]
        rows = sorted(rows, key=lambda r: fnum(r["normalized_bias_error_plotted"]))
        for rank, row in enumerate(rows, start=1):
            out.append(
                {
                    "panel_label": panel,
                    "method_label": row["method_label"],
                    "group": row["group"],
                    "rank": rank,
                    "bias_error": fnum(row["normalized_bias_error_plotted"]),
                }
            )
    return out


def plot_baseline_rank_profiles(rank_rows: list[dict[str, object]]) -> None:
    panels = PANEL_ORDER
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for ax, panel in zip(axes, panels):
        rows = [r for r in rank_rows if str(r["panel_label"]) == panel]
        top = rows[:10]
        y = np.arange(len(top))
        colors = [
            "#FDBA74" if str(r["method_label"]).startswith("DIY") else "#BFD7FF"
            for r in top
        ]
        ax.set_facecolor("#FCFCFD")
        bars = ax.barh(
            y,
            [float(r["bias_error"]) for r in top],
            color=colors,
            edgecolor="#252A2E",
            linewidth=0.85,
            zorder=3,
        )
        for bar, row in zip(bars, top):
            ax.text(
                float(row["bias_error"]) + 0.02 * max(float(r["bias_error"]) for r in top),
                bar.get_y() + bar.get_height() / 2,
                f"#{int(row['rank'])}",
                va="center",
                fontsize=8.2,
                color="#1F2937",
            )
        ax.set_yticks(y)
        ax.set_yticklabels([str(r["method_label"]) for r in top], fontsize=7.8)
        ax.invert_yaxis()
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle="--", color="#DCE2EA", linewidth=0.75, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
    fig.suptitle("Baseline and DIY ranks by dataset", y=0.995, fontsize=16.2, fontweight="bold")
    fig.supxlabel("Bias error", y=0.055, fontsize=11.3, fontweight="semibold")
    fig.text(
        0.012,
        0.02,
        "Top ten methods per benchmark using the same normalized bias-error metric. Lower bars and lower ranks are better.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=1.3, h_pad=1.25)
    savefig(fig, "baseline_rank_profiles_by_dataset")


def parse_reasoning_config(row: dict[str, str]) -> dict[str, object] | None:
    dataset = row["dataset_key"]
    if dataset not in {key for key, _ in REASONING_ORDER}:
        return None
    if row["score_label"] != "accuracy" or row["type"] != "method":
        return None
    model = row["model"]
    if row["name"] == "reasoning_check":
        if model == f"reasoning_check__base__{dataset}":
            return {
                "figure_slug": "base",
                "strategy_label": "Base",
                "ft_checkpoint": "",
                "method": "Base Model Inference",
                "method_family": "Base",
                "shot": "",
                "dataset_key": dataset,
                "accuracy": float(row["score"]) * 100.0,
            }
        match = re.match(rf"reasoning_check__finetuned_ms-500-(.+)__{dataset}$", model)
        if not match:
            return None
        ft_slug = match.group(1)
        strategy, checkpoint, fig_slug = ft_slug_to_config(ft_slug)
        return {
            "figure_slug": fig_slug,
            "strategy_label": strategy,
            "ft_checkpoint": checkpoint,
            "method": "DIY IT",
            "method_family": "DIY IT",
            "shot": "",
            "dataset_key": dataset,
            "accuracy": float(row["score"]) * 100.0,
        }
    if row["name"] != "m6_self_debiasing":
        return None
    match = re.match(r"m6_two_pass(_one|_two)?__(base|finetuned_ms-500-.+)__(.+)__(arc_challenge|arc_easy|balanced_copa)$", model)
    if not match:
        return None
    shot_suffix, model_part, strategy_key, bench = match.groups()
    if bench != dataset:
        return None
    shot = {"": 0, "_one": 1, "_two": 2}[shot_suffix or ""]
    if model_part == "base":
        strategy = strategy_key_to_label(strategy_key)
        fig_slug = strategy_key_to_primary_slug(strategy_key)
        checkpoint = ""
        method = f"DIY Two Pass (No IT), {shot}-shot"
        family = "Two Pass (No IT)"
    else:
        ft_slug = model_part.replace("finetuned_ms-500-", "")
        strategy, checkpoint, fig_slug = ft_slug_to_config(ft_slug)
        method = f"DIY Two Pass (IT), {shot}-shot"
        family = "Two Pass (IT)"
    return {
        "figure_slug": fig_slug,
        "strategy_label": strategy,
        "ft_checkpoint": checkpoint,
        "method": method,
        "method_family": family,
        "shot": str(shot),
        "dataset_key": dataset,
        "accuracy": float(row["score"]) * 100.0,
    }


def ft_slug_to_config(ft_slug: str) -> tuple[str, str, str]:
    mapping = {
        "allstrategies-opinion-action-event-allversions": (
            "All interventions",
            "all-version",
            "all_strategies_all_versions",
        ),
        "allstrategies-opinion": ("All interventions", "opinion", "all_strategies_opinion"),
        "allstrategies-action": ("All interventions", "action", "all_strategies_action"),
        "allstrategies-event": ("All interventions", "event", "all_strategies_event"),
        "stereotype-replacement-opinion-action-event-allversions": (
            "Stereotype replacement",
            "matched all-version",
            "stereotype_replacement_all_versions",
        ),
        "individuating-opinion-action-event-allversions": (
            "Individuation",
            "matched all-version",
            "individuation_all_versions",
        ),
        "perspective-taking-opinion-action-event-allversions": (
            "Perspective-taking",
            "matched all-version",
            "perspective_taking_all_versions",
        ),
        "counter-imaging-opinion-action-event-allversions": (
            "Counter-stereotypic imaging",
            "matched all-version",
            "counter_stereotypic_imaging_all_versions",
        ),
        "positive-contact-opinion-action-event-allversions": (
            "Positive contact",
            "matched all-version",
            "positive_contact_all_versions",
        ),
    }
    return mapping.get(ft_slug, (ft_slug, "", ft_slug.replace("-", "_")))


def strategy_key_to_label(strategy_key: str) -> str:
    return {
        "all_strategies": "All interventions",
        "stereotype_replacement": "Stereotype replacement",
        "individuating": "Individuation",
        "perspective_taking": "Perspective-taking",
        "counter_imaging": "Counter-stereotypic imaging",
        "positive_contact": "Positive contact",
    }.get(strategy_key, strategy_key)


def strategy_key_to_primary_slug(strategy_key: str) -> str:
    return {
        "all_strategies": "all_strategies_all_versions",
        "stereotype_replacement": "stereotype_replacement_all_versions",
        "individuating": "individuation_all_versions",
        "perspective_taking": "perspective_taking_all_versions",
        "counter_imaging": "counter_stereotypic_imaging_all_versions",
        "positive_contact": "positive_contact_all_versions",
    }.get(strategy_key, strategy_key)


def reasoning_means(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parsed = [p for r in raw_rows if (p := parse_reasoning_config(r)) is not None]
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in parsed:
        key = (
            str(row["figure_slug"]),
            str(row["method"]),
            str(row["shot"]),
            str(row["ft_checkpoint"]),
        )
        groups[key].append(row)
    out = []
    for (slug, method, shot, checkpoint), rows in groups.items():
        if len({r["dataset_key"] for r in rows}) < 3:
            continue
        first = rows[0]
        out.append(
            {
                "figure_slug": slug,
                "strategy_label": first["strategy_label"],
                "ft_checkpoint": checkpoint,
                "method": method,
                "method_family": first["method_family"],
                "shot": shot,
                "mean_reasoning_accuracy": float(np.mean([float(r["accuracy"]) for r in rows])),
                "n_reasoning_benchmarks": len(rows),
            }
        )
    return out


def bias_reasoning_tradeoff_rows(
    mean_rows: list[dict[str, object]], reasoning_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    bias_lookup = {}
    for r in mean_rows:
        checkpoint = "" if str(r["method_family"]) == "Two Pass (No IT)" else str(r["ft_checkpoint"])
        key = (
            str(r["figure_slug"]),
            str(r["method"]),
            str(r["shot"]),
            checkpoint,
        )
        bias_lookup[key] = r
    out = []
    for r in reasoning_rows:
        checkpoint = "" if str(r["method_family"]) == "Two Pass (No IT)" else str(r["ft_checkpoint"])
        key = (
            str(r["figure_slug"]),
            str(r["method"]),
            str(r["shot"]),
            checkpoint,
        )
        if key not in bias_lookup:
            continue
        b = bias_lookup[key]
        out.append(
            {
                "figure_slug": r["figure_slug"],
                "strategy_label": r["strategy_label"],
                "ft_checkpoint": r["ft_checkpoint"],
                "method": r["method"],
                "method_family": r["method_family"],
                "shot": r["shot"],
                "mean_bias_error": b["mean_bias_error"],
                "mean_reduction_vs_base": b["mean_reduction_vs_base"],
                "mean_reasoning_accuracy": r["mean_reasoning_accuracy"],
                "n_bias_panels": b["n_panels"],
                "n_reasoning_benchmarks": r["n_reasoning_benchmarks"],
            }
        )
    return out


def plot_reasoning_shot_trends(reasoning_rows: list[dict[str, object]]) -> None:
    rows = [
        r
        for r in reasoning_rows
        if str(r["figure_slug"]) in PRIMARY_STRATEGY_SLUGS
        and str(r["method_family"]) in {"Two Pass (No IT)", "Two Pass (IT)"}
    ]
    lookup = {
        (str(r["strategy_label"]), str(r["method_family"]), int(str(r["shot"]))): float(
            r["mean_reasoning_accuracy"]
        )
        for r in rows
    }
    strategies = [
        "All interventions",
        "Stereotype replacement",
        "Individuation",
        "Perspective-taking",
        "Counter-stereotypic imaging",
        "Positive contact",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)
    fig.patch.set_facecolor("white")
    shots = np.array([0, 1, 2])
    for ax, family in zip(axes, ["Two Pass (No IT)", "Two Pass (IT)"]):
        ax.set_facecolor("#FCFCFD")
        for strategy in strategies:
            vals = [lookup.get((strategy, family, s), np.nan) for s in shots]
            ax.plot(
                shots,
                vals,
                marker="o",
                markersize=7.2,
                linewidth=2.15,
                color=STRATEGY_COLORS[strategy],
                markeredgecolor="#252A2E",
                markeredgewidth=0.8,
                label=strategy,
            )
        ax.set_title(family, fontweight="bold", pad=9)
        ax.set_xticks(shots)
        ax.set_xlabel("Number of demonstrations", fontweight="semibold")
        ax.grid(axis="y", linestyle="--", color="#DCE2EA", linewidth=0.85, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
    axes[0].set_ylabel("Mean reasoning accuracy (%)", fontweight="semibold", color="#374151")
    handles = [
        Line2D(
            [0],
            [0],
            color=STRATEGY_COLORS[strategy],
            marker="o",
            markeredgecolor="#252A2E",
            linewidth=2.15,
            markersize=7.0,
            label=strategy,
        )
        for strategy in strategies
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="#B9C2CF",
    )
    fig.suptitle("Reasoning accuracy by shot and intervention", y=1.02, fontsize=16.2, fontweight="bold")
    fig.text(
        0.012,
        0.02,
        "Mean accuracy across ARC-Challenge, ARC-Easy, and Balanced COPA. Higher is better.",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86), w_pad=1.4)
    savefig(fig, "reasoning_shot_trends_by_strategy")


def plot_bias_reasoning_tradeoff(rows: list[dict[str, object]]) -> None:
    plotted = [
        r
        for r in rows
        if str(r["figure_slug"]) in PRIMARY_STRATEGY_SLUGS
        or str(r["strategy_label"]) == "All interventions"
    ]
    fig, ax = plt.subplots(figsize=(9.6, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    markers = {"DIY IT": "o", "Two Pass (No IT)": "s", "Two Pass (IT)": "D"}
    for family in METHOD_FAMILIES:
        subset = [r for r in plotted if str(r["method_family"]) == family]
        ax.scatter(
            [float(r["mean_bias_error"]) for r in subset],
            [float(r["mean_reasoning_accuracy"]) for r in subset],
            s=86 if family == "DIY IT" else 74,
            marker=markers[family],
            color=METHOD_COLORS[family],
            edgecolor="#252A2E",
            linewidth=0.9,
            alpha=0.88,
            label=family,
            zorder=3,
        )

    highlight_candidates = [
        min(plotted, key=lambda r: float(r["mean_bias_error"])),
        max(plotted, key=lambda r: float(r["mean_reasoning_accuracy"])),
    ]
    highlight_candidates.extend(
        sorted(
            [r for r in plotted if float(r["mean_bias_error"]) < 2.2],
            key=lambda r: -float(r["mean_reasoning_accuracy"]),
        )[:4]
    )
    seen = set()
    highlight = []
    for r in highlight_candidates:
        key = (r["figure_slug"], r["method"], r["shot"], r["ft_checkpoint"])
        if key not in seen:
            seen.add(key)
            highlight.append(r)
    offsets = [(6, 5), (6, -12), (8, 8), (8, -14), (10, 2), (10, -18)]
    for idx, r in enumerate(highlight):
        label = f"{str(r['strategy_label']).split()[0]} {compact_method(str(r['method']))}"
        ax.annotate(
            label,
            (float(r["mean_bias_error"]), float(r["mean_reasoning_accuracy"])),
            xytext=offsets[idx % len(offsets)],
            textcoords="offset points",
            fontsize=8.1,
            color="#1F2937",
        )
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(float(r["mean_bias_error"]) for r in plotted) * 1.2)
    xticks = [0.8, 1, 2, 3, 5, 10, 20]
    ax.set_xticks([t for t in xticks if t <= ax.get_xlim()[1]])
    ax.set_xticklabels([str(t) for t in ax.get_xticks()])
    ax.set_xlabel("Mean bias error", fontweight="semibold", color="#374151")
    ax.set_ylabel("Mean reasoning accuracy (%)", fontweight="semibold", color="#374151")
    ax.set_title("Fine-grained bias-utility tradeoff", fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", color="#DCE2EA", linewidth=0.8, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.legend(loc="lower left", frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.text(
        0.012,
        0.02,
        "Each point is a strategy/checkpoint/method/shot configuration with both bias and reasoning results. Lower-left is lower bias; higher is better utility.",
        fontsize=9.1,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    savefig(fig, "bias_reasoning_tradeoff_finegrained")


def main() -> None:
    ensure_dirs()
    set_style()
    sources = load_sources()

    mean_rows = config_means(sources["configs"])
    panel_rows = config_panel_records(sources["configs"])
    intervention_summary = summarize_interventions(sources["interventions"])
    rank_rows = baseline_rank_rows(sources["baselines"])
    reasoning = reasoning_means(sources["raw"])
    tradeoff = bias_reasoning_tradeoff_rows(mean_rows, reasoning)

    write_csv(CSV_OUT / "finegrained_config_means.csv", mean_rows)
    write_csv(CSV_OUT / "finegrained_config_panel_rows.csv", panel_rows)
    write_csv(CSV_OUT / "finegrained_intervention_mean_reductions.csv", intervention_summary)
    write_csv(CSV_OUT / "finegrained_baseline_ranks.csv", rank_rows)
    write_csv(CSV_OUT / "finegrained_reasoning_means.csv", reasoning)
    write_csv(CSV_OUT / "finegrained_bias_reasoning_tradeoff.csv", tradeoff)

    plot_strategy_method_mean(intervention_summary)
    plot_strategy_dataset_reductions(sources["interventions"])
    plot_shot_trends_by_strategy(panel_rows)
    plot_shot_trends_by_dataset(panel_rows)
    plot_checkpoint_version_trajectory(mean_rows)
    plot_checkpoint_dataset_best(panel_rows)
    plot_global_leaderboard(mean_rows)
    plot_dataset_leaderboards(panel_rows)
    plot_configuration_distribution(panel_rows)
    plot_baseline_rank_profiles(rank_rows)
    plot_reasoning_shot_trends(reasoning)
    plot_bias_reasoning_tradeoff(tradeoff)


if __name__ == "__main__":
    main()
