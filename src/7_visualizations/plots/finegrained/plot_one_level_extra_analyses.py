#!/usr/bin/env python3
"""One-level-extra versions of the existing DIY result figures.

These are deliberately not new analysis types. Each figure mirrors one of the
main figures under figures/llama8b/pdf and adds exactly one finer-grained axis:
shot, intervention, checkpoint/version, or dataset.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
PLOT_DIR = HERE.parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PLOT_DIR))
sys.path.insert(0, str(HERE))

from plot_style import use_nimbus_sans  # noqa: E402
import plot_debiasing_method_bars as core  # noqa: E402
import plot_finegrained_analyses as fg  # noqa: E402


OUT = ROOT / "figures/llama8b/finegrained/one_axis"
CSV_OUT = OUT / "csv"
PDF_OUT = OUT / "pdf"
SOURCE_CSV = ROOT / "figures/llama8b/csv"

PANELS = fg.PANEL_ORDER
REASONING = fg.REASONING_ORDER
STRATEGIES = [
    "All interventions",
    "Stereotype replacement",
    "Individuation",
    "Perspective-taking",
    "Counter-stereotypic imaging",
    "Positive contact",
]
INTERVENTIONS = STRATEGIES[1:]
CHECKPOINTS = ["all-version", "opinion", "action", "event"]

METHOD_COLORS = fg.METHOD_COLORS
STRATEGY_COLORS = fg.STRATEGY_COLORS
CHECKPOINT_COLORS = fg.CHECKPOINT_COLORS


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
            "axes.labelsize": 11.3,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.2,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (CSV_OUT / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(PDF_OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fnum(value: str | float | int | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def fmt(value: float) -> str:
    if abs(value) < 1:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def clean_method(method: str) -> str:
    return (
        method.replace("DIY Two Pass (No IT), ", "No IT ")
        .replace("DIY Two Pass (IT), ", "IT ")
        .replace("-shot", "")
        .replace("Base Model Inference", "Base")
    )


def method_family(method: str) -> str:
    return fg.method_family(method)


def base_error(config_rows: list[dict[str, str]], panel: str) -> float:
    return fg.base_error_for(config_rows, panel)


def panel_grid(title: str, ylabel: str = "Bias error") -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.05), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    fig.suptitle(title, y=0.995, fontsize=16.5, fontweight="bold", color="#111827")
    fig.supylabel(ylabel, x=0.026, fontsize=11.5, fontweight="semibold", color="#374151")
    return fig, axes


def style_axis(ax) -> None:
    ax.set_facecolor("#FCFCFD")
    ax.grid(axis="y", color="#DCE2EA", linewidth=0.82, linestyle="--", alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.tick_params(colors="#374151")


def config_panel_rows(config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return fg.config_panel_records(config_rows)


def config_means(config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return fg.config_means(config_rows)


def reasoning_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parsed = []
    for row in raw_rows:
        item = fg.parse_reasoning_config(row)
        if item is not None:
            parsed.append(item)
    return parsed


def mean_reasoning(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return fg.reasoning_means(raw_rows)


def plot_debiasing_by_shot(config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in config_rows
        if r["figure_slug"] == "all_strategies_all_versions"
        and r["method"]
        in {
            "Base Model Inference",
            "DIY IT",
            "DIY Two Pass (No IT), 0-shot",
            "DIY Two Pass (No IT), 1-shot",
            "DIY Two Pass (No IT), 2-shot",
            "DIY Two Pass (IT), 0-shot",
            "DIY Two Pass (IT), 1-shot",
            "DIY Two Pass (IT), 2-shot",
        }
    ]
    methods = [
        "Base Model Inference",
        "DIY IT",
        "DIY Two Pass (No IT), 0-shot",
        "DIY Two Pass (No IT), 1-shot",
        "DIY Two Pass (No IT), 2-shot",
        "DIY Two Pass (IT), 0-shot",
        "DIY Two Pass (IT), 1-shot",
        "DIY Two Pass (IT), 2-shot",
    ]
    colors = [
        "#BBC2CE",
        METHOD_COLORS["DIY IT"],
        "#B6EAC7",
        "#84D9A5",
        "#50C77B",
        "#FFD8A7",
        "#FFB86B",
        "#FF9E45",
    ]
    hatches = ["", "///", "", "\\\\\\", "...", "", "///", "..."]
    lookup = {(r["panel_label"], r["method"]): fnum(r["normalized_bias_error_plotted"]) for r in rows}

    fig, axes = panel_grid("Debiasing performance + shot")
    x = np.arange(len(methods))
    for ax, panel in zip(axes, PANELS):
        style_axis(ax)
        ax.axvspan(1.55, 4.45, color="#ECF9F0", alpha=0.62, zorder=0)
        ax.axvspan(4.55, 7.45, color="#FFF3E4", alpha=0.72, zorder=0)
        ax.axvline(1.5, color="#9AA4B2", linewidth=0.85)
        ax.axvline(4.5, color="#9AA4B2", linewidth=0.85)
        vals = [lookup.get((panel, method), np.nan) for method in methods]
        bars = ax.bar(x, vals, width=0.67, color=colors, edgecolor="#252A2E", linewidth=0.92, zorder=3)
        for bar, hatch, value in zip(bars, hatches, vals):
            bar.set_hatch(hatch)
            if not math.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025 * max(vals), fmt(value), ha="center", va="bottom", fontsize=8.2)
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([clean_method(m).replace(" ", "\n") for m in methods])
        ax.set_ylim(0, max(v for v in vals if not math.isnan(v)) * 1.25)
    fig.text(0.012, 0.02, "Same as debiasing-performance bars, with one extra level: two-pass shot count. Lower is better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0.02, 0.055, 1, 0.94), w_pad=1.1, h_pad=1.2)
    savefig(fig, "debiasing_performance_by_shot")
    return rows


def plot_debiasing_by_intervention(intervention_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in intervention_rows
        if r["dataset_key"] != "__mean__" and r["method"] != "Base Model"
    ]
    methods = ["DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"]
    lookup = {(r["panel_label"], r["method"], r["intervention_label"]): fnum(r["normalized_bias_error"]) for r in rows}
    fig, axes = panel_grid("Debiasing performance + intervention")
    x = np.arange(len(methods))
    width = 0.13
    offsets = np.linspace(-2 * width, 2 * width, len(INTERVENTIONS))
    for ax, panel in zip(axes, PANELS):
        style_axis(ax)
        maxv = 0.0
        for offset, strategy in zip(offsets, INTERVENTIONS):
            vals = [lookup.get((panel, method, strategy), np.nan) for method in methods]
            maxv = max(maxv, max(v for v in vals if not math.isnan(v)))
            ax.bar(x + offset, vals, width=width, color=STRATEGY_COLORS[strategy], edgecolor="#252A2E", linewidth=0.72, label=strategy, zorder=3)
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["DIY IT", "Two Pass\n(No IT)", "Two Pass\n(IT)"], fontweight="bold")
        ax.set_ylim(0, maxv * 1.2 if maxv else 1)
    handles = [Patch(facecolor=STRATEGY_COLORS[s], edgecolor="#252A2E", label=s) for s in INTERVENTIONS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.text(0.012, 0.02, "Same debiasing metric, with one extra level: intervention identity. Lower is better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0.02, 0.055, 1, 0.88), w_pad=1.1, h_pad=1.2)
    savefig(fig, "debiasing_performance_by_intervention")
    return rows


def plot_debiasing_by_checkpoint(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in panel_rows
        if r["strategy_label"] == "All interventions"
        and r["ft_checkpoint"] in CHECKPOINTS
        and (r["method_family"] == "DIY IT" or (r["method_family"] == "Two Pass (IT)" and r["shot"] == "0"))
    ]
    lookup = {(r["panel_label"], r["ft_checkpoint"], r["method_family"]): fnum(r["bias_error"]) for r in rows}
    families = ["DIY IT", "Two Pass (IT)"]
    fig, axes = panel_grid("Debiasing performance + IT checkpoint")
    x = np.arange(len(CHECKPOINTS))
    width = 0.34
    offsets = [-width / 2, width / 2]
    for ax, panel in zip(axes, PANELS):
        style_axis(ax)
        maxv = 0.0
        for family, offset in zip(families, offsets):
            vals = [lookup.get((panel, cp, family), np.nan) for cp in CHECKPOINTS]
            maxv = max(maxv, max(v for v in vals if not math.isnan(v)))
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=METHOD_COLORS[family],
                edgecolor="#252A2E",
                linewidth=0.82,
                label=family,
                zorder=3,
            )
            if family == "DIY IT":
                for bar in bars:
                    bar.set_hatch("///")
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["all", "opinion", "action", "event"])
        ax.set_ylim(0, maxv * 1.22 if maxv else 1)
    handles = [
        Patch(facecolor=METHOD_COLORS["DIY IT"], edgecolor="#252A2E", hatch="///", label="DIY IT"),
        Patch(facecolor=METHOD_COLORS["Two Pass (IT)"], edgecolor="#252A2E", label="DIY Two Pass (IT), 0-shot"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.text(0.012, 0.02, "Same debiasing metric, with one extra level: instruction-tuned checkpoint/version. Lower is better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0.02, 0.055, 1, 0.9), w_pad=1.1, h_pad=1.2)
    savefig(fig, "debiasing_performance_by_checkpoint")
    return rows


def plot_baseline_by_dataset(baseline_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    labels = {}
    groups = {}
    for r in baseline_rows:
        grouped[r["method_key"]].append(r)
        labels[r["method_key"]] = r["method_label"]
        groups[r["method_key"]] = r["group"]
    summary = []
    for key, rs in grouped.items():
        summary.append(
            {
                "method_key": key,
                "method_label": labels[key],
                "group": groups[key],
                "mean_bias_error": float(np.mean([fnum(r["normalized_bias_error_plotted"]) for r in rs])),
                "n_datasets": len(rs),
            }
        )
    summary = sorted(summary, key=lambda r: r["mean_bias_error"])
    methods = [r["method_key"] for r in summary]
    y = np.arange(len(methods)) * 1.12
    panel_colors = {
        "CrowS-Pairs": "#7DD3FC",
        "StereoSet": "#5EEAD4",
        "BBQ Ambig.": "#A7F3D0",
        "BBQ Disambig.": "#FDBA74",
        "WinoBias": "#C4B5FD",
        "WinoGender": "#FCA5A5",
    }
    fig, ax = plt.subplots(figsize=(11.2, 8.6))
    fig.patch.set_facecolor("white")
    style_axis(ax)
    row_lookup = {(r["method_key"], r["panel_label"]): fnum(r["normalized_bias_error_plotted"]) for r in baseline_rows}
    for yi, method_key in zip(y, methods):
        vals = [row_lookup.get((method_key, p), np.nan) for p in PANELS]
        finite = [v for v in vals if not math.isnan(v)]
        if finite:
            ax.hlines(yi, min(finite), max(finite), color="#A8B3C5", linewidth=1.4, zorder=1)
        for panel, val in zip(PANELS, vals):
            if not math.isnan(val):
                ax.scatter(val, yi, s=54, color=panel_colors[panel], edgecolor="#252A2E", linewidth=0.75, zorder=3)
        mean = next(r["mean_bias_error"] for r in summary if r["method_key"] == method_key)
        marker = "D" if method_key.startswith("diy_") else "o"
        ax.scatter(mean, yi, s=94, color="#FFFFFF", edgecolor="#111827", linewidth=1.25, marker=marker, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Bias error", fontweight="semibold", color="#374151")
    ax.set_title("Baseline comparison + dataset", fontsize=16.2, fontweight="bold", pad=12)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="#252A2E", label=p, markersize=7) for p, c in panel_colors.items()]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#FFFFFF", markeredgecolor="#111827", label="method mean", markersize=8))
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.text(0.012, 0.02, "Same baseline comparison, with one extra level: benchmark dataset. White marker is the mean across available bias metrics.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    savefig(fig, "baseline_comparison_by_dataset")
    return summary


def plot_reasoning_by_shot(config_rows: list[dict[str, str]], raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in reasoning_rows(raw_rows)
        if r["figure_slug"] in {"base", "all_strategies_all_versions"}
        and r["method"]
        in {
            "Base Model Inference",
            "DIY IT",
            "DIY Two Pass (No IT), 0-shot",
            "DIY Two Pass (No IT), 1-shot",
            "DIY Two Pass (No IT), 2-shot",
            "DIY Two Pass (IT), 0-shot",
            "DIY Two Pass (IT), 1-shot",
            "DIY Two Pass (IT), 2-shot",
        }
    ]
    methods = [
        "Base Model Inference",
        "DIY IT",
        "DIY Two Pass (No IT), 0-shot",
        "DIY Two Pass (No IT), 1-shot",
        "DIY Two Pass (No IT), 2-shot",
        "DIY Two Pass (IT), 0-shot",
        "DIY Two Pass (IT), 1-shot",
        "DIY Two Pass (IT), 2-shot",
    ]
    lookup = {(r["dataset_key"], r["method"]): fnum(r["accuracy"]) for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9), sharey=True)
    fig.patch.set_facecolor("white")
    x = np.arange(len(methods))
    colors = ["#BBC2CE", METHOD_COLORS["DIY IT"], "#B6EAC7", "#84D9A5", "#50C77B", "#FFD8A7", "#FFB86B", "#FF9E45"]
    for ax, (dataset, label) in zip(axes, REASONING):
        style_axis(ax)
        vals = [lookup.get((dataset, m), np.nan) for m in methods]
        ax.bar(x, vals, width=0.67, color=colors, edgecolor="#252A2E", linewidth=0.88, zorder=3)
        for xi, value in zip(x, vals):
            if not math.isnan(value):
                ax.text(xi, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=8.2)
        ax.set_title(label, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([clean_method(m).replace(" ", "\n") for m in methods])
        ax.set_ylim(60, 88)
    axes[0].set_ylabel("Accuracy (%)", fontweight="semibold", color="#374151")
    fig.suptitle("Reasoning performance + shot", y=0.995, fontsize=16.5, fontweight="bold")
    fig.text(0.012, 0.02, "Same reasoning-performance analysis, with one extra level: two-pass shot count. Higher is better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0, 0.075, 1, 0.92), w_pad=1.1)
    savefig(fig, "reasoning_performance_by_shot")
    return rows


def plot_reasoning_by_intervention(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parsed = reasoning_rows(raw_rows)
    rows = [
        r
        for r in parsed
        if r["strategy_label"] in INTERVENTIONS
        and (r["method"] == "DIY IT" or r["method"] in {"DIY Two Pass (No IT), 0-shot", "DIY Two Pass (IT), 0-shot"})
    ]
    methods = ["DIY IT", "DIY Two Pass (No IT), 0-shot", "DIY Two Pass (IT), 0-shot"]
    lookup = {(r["dataset_key"], r["method"], r["strategy_label"]): fnum(r["accuracy"]) for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharey=True)
    fig.patch.set_facecolor("white")
    x = np.arange(len(methods))
    width = 0.13
    offsets = np.linspace(-2 * width, 2 * width, len(INTERVENTIONS))
    for ax, (dataset, label) in zip(axes, REASONING):
        style_axis(ax)
        for offset, strategy in zip(offsets, INTERVENTIONS):
            vals = [lookup.get((dataset, method, strategy), np.nan) for method in methods]
            ax.bar(x + offset, vals, width=width, color=STRATEGY_COLORS[strategy], edgecolor="#252A2E", linewidth=0.72, zorder=3)
        ax.set_title(label, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["DIY IT", "No IT\n0", "IT\n0"], fontweight="bold")
        ax.set_ylim(52, 88)
    axes[0].set_ylabel("Accuracy (%)", fontweight="semibold", color="#374151")
    handles = [Patch(facecolor=STRATEGY_COLORS[s], edgecolor="#252A2E", label=s) for s in INTERVENTIONS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.suptitle("Reasoning performance + intervention", y=0.995, fontsize=16.5, fontweight="bold")
    fig.text(0.012, 0.02, "Same reasoning-performance analysis, with one extra level: intervention identity. Higher is better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0, 0.075, 1, 0.85), w_pad=1.1)
    savefig(fig, "reasoning_performance_by_intervention")
    return rows


def tradeoff_records(mean_rows: list[dict[str, object]], reasoning_mean_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return fg.bias_reasoning_tradeoff_rows(mean_rows, reasoning_mean_rows)


def plot_pareto_by_shot(mean_rows: list[dict[str, object]], reasoning_mean_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in tradeoff_records(mean_rows, reasoning_mean_rows)
        if r["strategy_label"] == "All interventions"
        and r["ft_checkpoint"] in {"", "all-version"}
        and (r["method"] == "DIY IT" or "Two Pass" in r["method"])
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.9))
    fig.patch.set_facecolor("white")
    style_axis(ax)
    marker_by_method = {"DIY IT": "o", "Two Pass (No IT)": "s", "Two Pass (IT)": "D"}
    color_by_shot = {"": "#7CC7F2", "0": "#FFD8A7", "1": "#FFB86B", "2": "#FF9E45"}
    for r in rows:
        fam = method_family(str(r["method"]))
        shot = str(r["shot"])
        ax.scatter(fnum(r["mean_bias_error"]), fnum(r["mean_reasoning_accuracy"]), s=112, marker=marker_by_method[fam], color=color_by_shot[shot], edgecolor="#252A2E", linewidth=1.0, zorder=3)
        ax.text(fnum(r["mean_bias_error"]) + 0.04, fnum(r["mean_reasoning_accuracy"]) + 0.15, clean_method(str(r["method"])), fontsize=8.5)
    ax.set_xlabel("Mean bias error", fontweight="semibold", color="#374151")
    ax.set_ylabel("Mean reasoning accuracy (%)", fontweight="semibold", color="#374151")
    ax.set_title("Bias-utility tradeoff + shot", fontsize=16.0, fontweight="bold", pad=12)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_by_shot[""], markeredgecolor="#252A2E", label="DIY IT", markersize=8),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=color_by_shot["0"], markeredgecolor="#252A2E", label="0-shot", markersize=8),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=color_by_shot["1"], markeredgecolor="#252A2E", label="1-shot", markersize=8),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=color_by_shot["2"], markeredgecolor="#252A2E", label="2-shot", markersize=8),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.text(0.012, 0.02, "Same Pareto analysis, with one extra level: two-pass shot count. Lower x and higher y are better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    savefig(fig, "bias_reasoning_pareto_by_shot")
    return rows


def plot_pareto_by_intervention(mean_rows: list[dict[str, object]], reasoning_mean_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in tradeoff_records(mean_rows, reasoning_mean_rows)
        if r["strategy_label"] in INTERVENTIONS
        and (r["method"] == "DIY IT" or r["method"] in {"DIY Two Pass (No IT), 0-shot", "DIY Two Pass (IT), 0-shot"})
    ]
    fig, ax = plt.subplots(figsize=(8.7, 6.0))
    fig.patch.set_facecolor("white")
    style_axis(ax)
    marker_by_method = {"DIY IT": "o", "Two Pass (No IT)": "s", "Two Pass (IT)": "D"}
    for r in rows:
        fam = method_family(str(r["method"]))
        strategy = str(r["strategy_label"])
        ax.scatter(fnum(r["mean_bias_error"]), fnum(r["mean_reasoning_accuracy"]), s=94, marker=marker_by_method[fam], color=STRATEGY_COLORS[strategy], edgecolor="#252A2E", linewidth=0.9, zorder=3)
    ax.set_xlabel("Mean bias error", fontweight="semibold", color="#374151")
    ax.set_ylabel("Mean reasoning accuracy (%)", fontweight="semibold", color="#374151")
    ax.set_title("Bias-utility tradeoff + intervention", fontsize=16.0, fontweight="bold", pad=12)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=STRATEGY_COLORS[s], markeredgecolor="#252A2E", label=s, markersize=7.5) for s in INTERVENTIONS]
    ax.legend(handles=handles, loc="lower left", ncol=1, frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.text(0.012, 0.02, "Same Pareto analysis, with one extra level: intervention identity. Lower x and higher y are better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    savefig(fig, "bias_reasoning_pareto_by_intervention")
    return rows


def plot_ablation_by_shot(panel_rows: list[dict[str, object]], config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = [
        r
        for r in panel_rows
        if r["strategy_label"] in INTERVENTIONS
        and r["method_family"] in {"Two Pass (No IT)", "Two Pass (IT)"}
        and r["shot"] in {"0", "1", "2"}
    ]
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["strategy_label"], r["method_family"], r["shot"])].append(fnum(r["bias_reduction_vs_base"]))
    summary = [
        {
            "strategy_label": strategy,
            "method_family": family,
            "shot": shot,
            "mean_reduction_vs_base": float(np.mean(vals)),
            "n_panels": len(vals),
        }
        for (strategy, family, shot), vals in grouped.items()
    ]
    lookup = {(r["strategy_label"], r["method_family"], r["shot"]): fnum(r["mean_reduction_vs_base"]) for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.25), sharey=True)
    fig.patch.set_facecolor("white")
    x = np.arange(len(INTERVENTIONS))
    width = 0.23
    shot_colors = {"0": "#A7F3D0", "1": "#FDBA74", "2": "#C4B5FD"}
    for ax, family in zip(axes, ["Two Pass (No IT)", "Two Pass (IT)"]):
        style_axis(ax)
        for idx, shot in enumerate(["0", "1", "2"]):
            vals = [lookup.get((strategy, family, shot), np.nan) for strategy in INTERVENTIONS]
            ax.bar(x + (idx - 1) * width, vals, width=width, color=shot_colors[shot], edgecolor="#252A2E", linewidth=0.78, label=f"{shot}-shot", zorder=3)
        ax.axhline(0, color="#48515A", linewidth=0.85)
        ax.set_title(family, fontweight="bold", pad=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Stereo.\nreplace", "Individ.", "Perspective", "Counter.\nimaging", "Positive\ncontact"])
    axes[0].set_ylabel("Mean bias-error reduction", fontweight="semibold", color="#374151")
    handles = [
        Patch(facecolor=shot_colors[shot], edgecolor="#252A2E", label=f"{shot}-shot")
        for shot in ["0", "1", "2"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=True, fancybox=True, edgecolor="#B9C2CF")
    fig.suptitle("Intervention ablation + shot", y=0.995, fontsize=16.5, fontweight="bold")
    fig.text(0.012, 0.02, "Same intervention-ablation analysis, with one extra level: two-pass shot count. Higher is better.", fontsize=9.1, color="#374151")
    fig.tight_layout(rect=(0, 0.075, 1, 0.86), w_pad=1.2)
    savefig(fig, "intervention_ablation_by_shot")
    return summary


def main() -> None:
    ensure_dirs()
    set_style()

    config_rows = read_csv(SOURCE_CSV / "debiasing_method_bars_by_shot_configs_data.csv")
    panel_rows = config_panel_rows(config_rows)
    mean_rows = config_means(config_rows)
    intervention_rows = read_csv(SOURCE_CSV / "intervention_ablation_data.csv")
    baseline_rows = read_csv(SOURCE_CSV / "baseline_comparison_lollipop_data.csv")
    raw_rows = read_csv(core.INPUT)
    reasoning_mean_rows = mean_reasoning(raw_rows)

    write_csv("debiasing_performance_by_shot", plot_debiasing_by_shot(config_rows))
    write_csv("debiasing_performance_by_intervention", plot_debiasing_by_intervention(intervention_rows))
    write_csv("debiasing_performance_by_checkpoint", plot_debiasing_by_checkpoint(panel_rows))
    write_csv("baseline_comparison_by_dataset", plot_baseline_by_dataset(baseline_rows))
    write_csv("reasoning_performance_by_shot", plot_reasoning_by_shot(config_rows, raw_rows))
    write_csv("reasoning_performance_by_intervention", plot_reasoning_by_intervention(raw_rows))
    write_csv("bias_reasoning_pareto_by_shot", plot_pareto_by_shot(mean_rows, reasoning_mean_rows))
    write_csv("bias_reasoning_pareto_by_intervention", plot_pareto_by_intervention(mean_rows, reasoning_mean_rows))
    write_csv("intervention_ablation_by_shot", plot_ablation_by_shot(panel_rows, config_rows))


if __name__ == "__main__":
    main()
