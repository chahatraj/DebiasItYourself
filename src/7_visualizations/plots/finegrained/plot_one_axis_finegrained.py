#!/usr/bin/env python3
"""Regenerate one-axis fine-grained Llama-8B diagnostic figures.

The main fine-grained script now writes the richer multi-axis plots. This
script keeps the older one-axis view refreshed from the current audit CSVs so
the whole figures tree is regenerated from reproducible plotting code.
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


ROOT = Path(__file__).resolve().parents[4]
PLOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLOT_DIR))
from plot_style import use_nimbus_sans  # noqa: E402
SOURCE = ROOT / "figures/llama8b/csv"
OUT = ROOT / "figures/llama8b/finegrained/one_axis"
CSV_OUT = OUT / "csv"
PDF_OUT = OUT / "pdf"

STRATEGY_ORDER = [
    "Stereotype replacement",
    "Individuation",
    "Perspective-taking",
    "Counter-stereotypic imaging",
    "Positive contact",
]

STRATEGY_SLUGS = {
    "stereotype_replacement_all_versions",
    "individuation_all_versions",
    "perspective_taking_all_versions",
    "counter_stereotypic_imaging_all_versions",
    "positive_contact_all_versions",
}

PALETTE = {
    "base": "#AEB7C4",
    "baseline": "#88A0D8",
    "diy": "#FF9E45",
    "DIY IT": "#7CC7F2",
    "Two Pass (No IT)": "#50C77B",
    "Two Pass (IT)": "#FF9E45",
    "All interventions": "#8EA7E9",
    "Stereotype replacement": "#7DD3FC",
    "Individuation": "#5EEAD4",
    "Perspective-taking": "#A7F3D0",
    "Counter-stereotypic imaging": "#FDBA74",
    "Positive contact": "#C4B5FD",
}


def set_style() -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11.8,
            "axes.titlesize": 14.0,
            "axes.labelsize": 12.2,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: str | float | int | None) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def method_family(method: str) -> str:
    if method == "DIY IT":
        return "DIY IT"
    if "No IT" in method:
        return "Two Pass (No IT)"
    if "Two Pass (IT)" in method:
        return "Two Pass (IT)"
    return method


def shot_value(method: str, shot: str) -> str:
    if shot:
        return str(int(float(shot)))
    match = re.search(r"(\d)-shot", method)
    return match.group(1) if match else ""


def base_errors(configs: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in configs:
        if row["method"].startswith("Base"):
            out[(row["figure_slug"], row["panel_label"])] = fnum(row["normalized_bias_error_plotted"])
    return out


def config_rows_with_reduction(configs: list[dict[str, str]]) -> list[dict[str, object]]:
    bases = base_errors(configs)
    rows: list[dict[str, object]] = []
    for row in configs:
        if row["method"].startswith("Base"):
            continue
        bias_error = fnum(row["normalized_bias_error_plotted"])
        base = bases.get((row["figure_slug"], row["panel_label"]))
        reduction = float("nan") if base is None else base - bias_error
        rows.append(
            {
                "figure_slug": row["figure_slug"],
                "strategy_key": row["strategy_key"],
                "strategy_label": row["strategy_label"],
                "ft_checkpoint": row["ft_checkpoint"],
                "dataset_key": row["dataset_key"],
                "panel_label": row["panel_label"],
                "method": row["method"],
                "method_family": method_family(row["method"]),
                "shot": shot_value(row["method"], row["shot"]),
                "bias_error": bias_error,
                "bias_reduction_vs_base": reduction,
                "source_model": row["model"],
                "metric_source_file": row["metric_source_file"],
            }
        )
    return rows


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(PDF_OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_mean(rows: list[dict[str, object]], keys: list[str], value: str) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    prototypes: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups[key].append(float(row[value]))
        prototypes[key] = row
    out: list[dict[str, object]] = []
    for key, vals in groups.items():
        proto = {k: prototypes[key][k] for k in keys}
        proto[f"mean_{value}"] = mean(vals)
        proto["n_panels"] = len(vals)
        out.append(proto)
    return out


def plot_horizontal_bars(rows: list[dict[str, object]], label_key: str, value_key: str, title: str, xlabel: str, name: str) -> None:
    rows = sorted(rows, key=lambda r: float(r[value_key]), reverse=True)
    fig_h = max(4.0, 0.35 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    y = np.arange(len(rows))
    colors = [
        PALETTE["diy"] if str(r.get("group", "")).startswith("diy") or str(r.get(label_key, "")).startswith("DIY") else PALETTE.get(str(r.get("group", "")), PALETTE["baseline"])
        for r in rows
    ]
    ax.barh(y, [float(r[value_key]) for r in rows], color=colors, edgecolor="#333333", linewidth=1.0)
    ax.set_yticks(y, [str(r[label_key]) for r in rows])
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.8)
    save(fig, name)


def baseline_comparison() -> None:
    source = read_csv(SOURCE / "baseline_comparison_lollipop_data.csv")
    groups: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, str]] = {}
    for row in source:
        groups[row["method_key"]].append(fnum(row["normalized_bias_error_plotted"]))
        meta[row["method_key"]] = row
    rows = [
        {
            "method_key": key,
            "method_label": meta[key]["method_label"],
            "group": meta[key]["group"],
            "mean_bias_error": mean(vals),
            "n_datasets": len(vals),
        }
        for key, vals in groups.items()
    ]
    rows.sort(key=lambda r: float(r["mean_bias_error"]))
    write_csv(CSV_OUT / "baseline_comparison_by_dataset.csv", rows)
    plot_horizontal_bars(
        rows,
        "method_label",
        "mean_bias_error",
        "Mean Bias Error Across Benchmark Panels",
        "Mean normalized bias error (lower is better)",
        "baseline_comparison_by_dataset",
    )


def debiasing_performance_by_intervention(interventions: list[dict[str, str]]) -> None:
    rows = [r for r in interventions if not r["method"].startswith("Base")]
    write_csv(CSV_OUT / "debiasing_performance_by_intervention.csv", rows, list(rows[0].keys()))

    means = grouped_mean(
        [
            {
                "intervention_label": r["intervention_label"],
                "method": r["method"],
                "bias_reduction_vs_base": fnum(r["bias_error_reduction_vs_base"]),
            }
            for r in rows
        ],
        ["intervention_label", "method"],
        "bias_reduction_vs_base",
    )
    x = np.arange(len(STRATEGY_ORDER))
    methods = ["DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"]
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    for i, method in enumerate(methods):
        vals = []
        for strategy in STRATEGY_ORDER:
            hit = [r for r in means if r["intervention_label"] == strategy and r["method"] == method]
            vals.append(float(hit[0]["mean_bias_reduction_vs_base"]) if hit else np.nan)
        fam = method_family(method)
        ax.bar(x + (i - 1) * width, vals, width, label=method, color=PALETTE[fam], edgecolor="#333333", linewidth=1.0)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, STRATEGY_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Bias error reduction vs. base")
    ax.set_title("Intervention Ablation by DIY Method", fontweight="bold")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    save(fig, "debiasing_performance_by_intervention")


def debiasing_performance_by_shot(configs: list[dict[str, str]]) -> None:
    rows = [r for r in configs if r["figure_slug"] == "all_strategies_all_versions"]
    write_csv(CSV_OUT / "debiasing_performance_by_shot.csv", rows, list(rows[0].keys()))

    reduced = config_rows_with_reduction(rows)
    means = grouped_mean(reduced, ["method", "method_family", "shot"], "bias_error")
    means.sort(key=lambda r: (str(r["method_family"]), str(r["shot"])))
    labels = [str(r["method"]).replace("DIY Two Pass (No IT), ", "No IT ").replace("DIY Two Pass (IT), ", "IT ") for r in means]
    colors = [PALETTE.get(str(r["method_family"]), PALETTE["baseline"]) for r in means]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(means))
    ax.bar(x, [float(r["mean_bias_error"]) for r in means], color=colors, edgecolor="#333333", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Mean normalized bias error")
    ax.set_title("All-Intervention Bias Error by Shot", fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    save(fig, "debiasing_performance_by_shot")


def debiasing_performance_by_checkpoint(configs: list[dict[str, str]]) -> None:
    reduced = [
        r
        for r in config_rows_with_reduction(configs)
        if r["strategy_key"] == "all_strategies" and r["method_family"] in {"DIY IT", "Two Pass (IT)"}
    ]
    write_csv(CSV_OUT / "debiasing_performance_by_checkpoint.csv", reduced)
    means = grouped_mean(reduced, ["ft_checkpoint", "method_family", "shot"], "bias_reduction_vs_base")
    checkpoints = ["all-version", "opinion", "action", "event"]
    series = [
        ("DIY IT", ""),
        ("Two Pass (IT)", "0"),
        ("Two Pass (IT)", "1"),
        ("Two Pass (IT)", "2"),
    ]
    x = np.arange(len(checkpoints))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for i, (family, shot) in enumerate(series):
        vals = []
        for ckpt in checkpoints:
            hit = [r for r in means if r["ft_checkpoint"] == ckpt and r["method_family"] == family and str(r["shot"]) == shot]
            vals.append(float(hit[0]["mean_bias_reduction_vs_base"]) if hit else np.nan)
        label = family if shot == "" else f"{family}, {shot}-shot"
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=PALETTE.get(family, PALETTE["baseline"]), edgecolor="#333333", linewidth=1.0, alpha=0.72 + 0.08 * i)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, checkpoints)
    ax.set_ylabel("Bias error reduction vs. base")
    ax.set_title("Checkpoint Sensitivity for All-Intervention DIY", fontweight="bold")
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    save(fig, "debiasing_performance_by_checkpoint")


def intervention_ablation_by_shot(configs: list[dict[str, str]]) -> None:
    reduced = [
        r
        for r in config_rows_with_reduction(configs)
        if r["figure_slug"] in STRATEGY_SLUGS and r["method_family"] in {"Two Pass (No IT)", "Two Pass (IT)"}
    ]
    rows = grouped_mean(reduced, ["strategy_label", "method_family", "shot"], "bias_reduction_vs_base")
    rows = [
        {
            "strategy_label": r["strategy_label"],
            "method_family": r["method_family"],
            "shot": r["shot"],
            "mean_reduction_vs_base": r["mean_bias_reduction_vs_base"],
            "n_panels": r["n_panels"],
        }
        for r in rows
    ]
    rows.sort(key=lambda r: (STRATEGY_ORDER.index(str(r["strategy_label"])), str(r["method_family"]), str(r["shot"])))
    write_csv(CSV_OUT / "intervention_ablation_by_shot.csv", rows)

    labels = [f"{r['strategy_label']} | {r['method_family'].replace('Two Pass ', '')} {r['shot']}" for r in rows]
    plot_horizontal_bars(
        rows,
        "strategy_label",
        "mean_reduction_vs_base",
        "Bias Error Reduction by Intervention, Method, and Shot",
        "Mean reduction vs. base (higher is better)",
        "intervention_ablation_by_shot",
    )
    # Replace crowded default y labels with compact labels.
    fig, ax = plt.subplots(figsize=(9.2, 9.2))
    y = np.arange(len(rows))
    colors = [PALETTE.get(str(r["method_family"]), PALETTE["baseline"]) for r in rows]
    ax.barh(y, [float(r["mean_reduction_vs_base"]) for r in rows], color=colors, edgecolor="#333333", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Mean reduction vs. base (higher is better)")
    ax.set_title("Bias Error Reduction by Intervention, Method, and Shot", fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.8)
    save(fig, "intervention_ablation_by_shot")


def refresh_reasoning_csv_and_plot(name: str) -> list[dict[str, str]]:
    path = CSV_OUT / f"{name}.csv"
    rows = read_csv(path)
    write_csv(path, rows, list(rows[0].keys()))

    means = grouped_mean(
        [
            {
                "strategy_label": r.get("strategy_label", "All interventions"),
                "method_family": r["method_family"],
                "shot": r["shot"],
                "accuracy": fnum(r["accuracy"]),
            }
            for r in rows
        ],
        ["strategy_label", "method_family", "shot"],
        "accuracy",
    )
    if "intervention" in name:
        plot_rows = [
            r
            for r in means
            if r["strategy_label"] in STRATEGY_ORDER and (r["shot"] in {"", "0"} or r["method_family"] == "DIY IT")
        ]
        labels = [f"{r['strategy_label']} | {r['method_family'].replace('Two Pass ', '')}" for r in plot_rows]
        title = "Reasoning Accuracy by Intervention"
    else:
        plot_rows = sorted(means, key=lambda r: (str(r["method_family"]), str(r["shot"])))
        labels = [f"{r['method_family'].replace('Two Pass ', '')} {r['shot']}".strip() for r in plot_rows]
        title = "Reasoning Accuracy by Shot"

    fig_h = 4.8 if len(plot_rows) <= 10 else 7.6
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    y = np.arange(len(plot_rows))
    colors = [PALETTE.get(str(r["method_family"]), PALETTE["baseline"]) for r in plot_rows]
    ax.barh(y, [float(r["mean_accuracy"]) for r in plot_rows], color=colors, edgecolor="#333333", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean reasoning accuracy")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.8)
    save(fig, name)
    return rows


def pareto_from_bias_and_reasoning(
    configs: list[dict[str, str]], reasoning_rows: list[dict[str, str]], by_intervention: bool
) -> list[dict[str, object]]:
    reduced = config_rows_with_reduction(configs)
    if by_intervention:
        bias_source = [
            r
            for r in reduced
            if r["figure_slug"] in STRATEGY_SLUGS
            and (r["method_family"] == "DIY IT" or str(r["shot"]) == "0")
        ]
        reason_key_fields = ["figure_slug", "method_family", "shot"]
    else:
        bias_source = [r for r in reduced if r["figure_slug"] == "all_strategies_all_versions"]
        reason_key_fields = ["method_family", "shot"]

    bias_means = grouped_mean(
        bias_source,
        ["figure_slug", "strategy_label", "ft_checkpoint", "method", "method_family", "shot"],
        "bias_error",
    )
    reduction_means = grouped_mean(
        bias_source,
        ["figure_slug", "method", "method_family", "shot"],
        "bias_reduction_vs_base",
    )
    reduction_lookup = {
        (r["figure_slug"], r["method"], r["method_family"], r["shot"]): r["mean_bias_reduction_vs_base"]
        for r in reduction_means
    }

    reason_groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in reasoning_rows:
        key = tuple(row.get(k, "") for k in reason_key_fields)
        reason_groups[key].append(fnum(row["accuracy"]))

    out: list[dict[str, object]] = []
    for row in bias_means:
        key = tuple(str(row[k]) for k in reason_key_fields)
        if key not in reason_groups:
            continue
        out.append(
            {
                "figure_slug": row["figure_slug"],
                "strategy_label": row["strategy_label"],
                "ft_checkpoint": row["ft_checkpoint"],
                "method": row["method"],
                "method_family": row["method_family"],
                "shot": row["shot"],
                "mean_bias_error": row["mean_bias_error"],
                "mean_reduction_vs_base": reduction_lookup.get(
                    (row["figure_slug"], row["method"], row["method_family"], row["shot"]), float("nan")
                ),
                "mean_reasoning_accuracy": mean(reason_groups[key]),
                "n_bias_panels": row["n_panels"],
                "n_reasoning_benchmarks": len(reason_groups[key]),
            }
        )
    out.sort(key=lambda r: (float(r["mean_bias_error"]), -float(r["mean_reasoning_accuracy"])))
    return out


def plot_pareto(rows: list[dict[str, object]], title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for family in ["DIY IT", "Two Pass (No IT)", "Two Pass (IT)"]:
        sub = [r for r in rows if r["method_family"] == family]
        if not sub:
            continue
        ax.scatter(
            [float(r["mean_bias_error"]) for r in sub],
            [float(r["mean_reasoning_accuracy"]) for r in sub],
            s=110,
            color=PALETTE[family],
            edgecolor="#222222",
            linewidth=1.1,
            label=family,
        )
    ax.set_xlabel("Mean normalized bias error (lower is better)")
    ax.set_ylabel("Mean reasoning accuracy")
    ax.set_title(title, fontweight="bold")
    ax.grid(color="#E8E8E8", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    save(fig, name)


def main() -> None:
    set_style()
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)

    configs = read_csv(SOURCE / "debiasing_method_bars_by_shot_configs_data.csv")
    interventions = read_csv(SOURCE / "intervention_ablation_data.csv")

    baseline_comparison()
    debiasing_performance_by_intervention(interventions)
    debiasing_performance_by_shot(configs)
    debiasing_performance_by_checkpoint(configs)
    intervention_ablation_by_shot(configs)

    reasoning_by_shot = refresh_reasoning_csv_and_plot("reasoning_performance_by_shot")
    reasoning_by_intervention = refresh_reasoning_csv_and_plot("reasoning_performance_by_intervention")

    pareto_shot = pareto_from_bias_and_reasoning(configs, reasoning_by_shot, by_intervention=False)
    write_csv(CSV_OUT / "bias_reasoning_pareto_by_shot.csv", pareto_shot)
    plot_pareto(pareto_shot, "Bias and Reasoning Tradeoff by Shot", "bias_reasoning_pareto_by_shot")

    pareto_intervention = pareto_from_bias_and_reasoning(configs, reasoning_by_intervention, by_intervention=True)
    write_csv(CSV_OUT / "bias_reasoning_pareto_by_intervention.csv", pareto_intervention)
    plot_pareto(pareto_intervention, "Bias and Reasoning Tradeoff by Intervention", "bias_reasoning_pareto_by_intervention")

    print(f"Wrote one-axis fine-grained figures to {OUT}")


if __name__ == "__main__":
    main()
