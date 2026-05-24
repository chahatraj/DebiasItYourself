#!/usr/bin/env python3
"""Plot small Llama-8B instruction-tuning setting ablation.

This ablation compares LoRA instruction-tuning settings trained on 100 examples.
The figure answers which IT setting is most reliable across the completed
ablation metrics using the same normalization as the paper's debiasing plots:
distance from the unbiased target, with lower values better.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
RUN_TAG = "it_ablation_llama8b_small_20260506"
BBQ_FIX_RUN_TAG = "it_ablation_llama8b_small_20260506_bbq_stratified200_fix"
RESULTS_ROOT = ROOT / "results" / "new_results" / RUN_TAG
BBQ_FIX_RESULTS_ROOT = ROOT / "results" / "new_results" / BBQ_FIX_RUN_TAG
MODEL_ROOT = ROOT / "outputs" / "7_finetuned_models" / RUN_TAG
TRACKING_ROOT = ROOT / "tracking" / RUN_TAG
OUTDIR = ROOT / "figures" / "llama8b"

METRICS = [
    ("crows_error", "CrowS-Pairs\nstereotype preference"),
    ("bbq_ambig_error", "BBQ ambiguous\nbias score"),
    ("bbq_disambig_error", "BBQ disambiguated\nbias score"),
    ("winobias_gap", "WinoBias\npro/anti gap"),
]

COLORS = {
    "response_only": "#5EEAD4",
    "full_sequence": "#FDBA74",
}

EDGE = "#111827"


def read_csv_last(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[-1]


def as_float(value: str | int | float | None) -> float:
    if value is None:
        raise ValueError("Missing numeric value")
    return float(value)


def parse_trial_from_name(name: str, suffix: str) -> str:
    match = re.search(rf"ittrial_l8b_(.+?)_{suffix}", name)
    if not match:
        raise ValueError(f"Could not parse trial from {name}")
    return match.group(1)


def short_lr(value: str) -> str:
    return value.replace("e-0", "e-").replace("e-", "e-")


def trial_label(row: dict[str, object]) -> str:
    prefix = "Resp. only" if row["loss_mode"] == "response_only" else "Full seq."
    return (
        f"{prefix}\n"
        f"lr={short_lr(str(row['learning_rate']))}, "
        f"r={int(row['lora_r'])}, e={int(row['epochs'])}"
    )


def load_manifest() -> dict[str, dict[str, object]]:
    manifest_path = TRACKING_ROOT / "jobs_manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    return {trial["key"]: dict(trial) for trial in manifest["trials"]}


def collect_records() -> list[dict[str, object]]:
    records = load_manifest()

    for model_dir in MODEL_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        match = re.match(r"ittrial_l8b_(.+?)_ms-100-", model_dir.name)
        if not match:
            continue
        trial = match.group(1)
        if trial not in records:
            continue
        all_results = model_dir / "all_results.json"
        if all_results.exists():
            with all_results.open() as f:
                train_info = json.load(f)
            records[trial]["eval_loss"] = as_float(train_info.get("eval_loss"))

    for path in (RESULTS_ROOT / "evalshared").glob("crows_pairs_metrics_overall_*.csv"):
        trial = parse_trial_from_name(path.name, "crowspairs")
        if trial not in records:
            continue
        row = read_csv_last(path)
        pref = as_float(row["stereotype_preference_pct"])
        records[trial]["crows_stereotype_preference_pct"] = pref
        records[trial]["crows_error"] = abs(pref - 50.0)

    # Prefer the corrected stratified BBQ sample when present. The original
    # small run is retained as a fallback for any incomplete future reruns.
    seen_bbq_trials: set[str] = set()
    bbq_roots = [BBQ_FIX_RESULTS_ROOT / "bbq", RESULTS_ROOT / "bbq"]
    for bbq_root in bbq_roots:
        if not bbq_root.exists():
            continue
        for path in sorted(bbq_root.glob("bbq_metrics_*.csv")):
            trial = parse_trial_from_name(path.name, "bbq")
            if trial not in records or trial in seen_bbq_trials:
                continue
            row = read_csv_last(path)
            ambig = as_float(row["Bias_score_ambig"])
            disambig = as_float(row["Bias_score_disambig"])
            records[trial]["bbq_accuracy"] = as_float(row["Accuracy"])
            records[trial]["bbq_accuracy_ambig"] = as_float(row["Accuracy_ambig"])
            records[trial]["bbq_accuracy_disambig"] = as_float(row["Accuracy_disambig"])
            records[trial]["bbq_bias_ambig"] = ambig
            records[trial]["bbq_bias_disambig"] = disambig
            records[trial]["bbq_ambig_error"] = abs(ambig)
            records[trial]["bbq_disambig_error"] = abs(disambig)
            records[trial]["bbq_metric_source"] = str(path.relative_to(ROOT))
            seen_bbq_trials.add(trial)

    winobias_root = RESULTS_ROOT / "evalshared" / "winobias"
    for path in winobias_root.glob("*/winobias_metrics_overall_*.csv"):
        trial = parse_trial_from_name(path.name, "winobias")
        if trial not in records:
            continue
        row = read_csv_last(path)
        gap = as_float(row["abs_pro_anti_gap"])
        records[trial]["winobias_accuracy"] = as_float(row["accuracy"])
        records[trial]["winobias_gap"] = gap

    complete: list[dict[str, object]] = []
    for trial, row in records.items():
        missing = [metric_key for metric_key, _ in METRICS if metric_key not in row]
        if missing:
            raise RuntimeError(f"{trial} missing metrics: {missing}")
        row["trial"] = trial
        row["label"] = trial_label(row)
        complete.append(row)

    for row in complete:
        metric_values = [float(row[metric_key]) for metric_key, _ in METRICS]
        row["mean_normalized_bias_error"] = float(np.mean(metric_values))
        row["num_best_metrics"] = sum(
            1
            for metric_key, _ in METRICS
            if float(row[metric_key])
            == min(float(candidate[metric_key]) for candidate in complete)
        )

    return sorted(
        complete,
        key=lambda row: (
            float(row["mean_normalized_bias_error"]),
            str(row["trial"]),
        ),
    )


def write_csv(records: list[dict[str, object]]) -> None:
    (OUTDIR / "csv").mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "csv" / "it_setting_ablation_data.csv"
    fieldnames = [
        "overall_rank",
        "trial",
        "label",
        "loss_mode",
        "learning_rate",
        "lora_r",
        "lora_alpha",
        "epochs",
        "eval_loss",
        "mean_normalized_bias_error",
        "num_best_metrics",
        "crows_stereotype_preference_pct",
        "crows_error",
        "bbq_bias_ambig",
        "bbq_ambig_error",
        "bbq_bias_disambig",
        "bbq_disambig_error",
        "bbq_accuracy",
        "bbq_accuracy_ambig",
        "bbq_accuracy_disambig",
        "bbq_metric_source",
        "winobias_accuracy",
        "winobias_gap",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(records, start=1):
            out = {key: row.get(key, "") for key in fieldnames}
            out["overall_rank"] = i
            writer.writerow(out)


def add_bar_labels(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.2f}",
    pad: float = 0.05,
    min_text_pad: float = 0.0,
) -> None:
    for bar in bars:
        width = float(bar.get_width())
        ax.text(
            width + max(pad, min_text_pad),
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width),
            va="center",
            ha="left",
            fontsize=8.8,
            color="#374151",
        )


def plot(records: list[dict[str, object]]) -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.7,
            "axes.titlesize": 12.7,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.4,
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [str(row["label"]) for row in records]
    y = np.arange(len(records))
    colors = [COLORS[str(row["loss_mode"])] for row in records]
    hatches = ["///" if row["loss_mode"] == "full_sequence" else "" for row in records]

    fig = plt.figure(figsize=(14.8, 7.8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 5, width_ratios=[1.55, 1, 1, 1, 1], hspace=0.56, wspace=0.48)

    rank_ax = fig.add_subplot(gs[:, 0])
    bars = rank_ax.barh(
        y,
        [float(row["mean_normalized_bias_error"]) for row in records],
        color=colors,
        edgecolor=EDGE,
        linewidth=1.05,
        height=0.66,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    rank_ax.set_yticks(y)
    rank_ax.set_yticklabels(labels)
    rank_ax.invert_yaxis()
    rank_ax.set_xlabel("Mean bias error")
    rank_ax.set_title("Overall setting", fontweight="bold", pad=9)
    rank_ax.grid(axis="x", color="#E5E7EB", linestyle=":", linewidth=0.8, zorder=0)
    rank_ax.set_axisbelow(True)
    rank_ax.set_xlim(0, max(float(row["mean_normalized_bias_error"]) for row in records) * 1.17)
    add_bar_labels(rank_ax, bars, "{:.2f}", pad=0.06)
    rank_ax.text(
        0.0,
        -0.82,
        "Lower is better",
        fontsize=9,
        color="#4B5563",
        ha="left",
        va="center",
    )

    metric_axes = []
    for idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = fig.add_subplot(gs[idx // 2, 1 + (idx % 2) * 2 : 1 + (idx % 2) * 2 + 2])
        metric_axes.append(ax)
        values = [float(row[metric_key]) for row in records]
        bars = ax.barh(
            y,
            values,
            color=colors,
            edgecolor=EDGE,
            linewidth=0.95,
            height=0.64,
            zorder=3,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_title(metric_label, fontweight="bold", pad=7)
        ax.set_xlabel("Bias error")
        ax.grid(axis="x", color="#E5E7EB", linestyle=":", linewidth=0.75, zorder=0)
        ax.set_axisbelow(True)
        max_value = max(values)
        ax.set_xlim(0, max_value * 1.20 + 0.4)
        if max_value < 1:
            add_bar_labels(ax, bars, "{:.3f}", pad=max_value * 0.03 + 0.003)
        else:
            add_bar_labels(ax, bars, "{:.1f}", pad=max_value * 0.018 + 0.05)

    for ax in [rank_ax, *metric_axes]:
        ax.set_facecolor("#FCFCFD")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(colors="#374151")

    handles = [
        Patch(facecolor=COLORS["response_only"], edgecolor=EDGE, linewidth=0.95, label="Response-only loss"),
        Patch(facecolor=COLORS["full_sequence"], edgecolor=EDGE, linewidth=0.95, hatch="///", label="Full-sequence loss"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#B9C2CF",
        borderpad=0.45,
        columnspacing=1.4,
        handlelength=1.7,
    )

    best = records[0]
    fig.text(
        0.5,
        0.985,
        "Instruction-tuning setting ablation",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=EDGE,
    )
    fig.text(
        0.5,
        0.952,
        f"Best aggregate setting: {str(best['label']).replace(chr(10), ' ')}",
        ha="center",
        va="top",
        fontsize=11.2,
        color="#374151",
    )
    fig.text(
        0.012,
        0.022,
        "Normalization matches the paper figures: bars show distance from the unbiased target "
        "(50 for CrowS-Pairs and 0 for BBQ/WinoBias); lower is better. "
        "BBQ uses the corrected stratified-200 ablation sample when available.",
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#374151",
    )

    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.11)
    (OUTDIR / "pdf").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "pdf" / "it_setting_ablation.pdf", bbox_inches="tight")


def main() -> None:
    records = collect_records()
    write_csv(records)
    plot(records)


if __name__ == "__main__":
    main()
