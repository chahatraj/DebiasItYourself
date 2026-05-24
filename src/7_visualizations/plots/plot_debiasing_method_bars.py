#!/usr/bin/env python3
"""Bar chart for core DIY debiasing-performance settings.

The plot uses the benchmark-native metrics requested for the paper, normalized
as bias error so lower is always better:
  - CrowS-Pairs: stereotype preference %
  - StereoSet: SS Score, overall
  - BBQ: ambiguous and disambiguated bias scores
  - WinoBias: absolute pro/anti accuracy gap
  - WinoGender: male/female pair disagreement rate
"""

from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "results/new_results/all_results_dataset_slides.test.csv"
OUTDIR = ROOT / "figures" / "llama8b"
ICL_ROOT = (
    ROOT
    / "results/new_results/m4_base_icl_zero/m4_base_icl_zero_allmodels_20260514_005736/llama8b"
)

PANELS = [
    ("crowspairs", "CrowS-Pairs", "stereotype_preference_pct", "Stereotype preference %", 50.0),
    ("stereoset", "StereoSet", "SS Score", "SS Score overall", 50.0),
    ("bbq", "BBQ Ambig.", "Bias_score_ambig", "Bias score ambig.", 0.0),
    ("bbq", "BBQ Disambig.", "Bias_score_disambig", "Bias score disambig.", 0.0),
    ("winobias", "WinoBias", "abs_pro_anti_gap", "Abs. pro/anti gap", 0.0),
    (
        "winogender",
        "WinoGender",
        "male_female_pair_disagreement_rate",
        "Male/female pair disagreement",
        0.0,
    ),
]

METHODS = [
    "Base Model\nInference",
    "ICL",
    "DIY IT",
    "DIY Two Pass\n(No IT)",
    "DIY Two Pass\n(IT)",
]

COLORS = {
    "Base Model\nInference": "#C7CCD8",
    "ICL": "#C4B5FD",
    "DIY IT": "#9BD4F0",
    "DIY Two Pass\n(No IT)": "#B9E6C9",
    "DIY Two Pass\n(IT)": "#FFD2A8",
}

HATCHES = {
    "Base Model\nInference": "",
    "ICL": "xx",
    "DIY IT": "///",
    "DIY Two Pass\n(No IT)": "\\\\\\",
    "DIY Two Pass\n(IT)": "...",
}

PREFERRED_M6_RUNS = [
    "m6sd_20260326_bbq_opinion_zero_mem120",
    "m6sd_20260325_bbq_allstrategies_maxlenfix_zero",
    "m6sd_20260324_twopass_zero_allstrategies_truncfix",
]

FT_ALL_STRATEGIES_MODEL = (
    "m6_two_pass__finetuned_ms-500-allstrategies-opinion-action-event-allversions"
    "__all_strategies"
)
BBQ_ARCHIVE_OUTPUTS = (
    ROOT
    / "archive/m6_llama8b_dedup_apply_20260504_000252/outputs/new_outputs"
)
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


def valid_score(row: dict[str, str]) -> bool:
    try:
        float(row["score"])
    except (TypeError, ValueError):
        return False
    return True


def bias_error(row: dict[str, str]) -> float:
    score = float(row["score"])
    direction = row["direction"]
    if direction == "target50":
        return abs(score - 50.0)
    if direction == "lower":
        return score
    if direction == "higher":
        return 100.0 - score
    raise ValueError(f"Unknown direction: {direction}")


def normalize_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    while text.endswith("."):
        text = text[:-1].strip()
    return text


def load_bbq_target_locations() -> dict[tuple[str, str], int]:
    path = ROOT / "data/BBQ/analysis_scripts/additional_metadata.csv"
    targets: dict[tuple[str, str], int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                target_loc = int(float(row["target_loc"]))
            except (KeyError, TypeError, ValueError):
                continue
            category = row["category"].replace(".jsonl", "").strip()
            key = (category, str(row["example_id"]).strip())
            targets.setdefault(key, target_loc)
    return targets


def bbq_pred_category(row: dict[str, str]) -> str:
    try:
        pred_idx = int(float(row.get("pred_index") or row.get("pred_label") or ""))
    except (TypeError, ValueError):
        return ""
    selected_answer = normalize_text(row.get(f"ans{pred_idx}", row.get("model_output", "")))
    if selected_answer in BBQ_UNKNOWN_STRINGS:
        return "unknown"
    return selected_answer


def context_bias_score(rows: list[dict[str, bool | str]]) -> float:
    non_unknown = [r for r in rows if not r["unknown"]]
    if not non_unknown:
        return float("nan")
    target_rate = sum(bool(r["target_selected"]) for r in non_unknown) / len(non_unknown)
    return target_rate * 2.0 - 1.0


def recompute_bbq_metrics_from_predictions(pred_dir: Path) -> dict[str, float]:
    targets = load_bbq_target_locations()
    eval_rows: list[dict[str, bool | str]] = []
    for pred_path in sorted(glob.glob(str(pred_dir / "bbq_preds*.csv"))):
        with open(pred_path, newline="") as f:
            for row in csv.DictReader(f):
                category = (
                    row.get("source_file") or row.get("category") or ""
                ).replace(".jsonl", "").strip()
                key = (category, str(row.get("example_id", "")).strip())
                if key not in targets:
                    continue
                try:
                    pred_idx = int(float(row.get("pred_index") or row.get("pred_label") or ""))
                    label = int(float(row["label"]))
                except (KeyError, TypeError, ValueError):
                    continue
                eval_rows.append(
                    {
                        "context_condition": row["context_condition"].strip().lower(),
                        "correct": pred_idx == label,
                        "target_selected": pred_idx == targets[key],
                        "unknown": bbq_pred_category(row) == "unknown",
                    }
                )

    if not eval_rows:
        raise RuntimeError(f"No valid BBQ prediction rows found in {pred_dir}")

    ambig_rows = [r for r in eval_rows if r["context_condition"] == "ambig"]
    disambig_rows = [r for r in eval_rows if r["context_condition"] == "disambig"]
    acc_ambig = sum(bool(r["correct"]) for r in ambig_rows) / len(ambig_rows)
    ambig_bias = context_bias_score(ambig_rows)
    disambig_bias = context_bias_score(disambig_rows)

    return {
        "Bias_score_ambig": round((1.0 - acc_ambig) * ambig_bias * 100.0, 3),
        "Bias_score_disambig": round(disambig_bias * 100.0, 3),
    }


def archived_bbq_prediction_dir(source_file: str) -> Path | None:
    prefix = "results/new_results/"
    suffix = "/bbq/bbq_metrics.csv"
    if not source_file.startswith(prefix) or not source_file.endswith(suffix):
        return None
    rel = source_file[len(prefix) : -len("/bbq_metrics.csv")]
    pred_dir = BBQ_ARCHIVE_OUTPUTS / rel
    return pred_dir if pred_dir.is_dir() else None


def read_metric_file(row: dict[str, str], metric: str) -> float:
    path = ROOT / row["source_file"]
    if not path.exists():
        if row["dataset_key"] == "bbq":
            pred_dir = archived_bbq_prediction_dir(row["source_file"])
            if pred_dir is not None:
                return recompute_bbq_metrics_from_predictions(pred_dir)[metric]
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        metric_rows = list(csv.DictReader(f))

    if row["dataset_key"] == "stereoset":
        metric_rows = [
            r
            for r in metric_rows
            if r.get("split") == "overall" and r.get("domain") == "overall"
        ]
    elif row["dataset_key"] == "bbq":
        metric_rows = [
            r
            for r in metric_rows
            if r.get("input_file") == "__overall__" or r.get("Model") == row["model"]
        ]

    if not metric_rows:
        raise RuntimeError(f"No metric rows found in {path}")
    if metric not in metric_rows[-1]:
        raise KeyError(f"{metric} missing from {path}")
    return float(metric_rows[-1][metric])


def icl_bias_path(dataset_key: str) -> Path | None:
    if dataset_key == "bbq":
        return ICL_ROOT / "bbq/bbq_metrics_m4_baseicl_zero_llama8b_allstrat_bbq.csv"
    if dataset_key == "crowspairs":
        return ICL_ROOT / "evalshared/crows_pairs_metrics_overall_m4_baseicl_zero_llama8b_allstrat_crowspairs.csv"
    if dataset_key == "stereoset":
        return ICL_ROOT / "evalshared/stereoset_metrics_m4_baseicl_zero_llama8b_allstrat_stereoset.csv"
    if dataset_key == "winobias":
        return (
            ICL_ROOT
            / "evalshared/winobias/m4_baseicl_zero_llama8b_allstrat_winobias/"
            / "winobias_metrics_overall_m4_baseicl_zero_llama8b_allstrat_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            ICL_ROOT
            / "evalshared/winogender/m4_baseicl_zero_llama8b_allstrat_winogender/"
            / "winogender_metrics_overall_m4_baseicl_zero_llama8b_allstrat_winogender.csv"
        )
    return None


def read_metric_path(path: Path, dataset_key: str, metric: str) -> float:
    with path.open(newline="") as f:
        metric_rows = list(csv.DictReader(f))

    if dataset_key == "stereoset":
        metric_rows = [
            r
            for r in metric_rows
            if r.get("split") == "overall" and r.get("domain") == "overall"
        ]
    elif dataset_key == "bbq":
        metric_rows = [
            r
            for r in metric_rows
            if r.get("input_file") == "__overall__"
            or r.get("Category", "").lower() == "overall"
            or r.get("Model", "").lower() == "overall"
        ] or metric_rows

    if not metric_rows:
        raise RuntimeError(f"No metric rows found in {path}")
    if metric not in metric_rows[-1]:
        raise KeyError(f"{metric} missing from {path}")
    return float(metric_rows[-1][metric])


def metric_value(row: dict[str, str], metric: str) -> tuple[float, str]:
    metric_path = ROOT / row["source_file"]
    if not metric_path.exists() and row["dataset_key"] == "bbq":
        pred_dir = archived_bbq_prediction_dir(row["source_file"])
        if pred_dir is not None:
            rel_pred_dir = pred_dir.relative_to(ROOT)
            return (
                recompute_bbq_metrics_from_predictions(pred_dir)[metric],
                f"recomputed_from_predictions:{rel_pred_dir}",
            )
    try:
        return read_metric_file(row, metric), row["source_file"]
    except (FileNotFoundError, KeyError, RuntimeError):
        # The aggregate CSV carries only one score per dataset. This fallback is
        # only valid when that aggregate score is exactly the requested metric.
        aggregate_aliases = {
            "stereotype_preference_pct": "metric_score",
            "abs_pro_anti_gap": "abs_pro_anti_gap",
        }
        if row["score_label"] == aggregate_aliases.get(metric):
            return float(row["score"]), row["source_file"]
        raise


def normalized_bias_error(value: float, ideal: float) -> float:
    return abs(value - ideal)


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="") as f:
        return list(csv.DictReader(f))


def base_row(rows: list[dict[str, str]], dataset_key: str) -> dict[str, str]:
    candidates = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key
        and r["model"] == "llama_8b"
        and "base" in r["name"]
        and valid_score(r)
    ]
    if not candidates:
        candidates = [
            r
            for r in rows
            if r["dataset_key"] == dataset_key and r["model"] == "llama_8b" and valid_score(r)
        ]
    if not candidates:
        raise RuntimeError(f"No base row found for {dataset_key}")
    return candidates[0]


def preferred_run_rank(row: dict[str, str]) -> tuple[int, str]:
    source_path = ROOT / row["source_file"]
    recoverable = (
        row["dataset_key"] == "bbq"
        and archived_bbq_prediction_dir(row["source_file"]) is not None
    )
    exists_penalty = 0 if source_path.exists() or recoverable else 100
    source = row["source_file"]
    for i, run in enumerate(PREFERRED_M6_RUNS):
        if run in source:
            return exists_penalty + i, source
    return exists_penalty + len(PREFERRED_M6_RUNS), source


def exact_m3_model(dataset_key: str) -> re.Pattern[str]:
    if dataset_key == "bbq":
        return re.compile(r"^m3_finetune_allstrat:")
    if dataset_key == "winobias":
        return re.compile(r"^m3_finetune_allstrat_winobias$")
    if dataset_key == "winogender":
        return re.compile(r"^m3_finetune_allstrat_winogender$")
    return re.compile(r"^m3_finetune_allstrat$")


def canonical_method_row(
    rows: list[dict[str, str]], dataset_key: str, method_label: str
) -> dict[str, str] | None:
    dataset_rows = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key and r["type"] == "method" and valid_score(r)
    ]

    if method_label == "DIY IT":
        model_pattern = exact_m3_model(dataset_key)
        candidates = [
            r
            for r in dataset_rows
            if r["name"] == "m3_finetune_llama_ms500"
            and model_pattern.search(r["model"])
        ]
        return candidates[0] if candidates else None

    if method_label == "DIY Two Pass\n(No IT)":
        candidates = [
            r
            for r in dataset_rows
            if r["name"] == "m6_self_debiasing"
            and r["model"].startswith("m6_two_pass__base__all_strategies")
            and "two_pass_one" not in r["model"]
            and "two_pass_two" not in r["model"]
            and "two_pass_five" not in r["model"]
            and "same_thread" not in r["model"]
            and "noncog" not in r["model"]
        ]
        return sorted(candidates, key=preferred_run_rank)[0] if candidates else None

    if method_label == "DIY Two Pass\n(IT)":
        candidates = [
            r
            for r in dataset_rows
            if r["name"] == "m6_self_debiasing"
            and r["model"].startswith(FT_ALL_STRATEGIES_MODEL)
            and "two_pass_one" not in r["model"]
            and "two_pass_two" not in r["model"]
            and "two_pass_five" not in r["model"]
            and "same_thread" not in r["model"]
            and "noncog" not in r["model"]
        ]
        return sorted(candidates, key=preferred_run_rank)[0] if candidates else None

    raise ValueError(f"Unknown method label: {method_label}")


def build_table(rows: list[dict[str, str]]):
    matrix = np.full((len(PANELS), len(METHODS)), np.nan)
    audit: list[dict[str, str]] = []

    for i, (dataset_key, panel_label, metric, metric_label, ideal) in enumerate(PANELS):
        base = base_row(rows, dataset_key)
        native_value, metric_source = metric_value(base, metric)
        value = normalized_bias_error(native_value, ideal)
        matrix[i, 0] = value
        audit.append(
            {
                "dataset_key": dataset_key,
                "panel_label": panel_label,
                "method": METHODS[0].replace("\n", " "),
                "metric": metric,
                "metric_label": metric_label,
                "ideal_value": f"{ideal:.6g}",
                "native_value": f"{native_value:.6g}",
                "normalized_bias_error_plotted": f"{value:.6g}",
                "aggregate_score": base["score"],
                "aggregate_score_label": base["score_label"],
                "aggregate_direction": base["direction"],
                "name": base["name"],
                "model": base["model"],
                "strategy": base["strategy"],
                "metric_source_file": metric_source,
            }
        )

        icl_path = icl_bias_path(dataset_key)
        if icl_path is not None and icl_path.exists():
            native_value = read_metric_path(icl_path, dataset_key, metric)
            value = normalized_bias_error(native_value, ideal)
            matrix[i, 1] = value
            audit.append(
                {
                    "dataset_key": dataset_key,
                    "panel_label": panel_label,
                    "method": METHODS[1].replace("\n", " "),
                    "metric": metric,
                    "metric_label": metric_label,
                    "ideal_value": f"{ideal:.6g}",
                    "native_value": f"{native_value:.6g}",
                    "normalized_bias_error_plotted": f"{value:.6g}",
                    "aggregate_score": "",
                    "aggregate_score_label": "",
                    "aggregate_direction": "",
                    "name": "m4_base_icl_zero",
                    "model": "m4_baseicl_zero_llama8b_allstrat",
                    "strategy": "all_strategies",
                    "metric_source_file": str(icl_path.relative_to(ROOT)),
                }
            )

        for j, method_label in enumerate(METHODS[2:], start=2):
            row = canonical_method_row(rows, dataset_key, method_label)
            if row is None:
                continue
            native_value, metric_source = metric_value(row, metric)
            value = normalized_bias_error(native_value, ideal)
            matrix[i, j] = value
            audit.append(
                {
                    "dataset_key": dataset_key,
                    "panel_label": panel_label,
                    "method": method_label.replace("\n", " "),
                    "metric": metric,
                    "metric_label": metric_label,
                    "ideal_value": f"{ideal:.6g}",
                    "native_value": f"{native_value:.6g}",
                    "normalized_bias_error_plotted": f"{value:.6g}",
                    "aggregate_score": row["score"],
                    "aggregate_score_label": row["score_label"],
                    "aggregate_direction": row["direction"],
                    "name": row["name"],
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "metric_source_file": metric_source,
                }
            )

    return matrix, audit


def write_csv(audit: list[dict[str, str]]) -> None:
    path = OUTDIR / "csv/debiasing_method_bars_data.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        writer.writeheader()
        writer.writerows(audit)


def plot(matrix: np.ndarray) -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.2,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.1, 6.55), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    x = np.arange(len(METHODS))
    method_labels = ["Base", "ICL", "DIY\nIT", "DIY Two\nPass\n(No IT)", "DIY Two\nPass\n(IT)"]

    for i, (ax, (_, panel_label, _, metric_label, ideal)) in enumerate(zip(axes, PANELS)):
        values = matrix[i]
        ax.set_facecolor("#FBFCFE")
        bars = ax.bar(
            x,
            values,
            width=0.68,
            color=[COLORS[m] for m in METHODS],
            edgecolor="#2F3437",
            linewidth=0.75,
        )
        for bar, method in zip(bars, METHODS):
            bar.set_hatch(HATCHES[method])

        ax.set_title(
            panel_label,
            pad=9,
            fontsize=13,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(
                facecolor="#EEF2F7",
                edgecolor="#CBD5E1",
                boxstyle="round,pad=0.25",
                linewidth=0.6,
            ),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=0, ha="center")
        ax.set_ylabel("Bias error", fontsize=11, color="#374151", fontweight="semibold")
        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            lo = 0.0
            hi = float(finite_values.max())
            pad = hi * 0.22 if hi else 1.0
            hi += pad
            ax.set_ylim(lo, hi)
        ax.grid(axis="y", color="#DDE3EC", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
        ax.tick_params(axis="x", length=0, pad=4, colors="#374151")
        ax.tick_params(axis="y", colors="#374151")

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ymin, ymax = ax.get_ylim()
            offset = 0.028 * (ymax - ymin)
            y = value + offset
            label = f"{value:.3f}" if value < 1 else f"{value:.2f}" if value < 10 else f"{value:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=9.3,
                fontweight="medium",
                color="#1F2937",
            )

    legend_handles = [
        Patch(
            facecolor=COLORS[m],
            edgecolor="#2F3437",
            hatch=HATCHES[m],
            linewidth=0.75,
            label=m.replace("\n", " "),
        )
        for m in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#B9C2CF",
        handlelength=1.45,
        columnspacing=1.2,
    )

    fig.suptitle(
        "Debiasing performance across benchmarks",
        y=0.992,
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )

    fig.text(
        0.01,
        0.043,
        "Metrics: CrowS stereotype preference, StereoSet SS overall, BBQ ambiguous/disambiguated bias score, "
        "WinoBias pro-anti gap, and WinoGender male-female disagreement.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.text(
        0.01,
        0.024,
        "Normalization: bars show distance from the unbiased target, 50 for CrowS/StereoSet and 0 for BBQ/Wino metrics; "
        "lower is better. ICL and DIY settings use all bias-reducing interventions.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.865), w_pad=1.45, h_pad=1.35)
    fig.savefig(OUTDIR / "pdf/debiasing_method_bars.pdf", bbox_inches="tight")


def main() -> None:
    rows = load_rows()
    matrix, audit = build_table(rows)
    write_csv(audit)
    plot(matrix)


if __name__ == "__main__":
    main()
