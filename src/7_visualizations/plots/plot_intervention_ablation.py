#!/usr/bin/env python3
"""Grouped bar chart for intervention-strategy ablations.

The plot compares five bias-reducing intervention strategies within each of
the three DIY methods. Each bar is the absolute reduction in mean normalized
bias error relative to the corresponding base-model score. The figure uses one
subplot per benchmark metric.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.patches import Patch

import plot_debiasing_method_bars as core


ROOT = core.ROOT
OUTDIR = core.OUTDIR

INTERVENTIONS = [
    ("stereotype_replacement", "stereotype-replacement", "Stereotype replacement"),
    ("individuating", "individuating", "Individuation"),
    ("perspective_taking", "perspective-taking", "Perspective-taking"),
    ("counter_imaging", "counter-imaging", "Counter-stereotypic imaging"),
    ("positive_contact", "positive-contact", "Positive contact"),
]

METHODS = [
    "ICL",
    "DIY IT",
    "DIY Two Pass (No IT)",
    "DIY Two Pass (IT)",
]

PLOT_LABELS = {
    "ICL": "ICL",
    "DIY IT": "DIY IT",
    "DIY Two Pass (No IT)": "DIY Two Pass\n(No IT)",
    "DIY Two Pass (IT)": "DIY Two Pass\n(IT)",
}

ICL_TAGS = {
    "stereotype_replacement": "sr",
    "individuating": "ind",
    "perspective_taking": "pt",
    "counter_imaging": "ci",
    "positive_contact": "pc",
}

INTERVENTION_STYLES = {
    "Stereotype replacement": ("#7DD3FC", "///"),
    "Individuation": ("#5EEAD4", "\\\\\\"),
    "Perspective-taking": ("#A7F3D0", "xxx"),
    "Counter-stereotypic imaging": ("#FDBA74", "..."),
    "Positive contact": ("#C4B5FD", "---"),
}

SINGLE_INTERVENTION_RUN_PRIORITY = [
    "m6sd_20260323_091321_twopass_zero_fix",
    "m6sd_20260317_main3",
    "pilot_fix2_20260323_gpu",
    "pilot_fix_20260323_gpu",
]


def flat_label(label: str) -> str:
    return label.replace("-\n", "-").replace("\n", " ")


def has_bbq_overall(row: dict[str, str]) -> bool:
    if row["dataset_key"] != "bbq":
        return True
    path = ROOT / row["source_file"]
    if not path.exists():
        return local_bbq_prediction_dir(row) is not None
    with path.open(newline="") as f:
        if any(metric_row.get("input_file") == "__overall__" for metric_row in csv.DictReader(f)):
            return True
    return local_bbq_prediction_dir(row) is not None


def local_bbq_prediction_dir(row: dict[str, str]) -> Path | None:
    if row["dataset_key"] != "bbq":
        return None
    source = row["source_file"]
    suffix = "/bbq/bbq_metrics.csv"
    for prefix in (
        "results/new_results/",
        "results/new_results_curated/m6_self_debiasing/correct/",
        "results/new_results_curated/m6_self_debiasing/incorrect/",
    ):
        if source.startswith(prefix) and source.endswith(suffix):
            rel = source[len(prefix) : -len("/bbq_metrics.csv")]
            if prefix.startswith("results/new_results_curated/"):
                rel = "m6_self_debiasing/" + rel
            pred_dir = ROOT / "outputs/new_outputs" / rel
            if pred_dir.is_dir() and glob.glob(str(pred_dir / "bbq_preds*")):
                return pred_dir
    return None


def recompute_bbq_metrics_from_all_predictions(pred_dir: Path) -> dict[str, float]:
    targets = core.load_bbq_target_locations()
    eval_rows = []
    seen_categories: set[str] = set()

    for pred_path in sorted(glob.glob(str(pred_dir / "bbq_preds*"))):
        with open(pred_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        categories = {
            (row.get("source_file") or row.get("category") or "")
            .replace(".jsonl", "")
            .strip()
            for row in rows
        }
        categories.discard("")
        if not categories or categories.issubset(seen_categories):
            continue
        seen_categories.update(categories)

        for row in rows:
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
                    "unknown": core.bbq_pred_category(row) == "unknown",
                }
            )

    if not eval_rows:
        raise RuntimeError(f"No valid BBQ prediction rows found in {pred_dir}")

    ambig_rows = [r for r in eval_rows if r["context_condition"] == "ambig"]
    disambig_rows = [r for r in eval_rows if r["context_condition"] == "disambig"]
    acc_ambig = sum(bool(r["correct"]) for r in ambig_rows) / len(ambig_rows)
    ambig_bias = core.context_bias_score(ambig_rows)
    disambig_bias = core.context_bias_score(disambig_rows)
    return {
        "Bias_score_ambig": round((1.0 - acc_ambig) * ambig_bias * 100.0, 3),
        "Bias_score_disambig": round(disambig_bias * 100.0, 3),
    }


def single_intervention_rank(row: dict[str, str]) -> tuple[int, str]:
    source = row["source_file"]
    for i, run in enumerate(SINGLE_INTERVENTION_RUN_PRIORITY):
        if run in source:
            return i, source
    return len(SINGLE_INTERVENTION_RUN_PRIORITY), source


def curated_metric_path(source_file: str) -> Path | None:
    prefix = "results/new_results/m6_self_debiasing/"
    curated_prefix = "results/new_results_curated/m6_self_debiasing/incorrect/"
    if not source_file.startswith(prefix):
        return None
    candidate = ROOT / (curated_prefix + source_file[len(prefix) :])
    return candidate if candidate.exists() else None


def read_metric_from_path(path: Path, dataset_key: str, model: str, metric: str) -> float:
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
            if r.get("input_file") == "__overall__" or r.get("Model") == model
        ]

    if not metric_rows:
        raise RuntimeError(f"No metric rows found in {path}")
    if metric not in metric_rows[-1]:
        raise KeyError(f"{metric} missing from {path}")
    return float(metric_rows[-1][metric])


def metric_value(row: dict[str, str], metric: str) -> tuple[float, str]:
    if not (ROOT / row["source_file"]).exists():
        alternate = curated_metric_path(row["source_file"])
        if alternate is not None:
            return (
                read_metric_from_path(alternate, row["dataset_key"], row["model"], metric),
                str(alternate.relative_to(ROOT)),
            )
    try:
        return core.metric_value(row, metric)
    except (FileNotFoundError, KeyError, RuntimeError):
        alternate = curated_metric_path(row["source_file"])
        if alternate is not None:
            return (
                read_metric_from_path(alternate, row["dataset_key"], row["model"], metric),
                str(alternate.relative_to(ROOT)),
            )
        raise


def method_row(
    rows: list[dict[str, str]],
    dataset_key: str,
    intervention_key: str,
    intervention_slug: str,
    method: str,
) -> dict[str, str] | None:
    dataset_rows = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key
        and r["type"] == "method"
        and r["strategy"] == intervention_key
        and core.valid_score(r)
    ]

    if method == "DIY IT":
        candidates = [
            r
            for r in dataset_rows
            if r["name"] == "m3_finetune_llama_ms500"
        ]
        return sorted(candidates, key=lambda r: (r["source_file"], r["model"]))[0] if candidates else None

    if method == "DIY Two Pass (No IT)":
        prefix = f"m6_two_pass__base__{intervention_key}"
        candidates = [
            r
            for r in dataset_rows
            if r["name"] == "m6_self_debiasing"
            and r["model"].startswith(prefix)
            and "two_pass_one" not in r["model"]
            and "two_pass_two" not in r["model"]
            and "two_pass_five" not in r["model"]
            and "same_thread" not in r["model"]
            and "noncog" not in r["model"]
            and has_bbq_overall(r)
        ]
        return sorted(candidates, key=single_intervention_rank)[0] if candidates else None

    if method == "DIY Two Pass (IT)":
        prefix = (
            f"m6_two_pass__finetuned_ms-500-{intervention_slug}"
            f"-opinion-action-event-allversions__{intervention_key}"
        )
        candidates = [
            r
            for r in dataset_rows
            if r["name"] == "m6_self_debiasing"
            and r["model"].startswith(prefix)
            and "two_pass_one" not in r["model"]
            and "two_pass_two" not in r["model"]
            and "two_pass_five" not in r["model"]
            and "same_thread" not in r["model"]
            and "noncog" not in r["model"]
            and has_bbq_overall(r)
        ]
        return sorted(candidates, key=single_intervention_rank)[0] if candidates else None

    raise ValueError(f"Unknown method: {method}")


def add_record(
    records: list[dict[str, str]],
    *,
    method: str,
    intervention_key: str,
    intervention_label: str,
    dataset_key: str,
    panel_label: str,
    metric: str,
    metric_label: str,
    ideal: float,
    row: dict[str, str],
) -> None:
    metric_source = ""
    try:
        if row["dataset_key"] == "bbq" and row["name"] == "m6_self_debiasing":
            path = ROOT / row["source_file"]
            if path.exists():
                with path.open(newline="") as f:
                    has_overall = any(
                        metric_row.get("input_file") == "__overall__"
                        for metric_row in csv.DictReader(f)
                    )
            else:
                has_overall = False
            if not has_overall:
                pred_dir = local_bbq_prediction_dir(row)
                if pred_dir is None:
                    raise RuntimeError(f"No overall BBQ row or prediction dir for {row['source_file']}")
                native_value = recompute_bbq_metrics_from_all_predictions(pred_dir)[metric]
                metric_source = f"recomputed_from_predictions:{pred_dir.relative_to(ROOT)}"
            else:
                native_value, metric_source = metric_value(row, metric)
        else:
            native_value, metric_source = metric_value(row, metric)
    except (FileNotFoundError, KeyError, RuntimeError):
        if row["dataset_key"] == "winobias" and row["score_label"] == "abs_pro_anti_gap":
            native_value = float(row["score"])
            metric_source = f"aggregate_csv:{row['source_file']}"
        else:
            raise
    error = core.normalized_bias_error(native_value, ideal)
    records.append(
        {
            "method": method,
            "intervention_key": intervention_key,
            "intervention_label": intervention_label,
            "dataset_key": dataset_key,
            "panel_label": panel_label,
            "metric": metric,
            "metric_label": metric_label,
            "ideal_value": f"{ideal:.6g}",
            "native_value": f"{native_value:.6g}",
            "normalized_bias_error": f"{error:.6g}",
            "base_normalized_bias_error": "",
            "bias_error_reduction_vs_base": "",
            "aggregate_score": row["score"],
            "aggregate_score_label": row["score_label"],
            "aggregate_direction": row["direction"],
            "name": row["name"],
            "model": row["model"],
            "strategy": row["strategy"],
            "metric_source_file": metric_source,
        }
    )


def icl_strategy_path(dataset_key: str, intervention_key: str) -> Path | None:
    tag = ICL_TAGS[intervention_key]
    root = core.ICL_ROOT
    if dataset_key == "bbq":
        return root / f"bbq/bbq_metrics_m4_baseicl_zero_llama8b_{tag}_bbq.csv"
    if dataset_key == "crowspairs":
        return root / f"evalshared/crows_pairs_metrics_overall_m4_baseicl_zero_llama8b_{tag}_crowspairs.csv"
    if dataset_key == "stereoset":
        return root / f"evalshared/stereoset_metrics_m4_baseicl_zero_llama8b_{tag}_stereoset.csv"
    if dataset_key == "winobias":
        return (
            root
            / f"evalshared/winobias/m4_baseicl_zero_llama8b_{tag}_winobias/"
            / f"winobias_metrics_overall_m4_baseicl_zero_llama8b_{tag}_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            root
            / f"evalshared/winogender/m4_baseicl_zero_llama8b_{tag}_winogender/"
            / f"winogender_metrics_overall_m4_baseicl_zero_llama8b_{tag}_winogender.csv"
        )
    return None


def add_icl_record(
    records: list[dict[str, str]],
    *,
    intervention_key: str,
    intervention_label: str,
    dataset_key: str,
    panel_label: str,
    metric: str,
    metric_label: str,
    ideal: float,
) -> None:
    path = icl_strategy_path(dataset_key, intervention_key)
    if path is None or not path.exists():
        return
    native_value = core.read_metric_path(path, dataset_key, metric)
    error = core.normalized_bias_error(native_value, ideal)
    records.append(
        {
            "method": "ICL",
            "intervention_key": intervention_key,
            "intervention_label": intervention_label,
            "dataset_key": dataset_key,
            "panel_label": panel_label,
            "metric": metric,
            "metric_label": metric_label,
            "ideal_value": f"{ideal:.6g}",
            "native_value": f"{native_value:.6g}",
            "normalized_bias_error": f"{error:.6g}",
            "base_normalized_bias_error": "",
            "bias_error_reduction_vs_base": "",
            "aggregate_score": "",
            "aggregate_score_label": "",
            "aggregate_direction": "lower",
            "name": "m4_base_icl_zero",
            "model": f"m4_baseicl_zero_llama8b_{ICL_TAGS[intervention_key]}",
            "strategy": intervention_key,
            "metric_source_file": str(path.relative_to(ROOT)),
        }
    )


def collect_panel_records() -> list[dict[str, str]]:
    rows = core.load_rows()
    records: list[dict[str, str]] = []

    for dataset_key, panel_label, metric, metric_label, ideal in core.PANELS:
        base = core.base_row(rows, dataset_key)
        add_record(
            records,
            method="Base Model",
            intervention_key="__base__",
            intervention_label="Base Model",
            dataset_key=dataset_key,
            panel_label=panel_label,
            metric=metric,
            metric_label=metric_label,
            ideal=ideal,
            row=base,
        )

    for intervention_key, intervention_slug, intervention_label in INTERVENTIONS:
        label = flat_label(intervention_label)
        for method in METHODS:
            for dataset_key, panel_label, metric, metric_label, ideal in core.PANELS:
                if method == "ICL":
                    add_icl_record(
                        records,
                        intervention_key=intervention_key,
                        intervention_label=label,
                        dataset_key=dataset_key,
                        panel_label=panel_label,
                        metric=metric,
                        metric_label=metric_label,
                        ideal=ideal,
                    )
                    continue
                row = method_row(
                    rows,
                    dataset_key,
                    intervention_key,
                    intervention_slug,
                    method,
                )
                if row is None:
                    continue
                add_record(
                    records,
                    method=method,
                    intervention_key=intervention_key,
                    intervention_label=label,
                    dataset_key=dataset_key,
                    panel_label=panel_label,
                    metric=metric,
                    metric_label=metric_label,
                    ideal=ideal,
                    row=row,
                )

    return records


def summarize(records: list[dict[str, str]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    groups = [("Base Model", "__base__", "Base Model")]
    groups.extend(
        (method, intervention_key, flat_label(label))
        for intervention_key, _, label in INTERVENTIONS
        for method in METHODS
    )

    for method, intervention_key, intervention_label in groups:
        subset = [
            r
            for r in records
            if r["method"] == method and r["intervention_key"] == intervention_key
        ]
        if not subset:
            continue
        values = [float(r["normalized_bias_error"]) for r in subset]
        summary.append(
            {
                "method": method,
                "intervention_key": intervention_key,
                "intervention_label": intervention_label,
                "dataset_key": "__mean__",
                "panel_label": "Mean over six bias metrics",
                "metric": "mean_normalized_bias_error",
                "metric_label": "Mean normalized bias error",
                "ideal_value": "",
                "native_value": "",
                "normalized_bias_error": f"{float(np.mean(values)):.6g}",
                "base_normalized_bias_error": "",
                "bias_error_reduction_vs_base": "",
                "aggregate_score": "",
                "aggregate_score_label": "",
                "aggregate_direction": "lower",
                "name": "summary",
                "model": "",
                "strategy": intervention_key,
                "metric_source_file": "computed_from_panel_rows",
            }
        )

    return summary


def write_csv(records: list[dict[str, str]], summary: list[dict[str, str]]) -> None:
    values = summary_lookup(summary)
    base_value = values[("Base Model", "__base__")]
    panel_base_values = {
        (row["dataset_key"], row["panel_label"], row["metric"]): float(
            row["normalized_bias_error"]
        )
        for row in records
        if row["method"] == "Base Model"
    }

    enriched_records = []
    for row in records:
        row = dict(row)
        panel_key = (row["dataset_key"], row["panel_label"], row["metric"])
        panel_base = panel_base_values.get(panel_key)
        if panel_base is None:
            row["base_normalized_bias_error"] = ""
            row["bias_error_reduction_vs_base"] = ""
        else:
            error = float(row["normalized_bias_error"])
            row["base_normalized_bias_error"] = f"{panel_base:.6g}"
            row["bias_error_reduction_vs_base"] = f"{panel_base - error:.6g}"
        enriched_records.append(row)

    enriched_summary = []
    for row in summary:
        row = dict(row)
        error = float(row["normalized_bias_error"])
        row["base_normalized_bias_error"] = f"{base_value:.6g}"
        row["bias_error_reduction_vs_base"] = f"{base_value - error:.6g}"
        enriched_summary.append(row)

    (OUTDIR / "csv").mkdir(parents=True, exist_ok=True)
    out = enriched_records + enriched_summary
    path = OUTDIR / "csv/intervention_ablation_data.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)


def summary_lookup(summary: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    return {
        (row["method"], row["intervention_key"]): float(row["normalized_bias_error"])
        for row in summary
    }


def panel_lookup(records: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    return {
        (row["method"], row["intervention_key"], row["panel_label"]): float(
            row["normalized_bias_error"]
        )
        for row in records
    }


def plot(records: list[dict[str, str]]) -> None:
    values = panel_lookup(records)

    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.8,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.1,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.9), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    x = np.arange(len(METHODS))
    width = 0.13
    offsets = np.linspace(-2 * width, 2 * width, len(INTERVENTIONS))

    for ax, (_, panel_label, _, metric_label, _) in zip(axes, core.PANELS):
        ax.set_facecolor("#FCFCFD")
        base_error = values[("Base Model", "__base__", panel_label)]
        panel_max = 0.0
        panel_min = 0.0

        for offset, (intervention_key, _, intervention_label) in zip(
            offsets, INTERVENTIONS
        ):
            color, hatch = INTERVENTION_STYLES[intervention_label]
            bar_values = [
                base_error - values[(method, intervention_key, panel_label)]
                if (method, intervention_key, panel_label) in values
                else np.nan
                for method in METHODS
            ]
            finite_values = [v for v in bar_values if np.isfinite(v)]
            if finite_values:
                panel_max = max(panel_max, max(finite_values))
                panel_min = min(panel_min, min(finite_values))

            bars = ax.bar(
                x + offset,
                bar_values,
                width=width,
                color=color,
                edgecolor="#111827",
                linewidth=0.75,
                hatch=hatch,
                zorder=3,
            )
        ax.set_title(panel_label, pad=8, fontweight="bold", color="#111827")
        ax.set_xticks(x)
        ax.set_xticklabels([PLOT_LABELS[method] for method in METHODS])
        ymin = min(0.0, panel_min)
        ymax = panel_max
        pad = 0.16 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - pad * 0.25, ymax + pad)
        ax.grid(axis="y", color="#E5E7EB", linestyle=":", linewidth=0.75, alpha=0.95)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="x", length=0, pad=5, colors="#374151")
        ax.tick_params(axis="y", colors="#374151")

    handles = [
        Patch(
            facecolor=color,
            edgecolor="#111827",
            hatch=hatch,
            linewidth=0.9,
            label=intervention_label,
        )
        for _, _, intervention_label in INTERVENTIONS
        for color, hatch in [INTERVENTION_STYLES[intervention_label]]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#B9C2CF",
        borderpad=0.45,
        columnspacing=1.0,
        handlelength=1.6,
    )

    fig.text(
        0.5,
        0.985,
        "Bias error reduction by intervention strategy",
        ha="center",
        va="top",
        fontsize=16.5,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.012,
        0.028,
        "Each subplot uses the benchmark-native metric normalized as bias error. "
        "Bars show absolute reduction from the base model on that metric; higher is better.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#374151",
    )

    fig.supxlabel("DIY method", y=0.07, fontsize=11, fontweight="semibold")
    fig.supylabel(
        "Bias error reduction",
        x=0.028,
        fontsize=11,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.78, wspace=0.24, hspace=0.42)
    (OUTDIR / "pdf").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "pdf/intervention_ablation.pdf", bbox_inches="tight")


def main() -> None:
    records = collect_panel_records()
    if not records:
        raise RuntimeError("No intervention-strategy records collected.")
    summary = summarize(records)
    write_csv(records, summary)
    plot(records)


if __name__ == "__main__":
    main()
