#!/usr/bin/env python3
"""Baseline comparison plot for DIY debiasing results.

This figure compares DIY against existing mitigation baselines using the same
normalized bias-error metrics as the main debiasing-performance plot.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.lines import Line2D

import plot_debiasing_method_bars as core


ROOT = core.ROOT
INPUT = core.INPUT
OUTDIR = core.OUTDIR

PANELS = core.PANELS

BASELINE_METHODS = [
    ("bba", "BBA"),
    ("cal", "CAL"),
    ("fairsteer", "FairSteer"),
    ("biasedit", "BiasEdit"),
    ("lftf", "LFTF"),
    ("dpo", "DPO"),
    ("peft", "PEFT"),
    ("debias_llms", "DebiasLLMs"),
    ("debias_nlg", "DebiasNLG"),
    ("reduce_social_bias", "RSB"),
    ("self_debiasing_reprompting", "SelfDebias"),
]

METHOD_ORDER = [
    ("base", "Base Model", "base"),
    ("icl", "ICL", "icl"),
    *[(key, label, "baseline") for key, label in BASELINE_METHODS],
    ("diy_instruction_tune", "DIY IT", "diy_tune"),
    ("diy_twopass", "DIY Two Pass (No IT)", "diy_twopass"),
    ("diy_tune_twopass", "DIY Two Pass (IT)", "diy_combo"),
]

GROUP_STYLES = {
    "base": {"color": "#A9B7D9", "marker": "o", "size": 92, "zorder": 4},
    "icl": {"color": "#C4B5FD", "marker": "D", "size": 108, "zorder": 5},
    "baseline": {"color": "#BFD7FF", "marker": "o", "size": 78, "zorder": 3},
    "diy_tune": {"color": "#7DD3FC", "marker": "D", "size": 112, "zorder": 5},
    "diy_twopass": {"color": "#86EFAC", "marker": "D", "size": 112, "zorder": 5},
    "diy_combo": {"color": "#FDBA74", "marker": "D", "size": 126, "zorder": 6},
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def source_rank(row: dict[str, str]) -> tuple[int, int, int, str]:
    source = row["source_file"]
    model = row["model"].lower()
    source_exists = (ROOT / source).exists()
    recoverable_selfdebias_bbq = row["dataset_key"] == "bbq" and row["name"].startswith(
        "self_debiasing"
    )
    missing_penalty = 0 if source_exists or recoverable_selfdebias_bbq else 100
    family_penalty = 0 if "results/3_baselines/" in source else 20
    model_penalty = 10 if "70b" in model or "llama_70b" in source else 0
    return missing_penalty, family_penalty, model_penalty, source


def select_baseline_row(
    rows: list[dict[str, str]], dataset_key: str, method_key: str, metric: str
) -> dict[str, str] | None:
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

    valid: list[tuple[tuple[int, int, int, str], dict[str, str]]] = []
    for row in candidates:
        try:
            baseline_metric_value(row, metric)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError):
            continue
        valid.append((source_rank(row), row))
    if not valid:
        return None
    return sorted(valid, key=lambda item: item[0])[0][1]


def overall_or_category_metric(path: Path, row: dict[str, str], metric: str) -> tuple[float, str]:
    metric_rows = read_csv_rows(path)
    dataset_key = row["dataset_key"]

    if dataset_key == "stereoset":
        overall = [
            r
            for r in metric_rows
            if r.get("split") == "overall" and r.get("domain") == "overall"
        ]
        if not overall:
            raise RuntimeError(f"No StereoSet overall row in {path}")
        value = safe_float(overall[-1].get(metric))
        if value is None:
            raise KeyError(metric)
        return value, str(path.relative_to(ROOT))

    if dataset_key == "bbq":
        overall = [
            r
            for r in metric_rows
            if r.get("input_file") == "__overall__"
            or r.get("Category", "").lower() == "overall"
            or r.get("Model", "").lower() == "overall"
        ]
        if overall:
            value = safe_float(overall[-1].get(metric))
            if value is None:
                raise KeyError(metric)
            return value, str(path.relative_to(ROOT))

        # Some baseline BBQ files only contain per-category rows. For the
        # baseline comparison, aggregate as mean absolute category bias error.
        vals = [abs(v) for r in metric_rows if (v := safe_float(r.get(metric))) is not None]
        if not vals:
            raise KeyError(metric)
        return float(np.mean(vals)), f"mean_abs_category:{path.relative_to(ROOT)}"

    value = core.read_metric_file(row, metric)
    return value, row["source_file"]


def selfdebias_split_bbq_value(method_key: str, metric: str) -> tuple[float, str]:
    if metric == "Bias_score_disambig":
        # The disambiguated BBQ self-debiasing baseline is stored in the
        # separate disambig aggregate file for reprompting only.
        if method_key != "self_debiasing_reprompting":
            raise RuntimeError("No disambiguated explanation baseline file found")
        path = ROOT / "results/3_baselines/self_debiasing_bbq_disambig/bbq_eval_llama_8b_selfdebiasing_all.csv"
        rows = read_csv_rows(path)
        vals = [
            abs(v)
            for r in rows
            if r.get("Method") == "reprompting"
            and (v := safe_float(r.get("Bias_score_disambig"))) is not None
        ]
        if not vals:
            raise RuntimeError("No disambiguated reprompting BBQ rows found")
        return float(np.mean(vals)), f"mean_abs_category:{path.relative_to(ROOT)}"

    method = "explanation" if method_key.endswith("explanation") else "reprompting"
    split_dir = ROOT / "results/3_baselines/self_debiasing/bbq"
    vals: list[float] = []
    for path in sorted(split_dir.glob("bbq_eval_llama_8b_selfdebiasing_*.csv")):
        for row in read_csv_rows(path):
            if row.get("Method") != method:
                continue
            value = safe_float(row.get(metric))
            if value is not None:
                vals.append(abs(value))
    if not vals:
        raise RuntimeError(f"No BBQ split rows found for {method_key} / {metric}")
    return float(np.mean(vals)), f"mean_abs_category:{split_dir.relative_to(ROOT)}"


def baseline_metric_value(row: dict[str, str], metric: str) -> tuple[float, str]:
    path = ROOT / row["source_file"]
    if path.exists():
        return overall_or_category_metric(path, row, metric)
    if row["dataset_key"] == "bbq" and row["name"].startswith("self_debiasing"):
        return selfdebias_split_bbq_value(row["name"], metric)
    raise FileNotFoundError(path)


def row_metric_value(row: dict[str, str], metric: str) -> tuple[float, str]:
    if row["type"] == "baseline":
        return baseline_metric_value(row, metric)
    return core.metric_value(row, metric)


def collect_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for dataset_key, panel_label, metric, metric_label, ideal in PANELS:
        for method_key, method_label, group in METHOD_ORDER:
            row = None
            if method_key == "base":
                row = core.base_row(rows, dataset_key)
            elif method_key == "icl":
                path = core.icl_bias_path(dataset_key)
                if path is None or not path.exists():
                    continue
                try:
                    native_value = core.read_metric_path(path, dataset_key, metric)
                except (KeyError, RuntimeError, ValueError):
                    continue
                error = core.normalized_bias_error(native_value, ideal)
                records.append(
                    {
                        "dataset_key": dataset_key,
                        "panel_label": panel_label,
                        "method_key": method_key,
                        "method_label": method_label,
                        "group": group,
                        "metric": metric,
                        "metric_label": metric_label,
                        "ideal_value": f"{ideal:.6g}",
                        "native_value": f"{native_value:.6g}",
                        "normalized_bias_error_plotted": f"{error:.6g}",
                        "row_name": "m4_base_icl_zero",
                        "row_model": "m4_baseicl_zero_llama8b_allstrat",
                        "row_source_file": str(path.relative_to(ROOT)),
                        "metric_source_file": str(path.relative_to(ROOT)),
                    }
                )
                continue
            elif method_key == "diy_instruction_tune":
                row = core.canonical_method_row(
                    rows, dataset_key, "DIY IT"
                )
            elif method_key == "diy_twopass":
                row = core.canonical_method_row(
                    rows, dataset_key, "DIY Two Pass\n(No IT)"
                )
            elif method_key == "diy_tune_twopass":
                row = core.canonical_method_row(
                    rows, dataset_key, "DIY Two Pass\n(IT)"
                )
            else:
                row = select_baseline_row(rows, dataset_key, method_key, metric)
            if row is None:
                continue

            try:
                native_value, metric_source = row_metric_value(row, metric)
            except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                continue

            error = core.normalized_bias_error(native_value, ideal)
            group_for_plot = (
                group
            )
            records.append(
                {
                    "dataset_key": dataset_key,
                    "panel_label": panel_label,
                    "method_key": method_key,
                    "method_label": method_label,
                    "group": group_for_plot,
                    "metric": metric,
                    "metric_label": metric_label,
                    "ideal_value": f"{ideal:.6g}",
                    "native_value": f"{native_value:.6g}",
                    "normalized_bias_error_plotted": f"{error:.6g}",
                    "row_name": row["name"],
                    "row_model": row["model"],
                    "row_source_file": row["source_file"],
                    "metric_source_file": metric_source,
                }
            )

    return records


def write_csv(records: list[dict[str, str]]) -> None:
    path = OUTDIR / "csv/baseline_comparison_lollipop_data.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def records_by_panel(records: list[dict[str, str]], panel_label: str) -> dict[str, dict[str, str]]:
    return {r["method_key"]: r for r in records if r["panel_label"] == panel_label}


def plot(records: list[dict[str, str]]) -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 12.2), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    method_keys = [m[0] for m in METHOD_ORDER]
    method_labels = [m[1] for m in METHOD_ORDER]
    y_positions = np.arange(len(method_keys)) * 1.62

    for ax_idx, (ax, (_, panel_label, _, _, _)) in enumerate(zip(axes, PANELS)):
        panel_records = records_by_panel(records, panel_label)
        ax.set_facecolor("white")

        values = []
        for y, method_key in zip(y_positions, method_keys):
            rec = panel_records.get(method_key)
            if rec is None:
                values.append(np.nan)
                continue
            value = float(rec["normalized_bias_error_plotted"])
            values.append(value)
            style = GROUP_STYLES[rec["group"]]
            ax.hlines(y, 0, value, color="#A8B3C5", linewidth=2.05, zorder=1, alpha=0.95)
            ax.scatter(
                value,
                y,
                s=style["size"],
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="#17202A",
                linewidth=1.45,
                zorder=style["zorder"],
            )
        
        finite_for_label = [
            float(r["normalized_bias_error_plotted"]) for r in panel_records.values()
        ]
        xmax_for_label = max(finite_for_label) if finite_for_label else 1.0
        for y, method_key in zip(y_positions, method_keys):
            rec = panel_records.get(method_key)
            if rec is None:
                continue
            value = float(rec["normalized_bias_error_plotted"])
            label = f"{value:.2f}" if value >= 1 else f"{value:.3f}"
            color = "#1F2937" if not method_key.startswith("diy_") else "#7A4A13"
            weight = "medium" if not method_key.startswith("diy_") else "bold"
            label_offset = max(xmax_for_label * 0.04, ax.get_xlim()[1] * 0.018)
            ax.text(
                value + label_offset,
                y,
                label,
                ha="left",
                va="center",
                fontsize=9.4,
                fontweight=weight,
                color=color,
                clip_on=False,
            )

        finite_values = [v for v in values if np.isfinite(v)]
        xmax = max(finite_values) if finite_values else 1.0
        ax.set_xlim(0, xmax * 1.34 if xmax else 1.0)
        ax.set_title(
            panel_label,
            pad=8,
            fontsize=15,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(
                facecolor="#EEF2F7",
                edgecolor="#CBD5E1",
                boxstyle="round,pad=0.25",
                linewidth=0.6,
            ),
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A97A8")
        ax.spines["bottom"].set_color("#8A97A8")
        ax.spines["left"].set_linewidth(1.05)
        ax.spines["bottom"].set_linewidth(1.05)
        ax.tick_params(axis="x", colors="#374151")
        ax.tick_params(axis="y", length=0, colors="#374151")
        ax.set_xlabel("Bias error", fontsize=12.5, fontweight="semibold")

        ax.set_yticks(y_positions)
        if ax_idx % 3 == 0:
            ax.set_yticklabels(method_labels)
            for label in ax.get_yticklabels():
                label.set_fontweight("medium")
        else:
            ax.set_yticklabels([])
        ax.set_ylim(y_positions[-1] + 1.1, y_positions[0] - 1.1)

    handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GROUP_STYLES["icl"]["color"],
            markeredgecolor="#27313A",
            label="ICL",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLES["base"]["color"],
            markeredgecolor="#27313A",
            label="Base model",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLES["baseline"]["color"],
            markeredgecolor="#27313A",
            label="Baselines",
            markersize=7.5,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GROUP_STYLES["diy_tune"]["color"],
            markeredgecolor="#27313A",
            label="DIY instruction tuning",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GROUP_STYLES["diy_twopass"]["color"],
            markeredgecolor="#27313A",
            label="DIY two pass",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GROUP_STYLES["diy_combo"]["color"],
            markeredgecolor="#27313A",
            label="DIY tune + two pass",
            markersize=8.5,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=6,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#B9C2CF",
    )

    fig.suptitle(
        "DIY compared with existing debiasing baselines",
        y=0.99,
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.01,
        0.035,
        "Metrics and normalization match the debiasing-performance figure; lower bias error is better. "
        "Baseline rows prefer Llama-8B results when available; DIY uses all bias-reducing interventions.",
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0.105, 0.06, 1, 0.875), w_pad=1.35, h_pad=1.3)
    fig.savefig(OUTDIR / "pdf/baseline_comparison_lollipop.pdf", bbox_inches="tight")


def main() -> None:
    rows = core.load_rows()
    records = collect_records(rows)
    write_csv(records)
    plot(records)


if __name__ == "__main__":
    main()
