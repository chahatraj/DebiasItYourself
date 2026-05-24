#!/usr/bin/env python3
"""Shot-expanded version of the DIY debiasing-performance bar chart.

This figure keeps the same benchmark-native metrics and bias-error
normalization as plot_debiasing_method_bars.py, but splits the two-pass DIY
settings by the number of demonstrations used in the self-debiasing prompt.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np

import plot_debiasing_method_bars as core


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "figures" / "llama8b"
ICL_ONE_ROOT = (
    ROOT
    / "results/new_results/m4_base_icl_one/m4_base_icl_one_allmodels_20260514_124333/llama8b"
)
FT_ICL_ZERO_ROOT = (
    ROOT
    / "results/new_results/m4_ft_icl_zero/m4_ft_icl_zero_allmodels_20260514_143756/llama8b"
)
FT_ICL_ONE_ROOT = (
    ROOT
    / "results/new_results/m4_ft_icl_one/m4_ft_icl_one_allmodels_20260514_143757/llama8b"
)
M4_STRATEGY_TAGS = ("sr", "ci", "ind", "pt", "pc")

IT_MODEL_SUFFIX = (
    "finetuned_ms-500-allstrategies-opinion-action-event-allversions"
    "__all_strategies"
)

METHODS = [
    {
        "kind": "base",
        "setting": "Base",
        "shot": "",
        "label": "Base",
        "audit_label": "Base Model Inference",
        "color": "#BBC2CE",
        "hatch": "",
    },
    {
        "kind": "icl",
        "setting": "ICL",
        "shot": 0,
        "label": "ICL\n0",
        "audit_label": "DIY-Show 0-shot",
        "color": "#C4B5FD",
        "hatch": "xx",
    },
    {
        "kind": "icl",
        "setting": "ICL",
        "shot": 1,
        "label": "ICL\n1",
        "audit_label": "DIY-Show 1-shot",
        "color": "#A78BFA",
        "hatch": "xx",
    },
    {
        "kind": "it",
        "setting": "DIY IT",
        "shot": "",
        "label": "DIY\nIT",
        "audit_label": "DIY IT",
        "color": "#7CC7F2",
        "hatch": "///",
    },
    {
        "kind": "two_pass_no_it",
        "setting": "No IT",
        "shot": 0,
        "label": "No IT\n0",
        "audit_label": "DIY Two Pass (No IT), 0-shot",
        "color": "#B6EAC7",
        "hatch": "",
    },
    {
        "kind": "two_pass_no_it",
        "setting": "No IT",
        "shot": 1,
        "label": "No IT\n1",
        "audit_label": "DIY Two Pass (No IT), 1-shot",
        "color": "#84D9A5",
        "hatch": "\\\\\\",
    },
    {
        "kind": "two_pass_no_it",
        "setting": "No IT",
        "shot": 2,
        "label": "No IT\n2",
        "audit_label": "DIY Two Pass (No IT), 2-shot",
        "color": "#50C77B",
        "hatch": "...",
    },
    {
        "kind": "two_pass_it",
        "setting": "IT",
        "shot": 0,
        "label": "IT\n0",
        "audit_label": "DIY Two Pass (IT), 0-shot",
        "color": "#FFD8A7",
        "hatch": "",
    },
    {
        "kind": "two_pass_it",
        "setting": "IT",
        "shot": 1,
        "label": "IT\n1",
        "audit_label": "DIY Two Pass (IT), 1-shot",
        "color": "#FFB86B",
        "hatch": "///",
    },
    {
        "kind": "two_pass_it",
        "setting": "IT",
        "shot": 2,
        "label": "IT\n2",
        "audit_label": "DIY Two Pass (IT), 2-shot",
        "color": "#FF9E45",
        "hatch": "...",
    },
    {
        "kind": "ft_icl",
        "setting": "IT+ICL",
        "shot": 0,
        "label": "IT+ICL\n0",
        "audit_label": "DIY-Teach-Show 0-shot",
        "color": "#FCD34D",
        "hatch": "++",
    },
    {
        "kind": "ft_icl",
        "setting": "IT+ICL",
        "shot": 1,
        "label": "IT+ICL\n1",
        "audit_label": "DIY-Teach-Show 1-shot",
        "color": "#FBBF24",
        "hatch": "++",
    },
]

SHOT_RUN_PRIORITIES = {
    "crowspairs": {
        0: ["m6sd_20260324_twopass_zero_allstrategies_truncfix"],
        1: [
            "m6sd_20260324_twopass_one_allstrategies_truncfix",
            "m6sd_20260319_shot125_twopass_sep",
        ],
        2: ["m6sd_20260326_twoshot_crows_stereoset_fix"],
    },
    "stereoset": {
        0: ["m6sd_20260324_twopass_zero_allstrategies_truncfix"],
        1: [
            "m6sd_20260324_twopass_one_allstrategies_truncfix",
            "m6sd_20260319_shot125_twopass_sep",
        ],
        2: ["m6sd_20260326_twoshot_crows_stereoset_fix"],
    },
    "bbq": {
        0: ["m6sd_20260325_bbq_allstrategies_maxlenfix_zero"],
        1: ["m6sd_20260325_bbq_allstrategies_maxlenfix_one"],
        2: ["m6sd_20260330_twoshot_bbq_allstrategies_fix"],
    },
    "winobias": {
        0: ["m6sd_20260324_twopass_zero_allstrategies_truncfix"],
        1: [
            "m6sd_20260324_twopass_one_allstrategies_truncfix",
            "m6sd_20260319_shot125_twopass_sep",
        ],
        2: ["m6sd_20260326_twoshot_wino_allstrategies_fix"],
    },
    "winogender": {
        0: ["m6sd_20260324_twopass_zero_allstrategies_truncfix"],
        1: [
            "m6sd_20260324_twopass_one_allstrategies_truncfix",
            "m6sd_20260319_shot125_twopass_sep",
        ],
        2: ["m6sd_20260326_twoshot_wino_allstrategies_fix"],
    },
}


def exact_m3_row(rows: list[dict[str, str]], dataset_key: str) -> dict[str, str] | None:
    candidates = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key
        and r["type"] == "method"
        and r["name"] == "m3_finetune_llama_ms500"
        and core.valid_score(r)
        and core.exact_m3_model(dataset_key).search(r["model"])
    ]
    return candidates[0] if candidates else None


def shot_model_prefix(kind: str, shot: int) -> str:
    shot_prefix = {
        0: "m6_two_pass__",
        1: "m6_two_pass_one__",
        2: "m6_two_pass_two__",
    }[shot]
    if kind == "two_pass_no_it":
        return f"{shot_prefix}base__all_strategies"
    if kind == "two_pass_it":
        return f"{shot_prefix}{IT_MODEL_SUFFIX}"
    raise ValueError(f"Unsupported shot method kind: {kind}")


def metric_source_available(row: dict[str, str]) -> bool:
    source = ROOT / row["source_file"]
    if source.exists():
        return True
    return (
        row["dataset_key"] == "bbq"
        and core.archived_bbq_prediction_dir(row["source_file"]) is not None
    )


def shot_run_rank(row: dict[str, str], dataset_key: str, shot: int) -> tuple[int, int, str]:
    source = row["source_file"]
    priorities = SHOT_RUN_PRIORITIES[dataset_key][shot]
    for idx, run in enumerate(priorities):
        if run in source:
            priority = idx
            break
    else:
        priority = len(priorities)

    availability_penalty = 0 if metric_source_available(row) else 100
    return priority, availability_penalty, source


def shot_method_row(
    rows: list[dict[str, str]], dataset_key: str, kind: str, shot: int
) -> dict[str, str] | None:
    prefix = shot_model_prefix(kind, shot)
    candidates = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key
        and r["type"] == "method"
        and r["name"] == "m6_self_debiasing"
        and core.valid_score(r)
        and r["model"].startswith(prefix)
        and "same_thread" not in r["model"]
        and "noncog" not in r["model"]
        and r["strategy"] == "all_strategies"
    ]
    return sorted(candidates, key=lambda r: shot_run_rank(r, dataset_key, shot))[0] if candidates else None


def icl_bias_path(dataset_key: str, shot: int) -> Path | None:
    root = ICL_ONE_ROOT if shot == 1 else core.ICL_ROOT
    shot_word = "one" if shot == 1 else "zero"
    prefix = f"m4_baseicl_{shot_word}_llama8b_allstrat"
    if dataset_key == "bbq":
        return root / f"bbq/bbq_metrics_{prefix}_bbq.csv"
    if dataset_key == "crowspairs":
        return root / f"evalshared/crows_pairs_metrics_overall_{prefix}_crowspairs.csv"
    if dataset_key == "stereoset":
        return root / f"evalshared/stereoset_metrics_{prefix}_stereoset.csv"
    if dataset_key == "winobias":
        return root / f"evalshared/winobias/{prefix}_winobias/winobias_metrics_overall_{prefix}_winobias.csv"
    if dataset_key == "winogender":
        return root / f"evalshared/winogender/{prefix}_winogender/winogender_metrics_overall_{prefix}_winogender.csv"
    return None


def ft_icl_bias_path(dataset_key: str, shot: int) -> Path | None:
    root = FT_ICL_ONE_ROOT if shot == 1 else FT_ICL_ZERO_ROOT
    shot_word = "one" if shot == 1 else "zero"
    prefix = f"m4_fticl_{shot_word}_llama8b_all_allversions"
    if dataset_key == "bbq":
        return root / f"bbq/bbq_metrics_{prefix}_bbq.csv"
    if dataset_key == "crowspairs":
        return root / f"evalshared/crows_pairs_metrics_overall_{prefix}_crowspairs.csv"
    if dataset_key == "stereoset":
        return root / f"evalshared/stereoset_metrics_{prefix}_stereoset.csv"
    if dataset_key == "winobias":
        return root / f"evalshared/winobias/{prefix}_winobias/winobias_metrics_overall_{prefix}_winobias.csv"
    if dataset_key == "winogender":
        return root / f"evalshared/winogender/{prefix}_winogender/winogender_metrics_overall_{prefix}_winogender.csv"
    return None


def mean_strategy_bbq_metric(kind: str, shot: int, metric: str) -> tuple[float, str] | None:
    if shot != 1 or kind not in {"icl", "ft_icl"}:
        return None
    root = ICL_ONE_ROOT if kind == "icl" else FT_ICL_ONE_ROOT
    prefix_base = "m4_baseicl_one_llama8b" if kind == "icl" else "m4_fticl_one_llama8b"
    values = []
    sources = []
    for tag in M4_STRATEGY_TAGS:
        path = root / "bbq" / f"bbq_metrics_{prefix_base}_{tag}_bbq.csv"
        if not path.exists():
            continue
        values.append(core.read_metric_path(path, "bbq", metric))
        sources.append(str(path.relative_to(ROOT)))
    if not values:
        return None
    return float(np.mean(values)), f"mean_strategy_metrics:{'|'.join(sources)}"


def selected_row(
    rows: list[dict[str, str]], dataset_key: str, method: dict[str, object]
) -> dict[str, str] | None:
    kind = str(method["kind"])
    if kind == "base":
        return core.base_row(rows, dataset_key)
    if kind == "it":
        return exact_m3_row(rows, dataset_key)
    if kind in {"icl", "ft_icl"}:
        return None
    return shot_method_row(rows, dataset_key, kind, int(method["shot"]))


def build_table(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    matrix = np.full((len(core.PANELS), len(METHODS)), np.nan)
    audit: list[dict[str, str]] = []

    for i, (dataset_key, panel_label, metric, metric_label, ideal) in enumerate(core.PANELS):
        for j, method in enumerate(METHODS):
            kind = str(method["kind"])
            if kind in {"icl", "ft_icl"}:
                shot = int(method["shot"])
                path = icl_bias_path(dataset_key, shot) if kind == "icl" else ft_icl_bias_path(dataset_key, shot)
                derived = mean_strategy_bbq_metric(kind, shot, metric) if dataset_key == "bbq" else None
                if derived is not None:
                    native_value, metric_source = derived
                else:
                    if path is None or not path.exists():
                        continue
                    native_value = core.read_metric_path(path, dataset_key, metric)
                    metric_source = str(path.relative_to(ROOT))
                plotted_value = core.normalized_bias_error(native_value, ideal)
                matrix[i, j] = plotted_value
                audit.append(
                    {
                        "dataset_key": dataset_key,
                        "panel_label": panel_label,
                        "method": str(method["audit_label"]),
                        "setting": str(method["setting"]),
                        "shot": str(method["shot"]),
                        "metric": metric,
                        "metric_label": metric_label,
                        "ideal_value": f"{ideal:.6g}",
                        "native_value": f"{native_value:.6g}",
                        "normalized_bias_error_plotted": f"{plotted_value:.6g}",
                        "aggregate_score": "",
                        "aggregate_score_label": "",
                        "aggregate_direction": "lower",
                        "name": "m4_base_icl" if kind == "icl" else "m4_ft_icl",
                        "model": path.stem,
                        "strategy": "all_strategies",
                        "metric_source_file": metric_source,
                    }
                )
                continue
            row = selected_row(rows, dataset_key, method)
            if row is None:
                continue
            native_value, metric_source = core.metric_value(row, metric)
            plotted_value = core.normalized_bias_error(native_value, ideal)
            matrix[i, j] = plotted_value
            audit.append(
                {
                    "dataset_key": dataset_key,
                    "panel_label": panel_label,
                    "method": str(method["audit_label"]),
                    "setting": str(method["setting"]),
                    "shot": str(method["shot"]),
                    "metric": metric,
                    "metric_label": metric_label,
                    "ideal_value": f"{ideal:.6g}",
                    "native_value": f"{native_value:.6g}",
                    "normalized_bias_error_plotted": f"{plotted_value:.6g}",
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
    path = OUTDIR / "csv/debiasing_method_bars_by_shot_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        writer.writeheader()
        writer.writerows(audit)


def value_label(value: float) -> str:
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def plot(matrix: np.ndarray) -> None:
    use_nimbus_sans(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.5,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 6.9), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    x = np.arange(len(METHODS))
    labels = [str(m["label"]) for m in METHODS]
    colors = [str(m["color"]) for m in METHODS]
    hatches = [str(m["hatch"]) for m in METHODS]

    for i, (ax, (_, panel_label, _, _, _)) in enumerate(zip(axes, core.PANELS)):
        values = matrix[i]
        ax.set_facecolor("#FCFCFD")
        ax.axvspan(0.55, 2.45, color="#F3E8FF", alpha=0.58, zorder=0)
        ax.axvspan(3.55, 6.45, color="#ECF9F0", alpha=0.62, zorder=0)
        ax.axvspan(6.55, 9.45, color="#FFF3E4", alpha=0.72, zorder=0)
        ax.axvspan(9.55, 11.45, color="#FEF3C7", alpha=0.62, zorder=0)
        for xpos in (0.5, 2.5, 3.5, 6.5, 9.5):
            ax.axvline(xpos, color="#9AA4B2", linewidth=0.75, alpha=0.7)

        bars = ax.bar(
            x,
            values,
            width=0.68,
            color=colors,
            edgecolor="#252A2E",
            linewidth=0.95,
            zorder=3,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_title(
            panel_label,
            pad=9,
            fontsize=14,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(
                facecolor="#F1F5F9",
                edgecolor="#CBD5E1",
                boxstyle="round,pad=0.24",
                linewidth=0.7,
            ),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if i % 3 == 0:
            ax.set_ylabel("Bias error", fontsize=11.5, color="#374151", fontweight="semibold")

        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            ymax = float(finite_values.max())
            ax.set_ylim(0, ymax * 1.24 if ymax else 1.0)

        ax.grid(axis="y", color="#DCE2EA", linewidth=0.85, linestyle="--", alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
        ax.tick_params(axis="x", length=0, pad=5, colors="#374151")
        ax.tick_params(axis="y", colors="#374151")

        ymin, ymax = ax.get_ylim()
        offset = 0.028 * (ymax - ymin)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                value_label(float(value)),
                ha="center",
                va="bottom",
                fontsize=9.6,
                color="#1F2937",
            )

        ymax = ax.get_ylim()[1]
        group_labels = [
            (1.5, "Show", "#5B21B6"),
            (5.0, "Revise", "#17623A"),
            (8.0, "Teach-Revise", "#8A4A12"),
            (10.5, "Teach-Show", "#92400E"),
        ]
        for xpos, text, color in group_labels:
            ax.text(
                xpos,
                ymax * 0.975,
                text,
                ha="center",
                va="top",
                fontsize=9.6,
                color=color,
                fontweight="semibold",
            )

    fig.suptitle(
        "Debiasing performance by two-pass shot setting",
        y=0.978,
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.01,
        0.048,
        "Metrics: CrowS stereotype preference, StereoSet SS overall, BBQ ambiguous/disambiguated bias score, "
        "WinoBias pro-anti gap, and WinoGender male-female disagreement.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#374151",
    )
    fig.text(
        0.01,
        0.027,
        "Normalization: bars show distance from the unbiased target, 50 for CrowS/StereoSet and 0 for BBQ/Wino metrics; "
        "lower is better. Shot labels 0/1/2 refer to demonstrations in the two-pass self-debiasing prompt.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.91), w_pad=1.25, h_pad=1.35)
    fig.savefig(OUTDIR / "pdf/debiasing_method_bars_by_shot.pdf", bbox_inches="tight")


def main() -> None:
    rows = core.load_rows()
    matrix, audit = build_table(rows)
    write_csv(audit)
    plot(matrix)


if __name__ == "__main__":
    main()
