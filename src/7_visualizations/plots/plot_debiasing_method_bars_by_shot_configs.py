#!/usr/bin/env python3
"""Shot-expanded debiasing plots for strategy/checkpoint configurations.

Each output figure uses the same six benchmark panels and the same normalized
bias-error metrics as the main debiasing-method bar chart. The figures vary
the self-debiasing intervention strategy and, for instruction-tuned two-pass
models, the finetuned checkpoint family.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_debiasing_method_bars as core
import plot_intervention_ablation as intervention


ROOT = core.ROOT
OUTDIR = core.OUTDIR
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


@dataclass(frozen=True)
class PlotConfig:
    slug: str
    title: str
    strategy_key: str
    strategy_label: str
    ft_slug: str
    ft_label: str
    it_key: str


CONFIGS = [
    PlotConfig(
        "all_strategies_all_versions",
        "All interventions, all-version IT checkpoint",
        "all_strategies",
        "All interventions",
        "allstrategies-opinion-action-event-allversions",
        "all-version",
        "allversions",
    ),
    PlotConfig(
        "all_strategies_opinion",
        "All interventions, opinion IT checkpoint",
        "all_strategies",
        "All interventions",
        "allstrategies-opinion",
        "opinion",
        "opinion",
    ),
    PlotConfig(
        "all_strategies_action",
        "All interventions, action IT checkpoint",
        "all_strategies",
        "All interventions",
        "allstrategies-action",
        "action",
        "action",
    ),
    PlotConfig(
        "all_strategies_event",
        "All interventions, event IT checkpoint",
        "all_strategies",
        "All interventions",
        "allstrategies-event",
        "event",
        "event",
    ),
    PlotConfig(
        "stereotype_replacement_all_versions",
        "Stereotype replacement, matched IT checkpoint",
        "stereotype_replacement",
        "Stereotype replacement",
        "stereotype-replacement-opinion-action-event-allversions",
        "matched all-version",
        "stereotype_replacement",
    ),
    PlotConfig(
        "individuation_all_versions",
        "Individuation, matched IT checkpoint",
        "individuating",
        "Individuation",
        "individuating-opinion-action-event-allversions",
        "matched all-version",
        "individuating",
    ),
    PlotConfig(
        "perspective_taking_all_versions",
        "Perspective-taking, matched IT checkpoint",
        "perspective_taking",
        "Perspective-taking",
        "perspective-taking-opinion-action-event-allversions",
        "matched all-version",
        "perspective_taking",
    ),
    PlotConfig(
        "counter_stereotypic_imaging_all_versions",
        "Counter-stereotypic imaging, matched IT checkpoint",
        "counter_imaging",
        "Counter-stereotypic imaging",
        "counter-imaging-opinion-action-event-allversions",
        "matched all-version",
        "counter_imaging",
    ),
    PlotConfig(
        "positive_contact_all_versions",
        "Positive contact, matched IT checkpoint",
        "positive_contact",
        "Positive contact",
        "positive-contact-opinion-action-event-allversions",
        "matched all-version",
        "positive_contact",
    ),
]

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

SHOT_PREFIX = {
    0: "m6_two_pass__",
    1: "m6_two_pass_one__",
    2: "m6_two_pass_two__",
}

RUN_PRIORITIES = [
    "m6sd_20260330_twoshot_bbq_allstrategies_fix",
    "m6sd_20260326_twoshot_wino_allstrategies_fix",
    "m6sd_20260326_twoshot_crows_stereoset_fix",
    "m6sd_20260325_bbq_allstrategies_maxlenfix_one",
    "m6sd_20260325_bbq_allstrategies_maxlenfix_zero",
    "m6sd_20260324_twopass_one_allstrategies_truncfix",
    "m6sd_20260324_twopass_zero_allstrategies_truncfix",
    "m6sd_20260324_twopass_zero_missing_stereoset",
    "m6sd_20260323_091321_twopass_zero_fix",
    "m6sd_20260319_shot125_twopass_sep",
    "m6sd_20260317_main3",
    "pilot_fix2_20260323_gpu",
    "pilot_fix_20260323_gpu",
]


def shot_prefix(shot: int) -> str:
    return SHOT_PREFIX[shot]


def metric_source_available(row: dict[str, str]) -> bool:
    source = ROOT / row["source_file"]
    if source.exists():
        return True
    if row["dataset_key"] == "bbq":
        return (
            core.archived_bbq_prediction_dir(row["source_file"]) is not None
            or intervention.local_bbq_prediction_dir(row) is not None
        )
    return intervention.curated_metric_path(row["source_file"]) is not None


def run_rank(row: dict[str, str]) -> tuple[int, int, str]:
    source = row["source_file"]
    for idx, run in enumerate(RUN_PRIORITIES):
        if run in source:
            run_priority = idx
            break
    else:
        run_priority = len(RUN_PRIORITIES)
    availability_penalty = 0 if metric_source_available(row) else 100
    return availability_penalty, run_priority, source


def m3_pattern(config: PlotConfig, dataset_key: str) -> re.Pattern[str]:
    if config.strategy_key == "all_strategies":
        if config.it_key == "allversions":
            return core.exact_m3_model(dataset_key)
        suffix = {
            "crowspairs": "crowspairs",
            "stereoset": "stereoset",
            "bbq": "bbq",
            "winobias": "winobias",
            "winogender": "winogender",
        }[dataset_key]
        return re.compile(rf"^m3_finetune_allstrat_{config.it_key}_{suffix}(?::|$)")

    prefixes = {
        "counter_imaging": "m3_finetune_ci",
        "individuating": "m3_finetune_ind",
        "positive_contact": "m3_finetune_pc",
        "perspective_taking": "m3_finetune_pt",
        "stereotype_replacement": "m3_finetune_sr_indft",
    }
    prefix = prefixes[config.strategy_key]
    if config.strategy_key == "stereotype_replacement":
        suffix = {
            "crowspairs": "crowspairs",
            "stereoset": "stereoset",
            "bbq": "bbq",
            "winobias": "winobias",
            "winogender": "winogender",
        }[dataset_key]
        return re.compile(rf"^{prefix}_{suffix}(?::|$)")
    if dataset_key in {"winobias", "winogender"}:
        return re.compile(rf"^{prefix}_{dataset_key}$")
    if dataset_key == "bbq":
        return re.compile(rf"^{prefix}(?::|$)")
    return re.compile(rf"^{prefix}$")


def instruction_tuning_row(
    rows: list[dict[str, str]], dataset_key: str, config: PlotConfig
) -> dict[str, str] | None:
    pattern = m3_pattern(config, dataset_key)
    candidates = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key
        and r["type"] == "method"
        and r["name"] == "m3_finetune_llama_ms500"
        and r["strategy"] == config.strategy_key
        and core.valid_score(r)
        and pattern.search(r["model"])
    ]
    return sorted(candidates, key=lambda r: (r["source_file"], r["model"]))[0] if candidates else None


def two_pass_model_prefix(config: PlotConfig, kind: str, shot: int) -> str:
    if kind == "two_pass_no_it":
        return f"{shot_prefix(shot)}base__{config.strategy_key}"
    if kind == "two_pass_it":
        return f"{shot_prefix(shot)}finetuned_ms-500-{config.ft_slug}__{config.strategy_key}"
    raise ValueError(f"Unsupported two-pass kind: {kind}")


def two_pass_row(
    rows: list[dict[str, str]], dataset_key: str, config: PlotConfig, kind: str, shot: int
) -> dict[str, str] | None:
    prefix = two_pass_model_prefix(config, kind, shot)
    candidates = [
        r
        for r in rows
        if r["dataset_key"] == dataset_key
        and r["type"] == "method"
        and r["name"] == "m6_self_debiasing"
        and r["strategy"] == config.strategy_key
        and core.valid_score(r)
        and r["model"].startswith(prefix)
        and "same_thread" not in r["model"]
        and "noncog" not in r["model"]
    ]
    return sorted(candidates, key=run_rank)[0] if candidates else None


def icl_tag(config: PlotConfig) -> str:
    if config.strategy_key == "all_strategies":
        return "allstrat"
    return intervention.ICL_TAGS[config.strategy_key]


def ft_icl_tag(config: PlotConfig) -> str:
    if config.strategy_key == "all_strategies":
        return f"all_{config.it_key}"
    return intervention.ICL_TAGS[config.strategy_key]


def m4_path(dataset_key: str, *, shot: int, tag: str, ft: bool) -> Path | None:
    shot_word = "one" if shot == 1 else "zero"
    if ft:
        root = FT_ICL_ONE_ROOT if shot == 1 else FT_ICL_ZERO_ROOT
        prefix = f"m4_fticl_{shot_word}_llama8b_{tag}"
    else:
        root = ICL_ONE_ROOT if shot == 1 else core.ICL_ROOT
        prefix = f"m4_baseicl_{shot_word}_llama8b_{tag}"

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


def selected_row(
    rows: list[dict[str, str]],
    dataset_key: str,
    config: PlotConfig,
    method: dict[str, object],
) -> dict[str, str] | None:
    kind = str(method["kind"])
    if kind == "base":
        return core.base_row(rows, dataset_key)
    if kind == "it":
        return instruction_tuning_row(rows, dataset_key, config)
    if kind in {"icl", "ft_icl"}:
        return None
    return two_pass_row(rows, dataset_key, config, kind, int(method["shot"]))


def metric_value(row: dict[str, str], metric: str) -> tuple[float, str]:
    if row["dataset_key"] == "bbq" and row["name"] == "m6_self_debiasing":
        path = ROOT / row["source_file"]
        has_overall = False
        if path.exists():
            with path.open(newline="") as f:
                has_overall = any(
                    metric_row.get("input_file") == "__overall__"
                    for metric_row in csv.DictReader(f)
                )
        if not has_overall:
            local_pred_dir = intervention.local_bbq_prediction_dir(row)
            if local_pred_dir is not None:
                return (
                    intervention.recompute_bbq_metrics_from_all_predictions(local_pred_dir)[
                        metric
                    ],
                    f"recomputed_from_predictions:{local_pred_dir.relative_to(ROOT)}",
                )
            archived_pred_dir = core.archived_bbq_prediction_dir(row["source_file"])
            if archived_pred_dir is not None:
                return (
                    core.recompute_bbq_metrics_from_predictions(archived_pred_dir)[metric],
                    f"recomputed_from_predictions:{archived_pred_dir.relative_to(ROOT)}",
                )
    return intervention.metric_value(row, metric)


def build_table(
    rows: list[dict[str, str]], config: PlotConfig
) -> tuple[np.ndarray, list[dict[str, str]]]:
    matrix = np.full((len(core.PANELS), len(METHODS)), np.nan)
    audit: list[dict[str, str]] = []

    for i, (dataset_key, panel_label, metric, metric_label, ideal) in enumerate(core.PANELS):
        for j, method in enumerate(METHODS):
            kind = str(method["kind"])
            if kind in {"icl", "ft_icl"}:
                shot = int(method["shot"])
                tag = icl_tag(config) if kind == "icl" else ft_icl_tag(config)
                path = m4_path(dataset_key, shot=shot, tag=tag, ft=(kind == "ft_icl"))
                if path is None or not path.exists():
                    continue
                native_value = core.read_metric_path(path, dataset_key, metric)
                plotted_value = core.normalized_bias_error(native_value, ideal)
                matrix[i, j] = plotted_value
                audit.append(
                    {
                        "figure_slug": config.slug,
                        "figure_title": config.title,
                        "strategy_key": config.strategy_key,
                        "strategy_label": config.strategy_label,
                        "ft_checkpoint": config.ft_label,
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
                        "strategy": config.strategy_key,
                        "metric_source_file": str(path.relative_to(ROOT)),
                    }
                )
                continue
            row = selected_row(rows, dataset_key, config, method)
            if row is None:
                continue
            native_value, source = metric_value(row, metric)
            plotted_value = core.normalized_bias_error(native_value, ideal)
            matrix[i, j] = plotted_value
            audit.append(
                {
                    "figure_slug": config.slug,
                    "figure_title": config.title,
                    "strategy_key": config.strategy_key,
                    "strategy_label": config.strategy_label,
                    "ft_checkpoint": config.ft_label,
                    "dataset_key": dataset_key,
                    "panel_label": panel_label,
                    "method": str(method["audit_label"]),
                    "setting": str(method["setting"]),
                    "shot": str(method.get("shot", "")),
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
                    "metric_source_file": source,
                }
            )

    return matrix, audit


def value_label(value: float) -> str:
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def plot_config(matrix: np.ndarray, config: PlotConfig) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11.3,
            "axes.titlesize": 13.2,
            "axes.labelsize": 11.4,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.6,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 6.95), sharey=False)
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
            pad=8,
            fontsize=13.5,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(
                facecolor="#F1F5F9",
                edgecolor="#CBD5E1",
                boxstyle="round,pad=0.22",
                linewidth=0.7,
            ),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if i % 3 == 0:
            ax.set_ylabel("Bias error", fontsize=11.3, color="#374151", fontweight="semibold")

        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            ymax = float(finite_values.max())
            ax.set_ylim(0, ymax * 1.25 if ymax else 1.0)

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
                fontsize=9.4,
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
                fontsize=9.4,
                color=color,
                fontweight="semibold",
            )

    fig.suptitle(
        config.title,
        y=0.978,
        fontsize=17.3,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.01,
        0.049,
        f"Strategy setting: {config.strategy_label}. IT checkpoint: {config.ft_label}. "
        "Bars use the benchmark-native metric normalized as bias error; lower is better.",
        ha="left",
        va="bottom",
        fontsize=9.3,
        color="#374151",
    )
    fig.text(
        0.01,
        0.028,
        "Metrics: CrowS stereotype preference, StereoSet SS overall, BBQ ambiguous/disambiguated bias score, "
        "WinoBias pro-anti gap, and WinoGender male-female disagreement. Shot labels are demonstrations for Show/Revise settings.",
        ha="left",
        va="bottom",
        fontsize=9.3,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.91), w_pad=1.25, h_pad=1.35)

    (OUTDIR / "pdf").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTDIR / "pdf" / f"debiasing_method_bars_by_shot_{config.slug}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_csv(rows: list[dict[str, str]]) -> None:
    path = OUTDIR / "csv/debiasing_method_bars_by_shot_configs_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = core.load_rows()
    all_audit: list[dict[str, str]] = []
    for config in CONFIGS:
        matrix, audit = build_table(rows, config)
        if not audit:
            raise RuntimeError(f"No rows collected for {config.slug}")
        all_audit.extend(audit)
        plot_config(matrix, config)
    write_csv(all_audit)


if __name__ == "__main__":
    main()
