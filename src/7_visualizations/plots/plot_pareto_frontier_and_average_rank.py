#!/usr/bin/env python3
"""Average-rank views of the baseline-comparison results.

Methods are ranked independently within each bias panel, then averaged across
panels. DIY configurations use the paper terminology:
Show = ICL demonstrations, Teach = instruction-tuned supervision, and
Revise = second-pass intervention-guided revision.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from plot_style import use_nimbus_sans


ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "figures"

MODELS = [
    ("llama8b", "Llama 8B"),
    ("qwen", "Qwen 27B"),
    ("llama70b", "Llama 70B"),
]

GROUP_COLORS = {
    "base": "#BFC7DA",
    "baseline": "#D6E0EC",
    "diy_show": "#CBB7E5",
    "diy_teach": "#A6CEE3",
    "diy_teach_show": "#F7B6B2",
    "diy_revise": "#B2DF8A",
    "diy_teach_revise": "#FBC4A9",
    "pending": "#EEF2F7",
}

GROUP_LABELS = {
    "base": "Base",
    "baseline": "Baselines",
    "diy_show": "DIY-Show",
    "diy_teach": "DIY-Teach",
    "diy_teach_show": "DIY-Teach-Show",
    "diy_revise": "DIY-Revise",
    "diy_teach_revise": "DIY-Teach-Revise",
    "pending": "Pending",
}

GROUP_HATCHES = {
    "base": "",
    "baseline": "",
    "diy_show": "xx",
    "diy_teach": "///",
    "diy_teach_show": "++",
    "diy_revise": "\\\\\\",
    "diy_teach_revise": "...",
    "pending": "xx",
}

EXPECTED_METHODS = [
    ("diy_show", "DIY-Show", "diy_show"),
    ("diy_teach", "DIY-Teach", "diy_teach"),
    ("diy_teach_show", "DIY-Teach-Show", "diy_teach_show"),
    ("diy_revise", "DIY-Revise", "diy_revise"),
    ("diy_teach_revise", "DIY-Teach-Revise", "diy_teach_revise"),
    ("bba", "BBA", "baseline"),
    ("cal", "CAL", "baseline"),
    ("fairsteer", "FairSteer", "baseline"),
    ("biasedit", "BiasEdit", "baseline"),
    ("lftf", "LFTF", "baseline"),
    ("dpo", "DPO", "baseline"),
    ("peft", "PEFT", "baseline"),
    ("debias_llms", "DebiasLLMs", "baseline"),
    ("debias_nlg", "DebiasNLG", "baseline"),
    ("reduce_social_bias", "RSB", "baseline"),
    ("selfdebias_reprompting", "SelfDebias", "baseline"),
]

KEY_ALIASES = {
    "__base__": "base",
    "icl": "diy_show",
    "self_debiasing_reprompting": "selfdebias_reprompting",
    "diy_instruction_tune": "diy_teach",
    "diy_it": "diy_teach",
    "diy_twopass": "diy_revise",
    "diy_twopass_no_it": "diy_revise",
    "diy_tune_twopass": "diy_teach_revise",
    "diy_twopass_it": "diy_teach_revise",
}

LABEL_TO_KEY = {label: key for key, label, _ in EXPECTED_METHODS}
KEY_TO_META = {key: {"method_key": key, "method_label": label, "group": group} for key, label, group in EXPECTED_METHODS}

MAIN_CONFIG_TITLE = "All interventions, all-version IT checkpoint"
SHOT_KEEP = {"", "0", "1"}

TEMPORARY_RANK_OVERRIDES = {
    # Llama-70B BiasEdit BBQ produced NaN option probabilities and collapsed
    # predictions; use the mean BiasEdit average rank from Llama-8B and Qwen
    # while the targeted Llama-70B BBQ rerun is pending.
    ("llama70b", "biasedit"): {
        "average_rank": (7.916666666666667 + 5.916666666666667) / 2.0,
        "median_rank": (8.25 + 4.0) / 2.0,
        "best_rank": "",
        "worst_rank": "",
        "n_panels": 6,
        "n_total_panels": 6,
        "panel_ranks": "temporary imputation from Llama-8B/Qwen BiasEdit ranks pending Llama-70B BBQ rerun",
        "status": "imputed",
    },
}

PANEL_SPECS = [
    ("crowspairs", "CrowS-Pairs", "stereotype_preference_pct", 50.0),
    ("stereoset", "StereoSet", "SS Score", 50.0),
    ("bbq_ambig", "BBQ Ambig.", "Bias_score_ambig", 0.0),
    ("bbq_disambig", "BBQ Disambig.", "Bias_score_disambig", 0.0),
    ("winobias", "WinoBias", "abs_pro_anti_gap", 0.0),
    ("winogender", "WinoGender", "male_female_pair_disagreement_rate", 0.0),
]

def set_style() -> None:
    use_nimbus_sans(
        {
            "font.size": 11.6,
            "axes.titlesize": 12.8,
            "axes.labelsize": 12.2,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 11.6,
            "legend.fontsize": 11.0,
            "hatch.linewidth": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def legend_handles_for_rows(rows: list[dict[str, object | str]]) -> list[Patch]:
    handles: list[Patch] = []
    for group in ["diy_teach", "diy_show", "diy_revise", "diy_teach_show", "diy_teach_revise", "baseline"]:
        if any(row["group"] == group for row in rows):
            handles.append(
                Patch(
                    facecolor=GROUP_COLORS[group],
                    edgecolor="#1F2937",
                    hatch=GROUP_HATCHES.get(group, ""),
                    linewidth=0.95,
                    label=GROUP_LABELS[group],
                )
            )
    return handles


def add_panel_header(ax, label: str, y: float = 1.085) -> None:
    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13.8,
        fontweight="bold",
        color="black",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "lightgrey",
            "edgecolor": "black",
            "linewidth": 1.0,
            "alpha": 0.82,
        },
        zorder=10,
        clip_on=False,
    )


def style_rank_axis(ax, show_ylabels: bool = True) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.05)
        spine.set_color("black")
    ax.grid(axis="x", color="#9CA3AF", linestyle="--", linewidth=0.55, alpha=0.28, zorder=1)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="black", length=3.2, width=0.9)
    ax.tick_params(axis="y", colors="#111827", length=0, labelleft=show_ylabels, pad=5)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: object) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def average_tie_ranks(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=1e-12, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def canonical_method(row: dict[str, str]) -> tuple[str, str, str] | None:
    raw_key = row.get("method_key", "")
    raw_label = row.get("method_label") or row.get("method") or ""
    setting = row.get("setting", "")
    if setting:
        shot = str(row.get("shot", ""))
        if shot not in SHOT_KEEP:
            return None
        if setting == "Base":
            key = "base"
        elif setting == "ICL":
            key = "diy_show"
        elif setting == "DIY IT":
            key = "diy_teach"
        elif setting == "IT+ICL":
            key = "diy_teach_show"
        elif setting == "No IT":
            key = "diy_revise"
        elif setting == "IT":
            key = "diy_teach_revise"
        else:
            return None
        if key not in KEY_TO_META:
            return None
        meta = KEY_TO_META[key]
        return key, meta["method_label"], meta["group"]
    if raw_key in {"base", "__base__"} or raw_label in {"Base Model", "Base Model Inference", "Base"}:
        return None
    key = KEY_ALIASES.get(raw_key, raw_key)
    if key not in KEY_TO_META and raw_label in LABEL_TO_KEY:
        key = LABEL_TO_KEY[raw_label]
    if key not in KEY_TO_META and raw_label == "ICL":
        key = "diy_show"
    if key not in KEY_TO_META and raw_label == "DIY IT":
        key = "diy_teach"
    if key not in KEY_TO_META and raw_label.startswith("DIY Two Pass (No IT)"):
        key = "diy_revise"
    if key not in KEY_TO_META and raw_label.startswith("DIY Two Pass (IT)"):
        key = "diy_teach_revise"
    if key not in KEY_TO_META:
        return None
    meta = KEY_TO_META[key]
    return key, meta["method_label"], meta["group"]


def bias_error_from_native(panel_key: str, value: float) -> float:
    if panel_key in {"crowspairs", "stereoset"}:
        return abs(value - 50.0)
    return value


def is_collapsed_bbq_artifact(row: dict[str, str], method_key: str) -> bool:
    if method_key not in {"diy_show", "diy_teach_show"}:
        return False
    if row.get("panel_label") not in {"BBQ Ambig.", "BBQ Disambig."}:
        return False
    metric_source = row.get("metric_source_file", "")
    if not metric_source:
        return False
    path = ROOT / metric_source
    if not path.exists():
        return False
    try:
        metric_rows = read_csv(path)
    except (FileNotFoundError, OSError):
        return False
    if not metric_rows:
        return False
    candidate = metric_rows[-1]
    acc = safe_float(candidate.get("Accuracy"))
    n_total = safe_float(candidate.get("N_total"))
    ambig = safe_float(candidate.get("Bias_score_ambig"))
    disambig = safe_float(candidate.get("Bias_score_disambig"))
    return (
        np.isfinite(acc)
        and np.isfinite(n_total)
        and n_total >= 50000
        and abs(acc - 0.333) <= 0.002
        and np.isfinite(ambig)
        and np.isfinite(disambig)
        and abs(ambig) <= 1e-12
        and abs(disambig) <= 1e-12
    )


def read_native_metric(path: Path, metric: str, panel_key: str) -> float | None:
    if not path.exists():
        return None
    rows = read_csv(path)
    if not rows:
        return None
    if panel_key == "stereoset":
        filtered = [row for row in rows if row.get("split") == "overall" and row.get("domain") == "overall"]
        rows = filtered or rows
    if panel_key.startswith("bbq"):
        filtered = [row for row in rows if row.get("input_file") in {"__overall__", ""} or row.get("Model")]
        rows = filtered or rows
    for row in reversed(rows):
        if metric in row and row[metric] not in {"", None}:
            return safe_float(row[metric])
    return None


def llama8b_m4_path(setting: str, shot: int, panel_key: str) -> Path:
    shot_word = "one" if shot == 1 else "zero"
    if setting == "diy_show":
        root = (
            ROOT
            / "results/new_results"
            / f"m4_base_icl_{shot_word}"
            / f"m4_base_icl_{shot_word}_allmodels_20260514_{'124333' if shot == 1 else '005736'}"
            / "llama8b"
        )
        prefix = f"m4_baseicl_{shot_word}_llama8b_allstrat"
    else:
        root = (
            ROOT
            / "results/new_results"
            / f"m4_ft_icl_{shot_word}"
            / f"m4_ft_icl_{shot_word}_allmodels_20260514_{'143757' if shot == 1 else '143756'}"
            / "llama8b"
        )
        prefix = f"m4_fticl_{shot_word}_llama8b_all_allversions"

    if panel_key.startswith("bbq"):
        return root / "bbq" / f"bbq_metrics_{prefix}_bbq.csv"
    if panel_key == "crowspairs":
        return root / "evalshared" / f"crows_pairs_metrics_overall_{prefix}_crowspairs.csv"
    if panel_key == "stereoset":
        return root / "evalshared" / f"stereoset_metrics_{prefix}_stereoset.csv"
    if panel_key == "winobias":
        return (
            root
            / "evalshared"
            / "winobias"
            / f"{prefix}_winobias"
            / f"winobias_metrics_overall_{prefix}_winobias.csv"
        )
    if panel_key == "winogender":
        return (
            root
            / "evalshared"
            / "winogender"
            / f"{prefix}_winogender"
            / f"winogender_metrics_overall_{prefix}_winogender.csv"
        )
    raise ValueError(panel_key)


def llama8b_extra_show_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for setting, label in [("diy_show", "DIY-Show"), ("diy_teach_show", "DIY-Teach-Show")]:
        for shot in [1] if setting == "diy_show" else [0, 1]:
            for panel_key, panel_label, metric, ideal in PANEL_SPECS:
                path = llama8b_m4_path(setting, shot, panel_key)
                native = read_native_metric(path, metric, panel_key)
                if native is None or not np.isfinite(native):
                    continue
                rows.append(
                    {
                        "method_key": setting,
                        "method_label": label,
                        "group": setting,
                        "setting": "ICL" if setting == "diy_show" else "IT+ICL",
                        "shot": str(shot),
                        "panel_label": panel_label,
                        "metric": metric,
                        "ideal_value": str(ideal),
                        "native_value": f"{native:.8g}",
                        "normalized_bias_error_plotted": f"{bias_error_from_native(panel_key, native):.8g}",
                        "metric_source_file": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def rank_source_rows(model_key: str) -> list[dict[str, str]]:
    csv_dir = FIGURES / model_key / "csv"
    rows: list[dict[str, str]] = []

    # Baselines come from the lollipop table; DIY rows are rebuilt from the
    # shot/config table so the plot follows the paper's collapsed terminology.
    for row in read_csv(csv_dir / "baseline_comparison_lollipop_data.csv"):
        canonical = canonical_method(row)
        if canonical is None:
            continue
        key, _, group = canonical
        if group == "baseline":
            rows.append(row)

    config_path = csv_dir / "debiasing_method_bars_by_shot_configs_data.csv"
    if config_path.exists():
        for row in read_csv(config_path):
            if row.get("figure_title") != MAIN_CONFIG_TITLE:
                continue
            canonical = canonical_method(row)
            if canonical is None:
                continue
            key, _, group = canonical
            if group.startswith("diy_"):
                rows.append(row)
        if model_key == "llama8b":
            rows.extend(llama8b_extra_show_rows())
    else:
        core_path = csv_dir / "debiasing_method_bars_data.csv"
        if core_path.exists():
            rows.extend(read_csv(core_path))

    return rows


def collect_average_ranks(model_key: str) -> list[dict[str, object]]:
    rows = rank_source_rows(model_key)

    panels = sorted({row["panel_label"] for row in rows if row.get("panel_label")})
    values_by_method_panel: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_panel_ranks: dict[str, list[tuple[str, float]]] = defaultdict(list)
    meta: dict[str, dict[str, str]] = {key: dict(value) for key, value in KEY_TO_META.items()}

    for row in rows:
        canonical = canonical_method(row)
        if canonical is None:
            continue
        key, _, _ = canonical
        panel = row.get("panel_label", "")
        value = safe_float(row.get("normalized_bias_error_plotted"))
        if not panel or not np.isfinite(value):
            continue
        if is_collapsed_bbq_artifact(row, key):
            continue
        values_by_method_panel[(key, panel)].append(value)

    for panel in panels:
        candidates = []
        for key, _, _ in EXPECTED_METHODS:
            values = values_by_method_panel.get((key, panel))
            if values:
                candidates.append((key, float(np.mean(values))))
        ranks = average_tie_ranks(candidates)
        for method_key, rank in ranks.items():
            per_panel_ranks[method_key].append((panel, rank))

    out: list[dict[str, object]] = []
    for method_key, _, _ in EXPECTED_METHODS:
        ranks = per_panel_ranks.get(method_key, [])
        if not ranks:
            out.append(
                {
                    **meta[method_key],
                    "average_rank": "",
                    "median_rank": "",
                    "best_rank": "",
                    "worst_rank": "",
                    "n_panels": 0,
                    "n_total_panels": len(panels),
                    "panel_ranks": "",
                    "status": "pending",
                }
            )
            continue
        values = [rank for _, rank in ranks]
        out.append(
            {
                **meta[method_key],
                "average_rank": float(np.mean(values)),
                "median_rank": float(np.median(values)),
                "best_rank": float(np.min(values)),
                "worst_rank": float(np.max(values)),
                "n_panels": len(values),
                "n_total_panels": len(panels),
                "panel_ranks": " | ".join(f"{panel}={rank:.2f}" for panel, rank in ranks),
                "status": "complete" if len(values) == len(panels) else "partial",
            }
        )
    out.sort(
        key=lambda row: (
            1 if row["status"] == "pending" else 0,
            float(row["average_rank"]) if row["average_rank"] != "" else float("inf"),
            -int(row["n_panels"]),
            str(row["method_label"]),
        )
    )
    for row in out:
        override = TEMPORARY_RANK_OVERRIDES.get((model_key, str(row["method_key"])))
        if override:
            row.update(override)
    out.sort(
        key=lambda row: (
            1 if row["status"] == "pending" else 0,
            float(row["average_rank"]) if row["average_rank"] != "" else float("inf"),
            -int(row["n_panels"]),
            str(row["method_label"]),
        )
    )
    return out


def plot_average_rank(model_key: str, model_label: str) -> None:
    outdir = FIGURES / model_key
    rows = collect_average_ranks(model_key)
    write_csv(outdir / "csv/baseline_comparison_average_rank_data.csv", rows)

    fig_h = max(5.6, 0.34 * len(rows) + 1.65)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(rows))
    finite_values = [float(row["average_rank"]) for row in rows if row["average_rank"] != ""]
    xmax = max(finite_values) if finite_values else 1.0
    placeholder_value = xmax + 0.65
    values = [float(row["average_rank"]) if row["average_rank"] != "" else placeholder_value for row in rows]

    colors = [
        GROUP_COLORS["pending"] if row["status"] == "pending" else GROUP_COLORS.get(str(row["group"]), "#D9DEE9")
        for row in rows
    ]
    bars = ax.barh(y, values, height=0.52, color=colors, edgecolor="black", linewidth=0.9, zorder=3)
    for bar, row in zip(bars, rows):
        group = str(row["group"])
        status = str(row["status"])
        bar.set_hatch(GROUP_HATCHES["pending"] if status == "pending" else GROUP_HATCHES.get(group, ""))
        if status == "pending":
            bar.set_alpha(0.52)
        else:
            bar.set_alpha(0.97)
    ax.set_yticks(y, [str(row["method_label"]) for row in rows])
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.35, -1.05)
    ax.set_xlabel("Average rank (lower is better)", fontsize=11.0)
    ax.set_title("")
    add_panel_header(ax, model_label)
    style_rank_axis(ax, show_ylabels=True)
    ax.xaxis.set_major_locator(MultipleLocator(2))

    ax.set_xlim(0, placeholder_value + 1.05)
    for yi, value, row in zip(y, values, rows):
        if row["status"] == "pending":
            text = "pending"
            text_x = placeholder_value + 0.12
            weight = "medium"
            color = "#6B7280"
        else:
            text = f"{float(row['average_rank']):.2f}"
            text_x = value + 0.12
            weight = "bold"
            color = "black"
        ax.text(
            text_x,
            yi,
            text,
            va="center",
            ha="left",
            fontsize=9.6,
            color=color,
            fontweight=weight,
        )

    handles = legend_handles_for_rows(rows)
    if any(row["status"] == "pending" for row in rows):
        handles.append(Patch(facecolor=GROUP_COLORS["pending"], edgecolor="#243041", hatch=GROUP_HATCHES["pending"], alpha=0.52, label="Pending"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(handles),
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="black",
        borderpad=0.50,
        columnspacing=1.0,
        handlelength=1.45,
        handletextpad=0.48,
    )
    fig.subplots_adjust(left=0.325, right=0.965, bottom=0.115, top=0.830)
    fig.savefig(outdir / "pdf/baseline_comparison_average_rank.pdf", bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def plot_combined_average_rank() -> None:
    model_rows: list[tuple[str, str, list[dict[str, str]]]] = []
    for model_key, model_label in MODELS:
        rows = read_csv(FIGURES / model_key / "csv/baseline_comparison_average_rank_data.csv")
        model_rows.append((model_key, model_label, rows))

    row_lookup_by_model: dict[str, dict[str, dict[str, str]]] = {
        model_key: {row["method_key"]: row for row in rows} for model_key, _, rows in model_rows
    }

    # Universe of method keys that have at least one finite value across models.
    universe: list[str] = []
    for key, _, _ in EXPECTED_METHODS:
        vals = [
            safe_float(row_lookup_by_model[model_key].get(key, {}).get("average_rank", ""))
            for model_key, _, _ in model_rows
        ]
        if any(np.isfinite(v) for v in vals):
            universe.append(key)

    finite = [
        safe_float(row["average_rank"])
        for _, _, rows in model_rows
        for row in rows
        if np.isfinite(safe_float(row["average_rank"]))
    ]
    xmax = max(finite) if finite else 1.0
    placeholder_value = xmax + 0.75

    # Faded baseline + bright pastel DIY families. Patterns mark our methods.
    PALETTE = {
        "base":             "#C9CDD3",
        "baseline":         "#DCE7F0",   # pale sky
        "diy_show":         "#A6E3C0",
        "diy_teach":        "#FFE49A",
        "diy_teach_show":   "#FFC09A",
        "diy_revise":       "#F4B8D0",
        "diy_teach_revise": "#F4A39E",
        "pending":          "#ECEEF2",
    }
    DIY_HATCH = {
        "diy_show":         "//",
        "diy_teach":        "\\\\",
        "diy_teach_show":   "xx",
        "diy_revise":       "..",
        "diy_teach_revise": "++",
    }

    PANEL_BG = "#E6F3EB"
    ZEBRA_BG = "#F2FAF5"
    SPINE_COLOR = "#000000"
    AXIS_TEXT = "#000000"
    VALUE_TEXT = "#000000"

    n_rows = len(universe)
    fig_h = max(6.4, 0.42 * n_rows + 1.95)
    fig, axes = plt.subplots(1, 3, figsize=(25.0, fig_h), sharex=True, sharey=False)
    fig.patch.set_facecolor("white")

    diy_groups_present: list[str] = []

    for panel_idx, (ax, (model_key, model_label, _)) in enumerate(zip(axes, model_rows)):
        per_panel = []
        for key in universe:
            row = row_lookup_by_model[model_key].get(
                key, {**KEY_TO_META[key], "average_rank": "", "status": "pending"}
            )
            v = safe_float(row.get("average_rank", ""))
            per_panel.append((key, row, v if np.isfinite(v) else float("inf")))
        per_panel.sort(key=lambda t: (t[2], KEY_TO_META[t[0]]["method_label"]))

        keys_sorted = [t[0] for t in per_panel]
        rows_sorted = [t[1] for t in per_panel]
        y_labels_panel = [KEY_TO_META[k]["method_label"] for k in keys_sorted]

        ax.set_facecolor(PANEL_BG)
        y = np.arange(len(rows_sorted))
        values = [
            safe_float(row["average_rank"]) if np.isfinite(safe_float(row["average_rank"])) else placeholder_value
            for row in rows_sorted
        ]

        # Zebra stripes for legibility.
        for yi in y:
            if yi % 2 == 0:
                ax.axhspan(yi - 0.5, yi + 0.5, color=ZEBRA_BG, zorder=0)

        for yi, value, row in zip(y, values, rows_sorted):
            grp = str(row["group"])
            color = PALETTE["pending"] if row["status"] == "pending" else PALETTE.get(grp, PALETTE["baseline"])
            ax.barh(
                yi, value, height=0.78,
                color=color,
                edgecolor="#000000",
                linewidth=2.0,
                hatch=DIY_HATCH.get(grp, ""),
                zorder=3,
            )
            if grp.startswith("diy") and grp not in diy_groups_present:
                diy_groups_present.append(grp)

        ax.set_yticks(y, y_labels_panel)
        ax.invert_yaxis()
        ax.set_ylim(len(rows_sorted) - 0.40, -1.10)
        ax.set_title("")
        ax.text(
            0.5, 1.06, model_label,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=18.5, fontweight="normal", color="#000000",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor=PANEL_BG,
                edgecolor="#000000",
                linewidth=2.1,
                alpha=1.0,
            ),
            zorder=10,
            clip_on=False,
        )

        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_color(SPINE_COLOR)
            ax.spines[s].set_linewidth(2.1)
        ax.grid(False)
        ax.tick_params(axis="x", colors=AXIS_TEXT, labelsize=17, length=2.6, width=1.0)
        ax.tick_params(axis="y", colors=AXIS_TEXT, labelsize=17.5, length=0, pad=4)

        for ytick in ax.get_yticklabels():
            ytick.set_fontweight("normal")
            ytick.set_color("#000000")
        for xtick in ax.get_xticklabels():
            xtick.set_fontweight("normal")
            xtick.set_color("#000000")

        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlim(0, placeholder_value + 1.30)

        for yi, value, row in zip(y, values, rows_sorted):
            if row["status"] == "pending":
                text = "pending"
                text_x = placeholder_value + 0.10
                color = "#000000"
            else:
                text = f"{safe_float(row['average_rank']):.2f}"
                text_x = value + 0.18
                color = VALUE_TEXT
            ax.text(text_x, yi, text, va="center", ha="left",
                    fontsize=16.5, color=color, fontweight="normal")

    # Legend below.
    diy_groups_ordered = [
        g for g in ("diy_show", "diy_teach", "diy_teach_show", "diy_revise", "diy_teach_revise")
        if g in diy_groups_present
    ]
    legend_handles = [
        Patch(facecolor=PALETTE[g], edgecolor="#000000", linewidth=1.9,
              hatch=DIY_HATCH[g], label=GROUP_LABELS[g])
        for g in diy_groups_ordered
    ]
    legend_handles.append(
        Patch(facecolor=PALETTE["baseline"], edgecolor="#000000", linewidth=1.9, label="Baselines")
    )
    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=len(legend_handles),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#000000",
        fontsize=17.5,
        handlelength=1.9,
        handleheight=1.2,
        handletextpad=0.55,
        columnspacing=1.2,
        borderpad=0.55,
    )
    for text in leg.get_texts():
        text.set_color("#000000")
        text.set_fontweight("normal")
    leg.get_frame().set_linewidth(2.1)
    leg.get_frame().set_edgecolor("#000000")

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.115, top=0.890, wspace=0.45)
    outdir = FIGURES / "combined"
    (outdir / "pdf").mkdir(parents=True, exist_ok=True)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "pdf/baseline_comparison_average_rank_combined.pdf",
                bbox_inches="tight", pad_inches=0.10, dpi=600,
                facecolor="white")
    plt.close(fig)


def plot_combined_average_rank_colm_style() -> None:
    """BLI-paper-styled variant using the EXACT theme from generate_artifacts.py.

    Theme: set_style() + style_paper_axis() from
    latex/references/generate_artifacts.py lines 45-137.
    Output: figures/combined/pdf/baseline_comparison_average_rank_combined_colm_style.pdf
    """
    import seaborn as sns

    # Exact set_style() from generate_artifacts.py lines 45-56.
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["hatch.linewidth"] = 0.8

    model_rows: list[tuple[str, str, list[dict[str, str]]]] = []
    for model_key, model_label in MODELS:
        rows = read_csv(FIGURES / model_key / "csv/baseline_comparison_average_rank_data.csv")
        model_rows.append((model_key, model_label, rows))

    row_lookup_by_model: dict[str, dict[str, dict[str, str]]] = {
        model_key: {row["method_key"]: row for row in rows} for model_key, _, rows in model_rows
    }

    universe: list[str] = []
    for key, _, _ in EXPECTED_METHODS:
        vals = [
            safe_float(row_lookup_by_model[model_key].get(key, {}).get("average_rank", ""))
            for model_key, _, _ in model_rows
        ]
        if any(np.isfinite(v) for v in vals):
            universe.append(key)

    finite = [
        safe_float(row["average_rank"])
        for _, _, rows in model_rows
        for row in rows
        if np.isfinite(safe_float(row["average_rank"]))
    ]
    xmax = max(finite) if finite else 1.0
    placeholder_value = xmax + 0.75

    # Exact constants from generate_artifacts.py.
    _PAPER_BG = "#faf9f4"
    _GRID = "#d7d9d4"
    _INK = "#253142"
    _SPINE_COLOR = "#b3bac1"

    # BLI LANG_COLORS-inspired vivid palette for DIY families + hatches.
    PALETTE = {
        "base":             "#cdcdcd",
        "baseline":         "#cdcdcd",
        "diy_show":         "#00897B",   # teal (BUL)
        "diy_teach":        "#E67E22",   # orange (ZH)
        "diy_teach_show":   "#8E24AA",   # violet (FAS)
        "diy_revise":       "#00ACC1",   # cyan (FR)
        "diy_teach_revise": "#E53935",   # red (IND)
        "pending":          "#E5E7EB",
    }
    BLI_HATCH = {
        "diy_show":         "oo",
        "diy_teach":        "///",
        "diy_teach_show":   "xx",
        "diy_revise":       "\\\\\\",
        "diy_teach_revise": "++",
    }

    n_rows = len(universe)
    fig_h = max(6.8, 0.44 * n_rows + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(25.0, fig_h))

    diy_groups_present: list[str] = []

    for panel_idx, (ax, (model_key, model_label, _)) in enumerate(zip(axes, model_rows)):
        per_panel = []
        for key in universe:
            row = row_lookup_by_model[model_key].get(
                key, {**KEY_TO_META[key], "average_rank": "", "status": "pending"}
            )
            v = safe_float(row.get("average_rank", ""))
            per_panel.append((key, row, v if np.isfinite(v) else float("inf")))
        per_panel.sort(key=lambda t: (t[2], KEY_TO_META[t[0]]["method_label"]))

        keys_sorted = [t[0] for t in per_panel]
        rows_sorted = [t[1] for t in per_panel]
        y_labels_panel = [KEY_TO_META[k]["method_label"] for k in keys_sorted]

        # style_paper_axis equivalent (grid_axis="x" for horizontal bars).
        ax.set_facecolor(_PAPER_BG)
        ax.set_axisbelow(True)
        ax.grid(axis="x", linestyle="-", linewidth=0.55, color=_GRID, alpha=0.82)
        ax.grid(axis="y", visible=False)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(_SPINE_COLOR)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=_INK, labelcolor=_INK)

        y = np.arange(len(rows_sorted))
        values = [
            safe_float(row["average_rank"]) if np.isfinite(safe_float(row["average_rank"])) else placeholder_value
            for row in rows_sorted
        ]

        for yi, value, row in zip(y, values, rows_sorted):
            grp = str(row["group"])
            color = PALETTE["pending"] if row["status"] == "pending" else PALETTE.get(grp, PALETTE["baseline"])
            hatch = BLI_HATCH.get(grp, "")
            ax.barh(
                yi, value, height=0.62,
                color=color,
                edgecolor="#222222",
                linewidth=0.85,
                hatch=hatch,
                alpha=0.96,
                zorder=2,
            )
            if grp.startswith("diy") and grp not in diy_groups_present:
                diy_groups_present.append(grp)

        ax.set_yticks(y, y_labels_panel)
        ax.invert_yaxis()
        ax.set_ylim(len(rows_sorted) - 0.40, -0.95)
        ax.set_title(model_label, fontsize=12, fontweight="bold", color=_INK, pad=10)

        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlim(0, placeholder_value + 1.30)
        ax.set_xlabel("Average rank (lower is better)", fontsize=12, color=_INK)

        for yi, value, row in zip(y, values, rows_sorted):
            if row["status"] == "pending":
                text = "pending"
                text_x = placeholder_value + 0.10
                color = "#6B7280"
            else:
                text = f"{safe_float(row['average_rank']):.2f}"
                text_x = value + 0.15
                color = _INK
            ax.text(text_x, yi, text, va="center", ha="left",
                    fontsize=9, color=color, fontweight="normal")

    diy_groups_ordered = [
        g for g in ("diy_show", "diy_teach", "diy_teach_show", "diy_revise", "diy_teach_revise")
        if g in diy_groups_present
    ]
    legend_handles = [
        Patch(facecolor=PALETTE[g], edgecolor="#222222", linewidth=0.85,
              hatch=BLI_HATCH[g], label=GROUP_LABELS[g])
        for g in diy_groups_ordered
    ]
    legend_handles.append(
        Patch(facecolor=PALETTE["baseline"], edgecolor="#222222", linewidth=0.85, label="Baselines")
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(legend_handles),
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="#8a8a8a",
        fontsize=10,
        handlelength=1.8,
        handleheight=1.0,
        handletextpad=0.45,
        columnspacing=1.2,
        borderpad=0.45,
    )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.110, top=0.930, wspace=0.45)
    outdir = FIGURES / "combined"
    (outdir / "pdf").mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "pdf/baseline_comparison_average_rank_combined_colm_style.pdf",
                bbox_inches="tight", pad_inches=0.04, dpi=450)
    plt.close(fig)


def main() -> None:
    set_style()
    for model_key, model_label in MODELS:
        outdir = FIGURES / model_key
        (outdir / "csv").mkdir(parents=True, exist_ok=True)
        (outdir / "pdf").mkdir(parents=True, exist_ok=True)
        plot_average_rank(model_key, model_label)
        print(f"Wrote average-rank figure for {model_label}.")
    plot_combined_average_rank()
    print("Wrote combined average-rank figure.")


if __name__ == "__main__":
    main()
