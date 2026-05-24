#!/usr/bin/env python3
"""Qwen versions of the paper figures.

The Llama-8B figures are built from an aggregate CSV. Qwen results currently
live as separate metric files, so this script reads the metric CSVs directly
and writes Qwen-only figures under figures/qwen/.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import use_nimbus_sans
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "figures" / "qwen"
RESULTS = ROOT / "results"
NEW_RESULTS = RESULTS / "new_results"

BASE_ROOT = NEW_RESULTS / "m3_qwen35_27b_base_all5"
IT_ROOT = NEW_RESULTS / "m3_finetune_qwen35_27b_ms500"
M6_ROOT = (
    NEW_RESULTS
    / "m6_qwen35_27b_reasoning_plus_original5"
    / "m6_qwen35_27b_reasoning_plus_original5_ordered_20260504_220932"
    / "m6"
)
ICL_ROOT = (
    NEW_RESULTS
    / "m4_base_icl_zero"
    / "m4_base_icl_zero_allmodels_20260514_005736"
    / "qwen35_27b"
)
ICL_ONE_ROOT = (
    NEW_RESULTS
    / "m4_base_icl_one"
    / "m4_base_icl_one_allmodels_20260514_124333"
    / "qwen35_27b"
)
FT_ICL_ZERO_ROOT = (
    NEW_RESULTS
    / "m4_ft_icl_zero"
    / "m4_ft_icl_zero_allmodels_20260514_143756"
    / "qwen35_27b"
)
FT_ICL_ONE_ROOT = (
    NEW_RESULTS
    / "m4_ft_icl_one"
    / "m4_ft_icl_one_allmodels_20260514_143757"
    / "qwen35_27b"
)
M4_STRATEGY_TAGS = ("sr", "ci", "ind", "pt", "pc")
ICL_REASONING_ROOT = (
    NEW_RESULTS
    / "m4_base_icl_zero"
    / "m4_base_icl_zero_reasoning_allmodels_20260514_011540"
    / "qwen35_27b"
)
BASELINES_ROOT = RESULTS / "11_baselines_qwen35_27b"
BASELINE_RUN_ROOTS = [
    BASELINES_ROOT / "baseline_additions_20260513_20260513_133647",
    *sorted(BASELINES_ROOT.glob("completion_missing*"), reverse=True),
    BASELINES_ROOT / "baselines_qwen35_27b_sep_20260504_101219",
]

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

REASONING = [
    ("arc_challenge", "ARC-Challenge"),
    ("arc_easy", "ARC-Easy"),
    ("balanced_copa", "Balanced COPA"),
]

CORE_METHODS = [
    ("base", "Base Model\nInference"),
    ("diy_it", "DIY IT"),
    ("diy_twopass_no_it", "DIY Two Pass\n(No IT)"),
    ("diy_twopass_it", "DIY Two Pass\n(IT)"),
]

DEBIASING_METHODS = [
    ("base", "Base Model\nInference"),
    ("icl", "ICL"),
    ("diy_it", "DIY IT"),
    ("diy_twopass_no_it", "DIY Two Pass\n(No IT)"),
    ("diy_twopass_it", "DIY Two Pass\n(IT)"),
]

REASONING_METHODS = [
    ("base", "Base Model\nInference"),
    ("icl", "ICL"),
    ("diy_it", "DIY IT"),
    ("diy_twopass_no_it", "DIY Two Pass\n(No IT)"),
    ("diy_twopass_it", "DIY Two Pass\n(IT)"),
]

CORE_COLORS = {
    "base": "#C7CCD8",
    "icl": "#C4B5FD",
    "diy_it": "#7CC7F2",
    "diy_twopass_no_it": "#84D9A5",
    "diy_twopass_it": "#FFB86B",
}

CORE_HATCHES = {
    "base": "",
    "icl": "xx",
    "diy_it": "///",
    "diy_twopass_no_it": "\\\\\\",
    "diy_twopass_it": "...",
}

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
    ("selfdebias_reprompting", "SelfDebias"),
]

INTERVENTIONS = [
    ("sr", "stereotype_replacement", "stereotype-replacement", "Stereotype replacement"),
    ("ind", "individuating", "individuating", "Individuation"),
    ("pt", "perspective_taking", "perspective-taking", "Perspective-taking"),
    ("ci", "counter_imaging", "counter-imaging", "Counter-stereotypic imaging"),
    ("pc", "positive_contact", "positive-contact", "Positive contact"),
]

INTERVENTION_STYLES = {
    "Stereotype replacement": ("#7DD3FC", "///"),
    "Individuation": ("#5EEAD4", "\\\\\\"),
    "Perspective-taking": ("#A7F3D0", "xxx"),
    "Counter-stereotypic imaging": ("#FDBA74", "..."),
    "Positive contact": ("#C4B5FD", "---"),
}

SHOT_METHODS = [
    {"key": "base", "setting": "Base", "shot": "", "label": "Base", "color": "#BBC2CE", "hatch": ""},
    {"key": "icl", "setting": "ICL", "shot": 0, "label": "ICL", "color": "#C4B5FD", "hatch": "xx"},
    {"key": "icl_1", "setting": "ICL", "shot": 1, "label": "ICL\n1", "color": "#A78BFA", "hatch": "xx"},
    {"key": "ft_icl_0", "setting": "IT+ICL", "shot": 0, "label": "IT+ICL\n0", "color": "#FCD34D", "hatch": "++"},
    {"key": "ft_icl_1", "setting": "IT+ICL", "shot": 1, "label": "IT+ICL\n1", "color": "#FBBF24", "hatch": "++"},
    {"key": "it", "setting": "DIY IT", "shot": "", "label": "DIY\nIT", "color": "#7CC7F2", "hatch": "///"},
    {"key": "no_it_0", "setting": "No IT", "shot": 0, "label": "No IT\n0", "color": "#B6EAC7", "hatch": ""},
    {"key": "no_it_1", "setting": "No IT", "shot": 1, "label": "No IT\n1", "color": "#84D9A5", "hatch": "\\\\\\"},
    {"key": "no_it_2", "setting": "No IT", "shot": 2, "label": "No IT\n2", "color": "#50C77B", "hatch": "..."},
    {"key": "it_0", "setting": "IT", "shot": 0, "label": "IT\n0", "color": "#FFD8A7", "hatch": ""},
    {"key": "it_1", "setting": "IT", "shot": 1, "label": "IT\n1", "color": "#FFB86B", "hatch": "///"},
    {"key": "it_2", "setting": "IT", "shot": 2, "label": "IT\n2", "color": "#FF9E45", "hatch": "..."},
]
REASONING_SHOT_METHODS = [
    method for method in SHOT_METHODS if str(method["key"]) not in {"icl_1", "ft_icl_0", "ft_icl_1"}
]

# Styles and labels for the bias-reasoning pareto plot.
PARETO_STYLES = {
    "base": ("#B8BFD6", "s", 104),
    "icl": ("#C4B5FD", "D", 98),
    "baseline": ("#D9DEE9", "o", 94),
    "diy_tune": ("#FF8FAB", "D", 98),
    "diy_twopass": ("#5EEAD4", "D", 98),
    "diy_combo": ("#FCD34D", "D", 98),
    "diy": ("#FF8FAB", "D", 98),
}

PARETO_LABEL_OFFSETS = {
    "BBA": (8, -12),
    "CAL": (8, 7),
    "FairSteer": (8, 13),
    "BiasEdit": (8, -6),
    "LFTF": (8, -2),
    "DPO": (8, 4),
    "PEFT": (8, -6),
    "DebiasLLMs": (8, -8),
    "DebiasNLG": (8, 4),
    "RSB": (-10, 14),
    "SelfDebias": (8, 12),
    "ICL": (8, 10),
    "DIY IT": (10, 1),
    "DIY Two Pass (No IT)": (-10, 14),
    "DIY Two Pass (IT)": (8, 12),
}

PARETO_SHORT_LABELS = {
    "RSB": "RSB",
    "SelfDebias": "SelfDebias",
    "ICL": "ICL",
    "DIY IT": "DIY IT",
    "DIY Two Pass (No IT)": "DIY\nNo IT",
    "DIY Two Pass (IT)": "DIY\nIT",
}


@dataclass(frozen=True)
class PlotConfig:
    slug: str
    title: str
    it_tag: str
    strategy_key: str
    strategy_label: str
    ft_label: str


CONFIGS = [
    PlotConfig(
        "all_strategies_all_versions",
        "All interventions, all-version IT checkpoint",
        "all_allversions",
        "all_strategies",
        "All interventions",
        "all-version",
    ),
    PlotConfig(
        "all_strategies_opinion",
        "All interventions, opinion IT checkpoint",
        "all_opinion",
        "all_strategies",
        "All interventions",
        "opinion",
    ),
    PlotConfig(
        "all_strategies_action",
        "All interventions, action IT checkpoint",
        "all_action",
        "all_strategies",
        "All interventions",
        "action",
    ),
    PlotConfig(
        "all_strategies_event",
        "All interventions, event IT checkpoint",
        "all_event",
        "all_strategies",
        "All interventions",
        "event",
    ),
    PlotConfig(
        "stereotype_replacement_all_versions",
        "Stereotype replacement, matched IT checkpoint",
        "sr",
        "stereotype_replacement",
        "Stereotype replacement",
        "matched all-version",
    ),
    PlotConfig(
        "individuation_all_versions",
        "Individuation, matched IT checkpoint",
        "ind",
        "individuating",
        "Individuation",
        "matched all-version",
    ),
    PlotConfig(
        "perspective_taking_all_versions",
        "Perspective-taking, matched IT checkpoint",
        "pt",
        "perspective_taking",
        "Perspective-taking",
        "matched all-version",
    ),
    PlotConfig(
        "counter_stereotypic_imaging_all_versions",
        "Counter-stereotypic imaging, matched IT checkpoint",
        "ci",
        "counter_imaging",
        "Counter-stereotypic imaging",
        "matched all-version",
    ),
    PlotConfig(
        "positive_contact_all_versions",
        "Positive contact, matched IT checkpoint",
        "pc",
        "positive_contact",
        "Positive contact",
        "matched all-version",
    ),
]

MISSING: list[dict[str, str]] = []


def set_style(kind: str = "bars") -> None:
    base = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if kind == "small":
        base.update(
            {
                "font.size": 8.8,
                "axes.titlesize": 10,
                "axes.labelsize": 9.8,
                "xtick.labelsize": 8.8,
                "ytick.labelsize": 8.8,
                "legend.fontsize": 7.8,
            }
        )
    elif kind == "lollipop":
        base.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12.5,
                "xtick.labelsize": 11.5,
                "ytick.labelsize": 11.5,
                "legend.fontsize": 11.5,
            }
        )
    else:
        base.update(
            {
                "font.size": 11.2,
                "axes.titlesize": 13.2,
                "axes.labelsize": 11.5,
                "xtick.labelsize": 10.5,
                "ytick.labelsize": 10.8,
                "legend.fontsize": 10.4,
                "hatch.linewidth": 0.5,
            }
        )
    use_nimbus_sans(base)


def ensure_dirs() -> None:
    for subdir in ("csv", "pdf"):
        (OUTDIR / subdir).mkdir(parents=True, exist_ok=True)


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_value(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def record_missing(figure: str, method: str, dataset_key: str, panel_label: str, reason: str) -> None:
    MISSING.append(
        {
            "figure": figure,
            "method": method.replace("\n", " "),
            "dataset_key": dataset_key,
            "panel_label": panel_label,
            "reason": reason,
        }
    )


def base_bias_path(dataset_key: str) -> Path | None:
    if dataset_key == "crowspairs":
        return BASE_ROOT / "evalshared/crows_pairs_metrics_overall_m3_qwen35_27b_base_crowspairs.csv"
    if dataset_key == "stereoset":
        return BASE_ROOT / "evalshared/stereoset_metrics_m3_qwen35_27b_base_stereoset.csv"
    if dataset_key == "bbq":
        return BASE_ROOT / "bbq/bbq_metrics_m3_qwen35_27b_base_bbq.csv"
    if dataset_key == "winobias":
        return (
            BASE_ROOT
            / "evalshared/winobias/m3_qwen35_27b_base_winobias/winobias_metrics_overall_m3_qwen35_27b_base_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            BASE_ROOT
            / "evalshared/winogender/m3_qwen35_27b_base_winogender/winogender_metrics_overall_m3_qwen35_27b_base_winogender.csv"
        )
    return None


def it_bias_path(dataset_key: str, tag: str = "all_allversions") -> Path | None:
    if dataset_key == "bbq":
        return IT_ROOT / f"bbq/bbq_metrics_m3_qwen35_27b_ms500_{tag}_bbq.csv"
    if dataset_key == "crowspairs":
        return IT_ROOT / f"evalshared/crows_pairs_metrics_overall_m3_qwen35_27b_ms500_{tag}_crowspairs.csv"
    if dataset_key == "stereoset":
        return IT_ROOT / f"evalshared/stereoset_metrics_m3_qwen35_27b_ms500_{tag}_stereoset.csv"
    if dataset_key == "winobias":
        return (
            IT_ROOT
            / f"evalshared/winobias/m3_qwen35_27b_ms500_{tag}_winobias/winobias_metrics_overall_m3_qwen35_27b_ms500_{tag}_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            IT_ROOT
            / f"evalshared/winogender/m3_qwen35_27b_ms500_{tag}_winogender/winogender_metrics_overall_m3_qwen35_27b_ms500_{tag}_winogender.csv"
        )
    return None


def m6_dir_path(shot: int, checkpoint: str, strategy: str) -> Path:
    shot_word = qwen_shot_word(shot)
    return M6_ROOT / "eval" / f"m6_qwen35_27b_two_pass_{shot_word}__{checkpoint}__{strategy}"


def first_glob(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(root.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def m6_bias_path(dataset_key: str, shot: int, checkpoint: str, strategy: str) -> Path | None:
    run_dir = m6_dir_path(shot, checkpoint, strategy)
    if dataset_key == "bbq":
        return first_glob(run_dir, ["bbq/bbq_metrics.csv"])
    if dataset_key == "crowspairs":
        return first_glob(run_dir, ["crowspairs/*metrics_overall*.csv"])
    if dataset_key == "stereoset":
        return first_glob(run_dir, ["stereoset/stereoset_metrics*.csv"])
    if dataset_key == "winobias":
        return first_glob(run_dir, ["winobias/**/winobias_metrics_overall*.csv"])
    if dataset_key == "winogender":
        return first_glob(run_dir, ["winogender/**/winogender_metrics_overall*.csv"])
    return None


def icl_root_for_shot(shot: int) -> Path:
    return ICL_ONE_ROOT if shot == 1 else ICL_ROOT


def ft_icl_root_for_shot(shot: int) -> Path:
    return FT_ICL_ONE_ROOT if shot == 1 else FT_ICL_ZERO_ROOT


def icl_bias_path(dataset_key: str, shot: int = 0) -> Path | None:
    root = icl_root_for_shot(shot)
    shot_word = "one" if shot == 1 else "zero"
    if dataset_key == "bbq":
        return root / f"bbq/bbq_metrics_m4_baseicl_{shot_word}_qwen35_27b_allstrat_bbq.csv"
    if dataset_key == "crowspairs":
        return root / f"evalshared/crows_pairs_metrics_overall_m4_baseicl_{shot_word}_qwen35_27b_allstrat_crowspairs.csv"
    if dataset_key == "stereoset":
        return root / f"evalshared/stereoset_metrics_m4_baseicl_{shot_word}_qwen35_27b_allstrat_stereoset.csv"
    if dataset_key == "winobias":
        return (
            root
            / f"evalshared/winobias/m4_baseicl_{shot_word}_qwen35_27b_allstrat_winobias/"
            / f"winobias_metrics_overall_m4_baseicl_{shot_word}_qwen35_27b_allstrat_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            root
            / f"evalshared/winogender/m4_baseicl_{shot_word}_qwen35_27b_allstrat_winogender/"
            / f"winogender_metrics_overall_m4_baseicl_{shot_word}_qwen35_27b_allstrat_winogender.csv"
        )
    return None


def ft_icl_bias_path(dataset_key: str, shot: int, tag: str) -> Path | None:
    root = ft_icl_root_for_shot(shot)
    shot_word = "one" if shot == 1 else "zero"
    if dataset_key == "bbq":
        return root / f"bbq/bbq_metrics_m4_fticl_{shot_word}_qwen35_27b_{tag}_bbq.csv"
    if dataset_key == "crowspairs":
        return root / f"evalshared/crows_pairs_metrics_overall_m4_fticl_{shot_word}_qwen35_27b_{tag}_crowspairs.csv"
    if dataset_key == "stereoset":
        return root / f"evalshared/stereoset_metrics_m4_fticl_{shot_word}_qwen35_27b_{tag}_stereoset.csv"
    if dataset_key == "winobias":
        return (
            root
            / f"evalshared/winobias/m4_fticl_{shot_word}_qwen35_27b_{tag}_winobias/"
            / f"winobias_metrics_overall_m4_fticl_{shot_word}_qwen35_27b_{tag}_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            root
            / f"evalshared/winogender/m4_fticl_{shot_word}_qwen35_27b_{tag}_winogender/"
            / f"winogender_metrics_overall_m4_fticl_{shot_word}_qwen35_27b_{tag}_winogender.csv"
        )
    return None


def icl_strategy_bias_path(dataset_key: str, strategy_key: str) -> Path | None:
    if strategy_key == "all_strategies":
        return icl_bias_path(dataset_key)
    tag = next((short for short, key, _, _ in INTERVENTIONS if key == strategy_key), strategy_key)
    if dataset_key == "bbq":
        return ICL_ROOT / f"bbq/bbq_metrics_m4_baseicl_zero_qwen35_27b_{tag}_bbq.csv"
    if dataset_key == "crowspairs":
        return ICL_ROOT / f"evalshared/crows_pairs_metrics_overall_m4_baseicl_zero_qwen35_27b_{tag}_crowspairs.csv"
    if dataset_key == "stereoset":
        return ICL_ROOT / f"evalshared/stereoset_metrics_m4_baseicl_zero_qwen35_27b_{tag}_stereoset.csv"
    if dataset_key == "winobias":
        return (
            ICL_ROOT
            / f"evalshared/winobias/m4_baseicl_zero_qwen35_27b_{tag}_winobias/"
            / f"winobias_metrics_overall_m4_baseicl_zero_qwen35_27b_{tag}_winobias.csv"
        )
    if dataset_key == "winogender":
        return (
            ICL_ROOT
            / f"evalshared/winogender/m4_baseicl_zero_qwen35_27b_{tag}_winogender/"
            / f"winogender_metrics_overall_m4_baseicl_zero_qwen35_27b_{tag}_winogender.csv"
        )
    return None


def m6_strategy_path(dataset_key: str, shot: int, tag: str, strategy_key: str, with_it: bool) -> Path | None:
    if with_it:
        checkpoint = tag
        strategy = "all_strategies" if strategy_key == "all_strategies" else strategy_key
    else:
        checkpoint = f"base_{strategy_key}"
        strategy = strategy_key
    return m6_bias_path(dataset_key, shot, checkpoint, strategy)


def baseline_bias_path(dataset_key: str, method_key: str) -> Path | None:
    lookup_keys = [method_key]
    if method_key == "selfdebias_reprompting":
        lookup_keys.append("self_debiasing")
    method_dirs = [root / key for root in BASELINE_RUN_ROOTS for key in lookup_keys if (root / key).exists()]
    if not method_dirs:
        return None

    def selfdebias_path_matches(path: Path) -> bool:
        text = str(path)
        return any(
            token in text
            for token in ("selfdebiasing_reprompting", "selfdebias_reprompting", "selfdebiasing_all")
        )

    def find_in_dirs(patterns: list[str]) -> Path | None:
        for method_dir in method_dirs:
            for pattern in patterns:
                for path in sorted(method_dir.glob(pattern)):
                    if method_key.startswith("selfdebias") and not selfdebias_path_matches(path):
                        continue
                    if dataset_key == "stereoset" and path.suffix != ".csv":
                        continue
                    return path
        return None

    if dataset_key == "crowspairs":
        return find_in_dirs(["crowspairs/*metrics_overall*.csv"])
    if dataset_key == "stereoset":
        return find_in_dirs(["stereoset/*stereoset*.csv"])
    if dataset_key == "bbq":
        return find_in_dirs(["bbq/*.csv"])
    if dataset_key == "winobias":
        direct = find_in_dirs(["winobias/**/winobias_metrics_overall*.csv"])
        if direct is not None:
            return direct
        if method_key in {"bba", "cal"}:
            alt = RESULTS / "3_baselines" / method_key / "winobias" / f"qwen35_27b_{method_key}_bbq_all"
            candidates = sorted(alt.glob("winobias_metrics_overall*.csv"))
            return candidates[0] if candidates else None
    if dataset_key == "winogender":
        direct = find_in_dirs(["winogender/**/winogender_metrics_overall*.csv"])
        if direct is not None:
            return direct
        if method_key in {"bba", "cal"}:
            alt = RESULTS / "3_baselines" / method_key / "winogender" / f"qwen35_27b_{method_key}_bbq_all"
            candidates = sorted(alt.glob("winogender_metrics_overall*.csv"))
            return candidates[0] if candidates else None
    return None


def read_metric(path: Path, dataset_key: str, metric: str) -> tuple[float, str]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    if dataset_key == "stereoset":
        rows = [
            r
            for r in rows
            if r.get("split") == "overall" and r.get("domain") == "overall"
        ]
        if not rows:
            raise RuntimeError(f"No StereoSet overall row in {path}")

    if dataset_key == "bbq":
        overall = [
            r
            for r in rows
            if r.get("input_file") == "__overall__"
            or r.get("Category", "").lower() == "overall"
            or r.get("Model", "").lower() == "overall"
        ]
        if overall:
            value = safe_float(overall[-1].get(metric))
            if value is None:
                raise KeyError(metric)
            return value, rel(path)

        if "Category" in rows[0] and len(rows) > 1:
            values = [abs(v) for r in rows if (v := safe_float(r.get(metric))) is not None]
            if not values:
                raise KeyError(metric)
            return float(np.mean(values)), f"mean_abs_category:{rel(path)}"

    value = safe_float(rows[-1].get(metric))
    if value is None:
        raise KeyError(f"{metric} missing from {path}")
    return value, rel(path)


def bias_error(value: float, ideal: float) -> float:
    return abs(value - ideal)


def read_bias_source(
    dataset_key: str,
    panel_label: str,
    metric: str,
    metric_label: str,
    ideal: float,
    method_key: str,
    method_label: str,
    path: Path | None,
    figure: str,
) -> dict[str, str] | None:
    if path is None:
        record_missing(figure, method_label, dataset_key, panel_label, "metric file not available")
        return None
    if not path.exists():
        record_missing(figure, method_label, dataset_key, panel_label, f"missing file: {rel(path)}")
        return None
    try:
        native, source = read_metric(path, dataset_key, metric)
    except (KeyError, RuntimeError, ValueError) as exc:
        record_missing(figure, method_label, dataset_key, panel_label, str(exc))
        return None
    return {
        "dataset_key": dataset_key,
        "panel_label": panel_label,
        "method_key": method_key,
        "method_label": method_label.replace("\n", " "),
        "metric": metric,
        "metric_label": metric_label,
        "ideal_value": f"{ideal:.6g}",
        "native_value": f"{native:.8g}",
        "normalized_bias_error_plotted": f"{bias_error(native, ideal):.8g}",
        "metric_source_file": source,
    }


def read_m4_bbq_strategy_mean_source(
    dataset_key: str,
    panel_label: str,
    metric: str,
    metric_label: str,
    ideal: float,
    method_key: str,
    method_label: str,
    figure: str,
) -> dict[str, str] | None:
    if dataset_key != "bbq":
        return None
    if method_key == "icl_1":
        paths = [
            ICL_ONE_ROOT / f"bbq/bbq_metrics_m4_baseicl_one_qwen35_27b_{tag}_bbq.csv"
            for tag in M4_STRATEGY_TAGS
        ]
    elif method_key == "ft_icl_1":
        paths = [
            FT_ICL_ONE_ROOT / f"bbq/bbq_metrics_m4_fticl_one_qwen35_27b_{tag}_bbq.csv"
            for tag in M4_STRATEGY_TAGS
        ]
    else:
        return None

    missing = [rel(path) for path in paths if not path.exists()]
    if missing:
        record_missing(
            figure,
            method_label,
            dataset_key,
            panel_label,
            "missing strategy files for mean fallback: " + ";".join(missing),
        )
        return None

    values = []
    sources = []
    for path in paths:
        try:
            native, source = read_metric(path, dataset_key, metric)
        except (KeyError, RuntimeError, ValueError) as exc:
            record_missing(figure, method_label, dataset_key, panel_label, f"{rel(path)}: {exc}")
            return None
        values.append(native)
        sources.append(source)

    native = float(np.mean(values))
    return {
        "dataset_key": dataset_key,
        "panel_label": panel_label,
        "method_key": method_key,
        "method_label": method_label.replace("\n", " "),
        "metric": metric,
        "metric_label": metric_label,
        "ideal_value": f"{ideal:.6g}",
        "native_value": f"{native:.8g}",
        "normalized_bias_error_plotted": f"{bias_error(native, ideal):.8g}",
        "metric_source_file": f"mean_strategy_metrics:{'|'.join(sources)}",
    }


def read_icl_bias_source(
    dataset_key: str,
    panel_label: str,
    metric: str,
    metric_label: str,
    ideal: float,
    figure: str,
) -> dict[str, str] | None:
    method_key = "icl"
    method_label = "ICL"
    path = icl_bias_path(dataset_key)
    if path is not None and path.exists():
        return read_bias_source(
            dataset_key,
            panel_label,
            metric,
            metric_label,
            ideal,
            method_key,
            method_label,
            path,
            figure,
        )
    if dataset_key != "bbq":
        return read_bias_source(
            dataset_key,
            panel_label,
            metric,
            metric_label,
            ideal,
            method_key,
            method_label,
            path,
            figure,
        )

    strategy_paths = sorted(ICL_ROOT.glob("bbq/bbq_metrics_m4_baseicl_zero_qwen35_27b_*_bbq.csv"))
    values = []
    sources = []
    for strategy_path in strategy_paths:
        if "allstrat" in strategy_path.name:
            continue
        try:
            native, source = read_metric(strategy_path, dataset_key, metric)
        except (KeyError, RuntimeError, ValueError):
            continue
        values.append(native)
        sources.append(source)
    if not values:
        record_missing(figure, method_label, dataset_key, panel_label, "ICL BBQ metric file not available")
        return None
    native = float(np.mean(values))
    return {
        "dataset_key": dataset_key,
        "panel_label": panel_label,
        "method_key": method_key,
        "method_label": method_label,
        "metric": metric,
        "metric_label": metric_label,
        "ideal_value": f"{ideal:.6g}",
        "native_value": f"{native:.8g}",
        "normalized_bias_error_plotted": f"{bias_error(native, ideal):.8g}",
        "metric_source_file": f"mean_available_icl_strategies:{len(sources)}",
    }


def collect_core_bias_records(figure: str = "debiasing_method_bars") -> list[dict[str, str]]:
    records = []
    for dataset_key, panel_label, metric, metric_label, ideal in PANELS:
        for method_key, method_label in DEBIASING_METHODS:
            if method_key == "base":
                path = base_bias_path(dataset_key)
            elif method_key == "icl":
                row = read_icl_bias_source(dataset_key, panel_label, metric, metric_label, ideal, figure)
                if row is not None:
                    records.append(row)
                continue
            elif method_key == "diy_it":
                path = it_bias_path(dataset_key, "all_allversions")
            elif method_key == "diy_twopass_no_it":
                path = m6_bias_path(dataset_key, 0, "base_all_strategies", "all_strategies")
            else:
                path = m6_bias_path(dataset_key, 0, "all_allversions", "all_strategies")
            row = read_bias_source(
                dataset_key,
                panel_label,
                metric,
                metric_label,
                ideal,
                method_key,
                method_label,
                path,
                figure,
            )
            if row is not None:
                records.append(row)
    return records


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def matrix_from_records(
    records: list[dict[str, str]], method_keys: list[str]
) -> np.ndarray:
    values = {
        (r["panel_label"], r["method_key"]): float(r["normalized_bias_error_plotted"])
        for r in records
    }
    matrix = np.full((len(PANELS), len(method_keys)), np.nan)
    for i, (_, panel_label, _, _, _) in enumerate(PANELS):
        for j, method_key in enumerate(method_keys):
            matrix[i, j] = values.get((panel_label, method_key), np.nan)
    return matrix


def annotate_missing_bars(ax, x: np.ndarray, values: np.ndarray) -> None:
    ymin, ymax = ax.get_ylim()
    y = ymin + 0.06 * (ymax - ymin)
    for xi, value in zip(x, values):
        if np.isfinite(value):
            continue
        ax.text(
            xi,
            y,
            "N/A",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#7B8794",
            rotation=90,
            fontweight="medium",
        )


def plot_debiasing_method_bars(records: list[dict[str, str]]) -> None:
    set_style("bars")
    method_keys = [m[0] for m in DEBIASING_METHODS]
    matrix = matrix_from_records(records, method_keys)
    fig, axes = plt.subplots(2, 3, figsize=(12.1, 6.55), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    x = np.arange(len(method_keys))
    tick_labels = ["Base", "ICL", "DIY\nIT", "DIY Two\nPass\n(No IT)", "DIY Two\nPass\n(IT)"]

    for i, (ax, (_, panel_label, _, _, _)) in enumerate(zip(axes, PANELS)):
        values = matrix[i]
        ax.set_facecolor("#FBFCFE")
        bars = ax.bar(
            x,
            values,
            width=0.68,
            color=[CORE_COLORS[k] for k in method_keys],
            edgecolor="#2F3437",
            linewidth=0.75,
        )
        for bar, method_key in zip(bars, method_keys):
            bar.set_hatch(CORE_HATCHES[method_key])
        finite = values[np.isfinite(values)]
        ymax = float(finite.max()) if len(finite) else 1.0
        ax.set_ylim(0, ymax * 1.28 if ymax else 1.0)
        ax.set_title(
            panel_label,
            pad=9,
            fontsize=13,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(facecolor="#EEF2F7", edgecolor="#CBD5E1", boxstyle="round,pad=0.25", linewidth=0.6),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Bias error", fontsize=11, color="#374151", fontweight="semibold")
        ax.grid(axis="y", color="#DDE3EC", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
        ax.tick_params(axis="x", length=0, pad=4, colors="#374151")
        ax.tick_params(axis="y", colors="#374151")
        ymin, ymax = ax.get_ylim()
        offset = 0.028 * (ymax - ymin)
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                format_value(float(value)),
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#1F2937",
            )
        annotate_missing_bars(ax, x, values)

    handles = [
        Patch(
            facecolor=CORE_COLORS[k],
            edgecolor="#2F3437",
            hatch=CORE_HATCHES[k],
            linewidth=0.75,
            label=label.replace("\n", " "),
        )
        for k, label in DEBIASING_METHODS
    ]
    fig.legend(
        handles=handles,
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
    fig.suptitle("Qwen debiasing performance across benchmarks", y=0.992, fontsize=17, fontweight="bold", color="#111827")
    fig.text(
        0.01,
        0.043,
        "Metrics: CrowS stereotype preference, StereoSet SS overall, BBQ ambiguous/disambiguated bias score, WinoBias pro-anti gap, and WinoGender male-female disagreement.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.text(
        0.01,
        0.024,
        "Normalization: bars show distance from the unbiased target, 50 for CrowS/StereoSet and 0 for BBQ/Wino metrics; lower is better. ICL and DIY settings use all bias-reducing interventions.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.865), w_pad=1.45, h_pad=1.35)
    fig.savefig(OUTDIR / "pdf/debiasing_method_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def collect_baseline_records() -> list[dict[str, str]]:
    records = []
    method_order = [("base", "Base Model", "base")]
    method_order.append(("icl", "ICL", "icl"))
    method_order.extend((key, label, "baseline") for key, label in BASELINE_METHODS)
    method_order.extend(
        [
            ("diy_it", "DIY IT", "diy_tune"),
            ("diy_twopass_no_it", "DIY Two Pass (No IT)", "diy_twopass"),
            ("diy_twopass_it", "DIY Two Pass (IT)", "diy_combo"),
        ]
    )
    for dataset_key, panel_label, metric, metric_label, ideal in PANELS:
        for method_key, method_label, group in method_order:
            if method_key == "base":
                path = base_bias_path(dataset_key)
            elif method_key == "icl":
                row = read_icl_bias_source(dataset_key, panel_label, metric, metric_label, ideal, "baseline_comparison_lollipop")
                if row is not None:
                    row["group"] = group
                    records.append(row)
                continue
            elif method_key == "diy_it":
                path = it_bias_path(dataset_key, "all_allversions")
            elif method_key in {"diy_twopass_no_it", "diy_twopass_it"}:
                path = None
            else:
                path = baseline_bias_path(dataset_key, method_key)
            row = read_bias_source(
                dataset_key,
                panel_label,
                metric,
                metric_label,
                ideal,
                method_key,
                method_label,
                path,
                "baseline_comparison_lollipop",
            )
            if row is not None:
                row["group"] = group
                records.append(row)
    return records


def plot_baseline_lollipop(records: list[dict[str, str]]) -> None:
    set_style("lollipop")
    method_order = [("base", "Base Model", "base")]
    method_order.append(("icl", "ICL", "icl"))
    method_order.extend((key, label, "baseline") for key, label in BASELINE_METHODS)
    method_order.extend(
        [
            ("diy_it", "DIY IT", "diy_tune"),
            ("diy_twopass_no_it", "DIY Two Pass (No IT)", "diy_twopass"),
            ("diy_twopass_it", "DIY Two Pass (IT)", "diy_combo"),
        ]
    )
    styles = {
        "base": {"color": "#A9B7D9", "marker": "o", "size": 92, "zorder": 4},
        "icl": {"color": "#C4B5FD", "marker": "D", "size": 108, "zorder": 5},
        "baseline": {"color": "#BFD7FF", "marker": "o", "size": 78, "zorder": 3},
        "diy_tune": {"color": "#7DD3FC", "marker": "D", "size": 112, "zorder": 5},
        "diy_twopass": {"color": "#86EFAC", "marker": "D", "size": 112, "zorder": 5},
        "diy_combo": {"color": "#FDBA74", "marker": "D", "size": 126, "zorder": 6},
    }
    by_panel = {(r["panel_label"], r["method_key"]): r for r in records}
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 12.2), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    y_positions = np.arange(len(method_order)) * 1.62

    for ax_idx, (ax, (_, panel_label, _, _, _)) in enumerate(zip(axes, PANELS)):
        ax.set_facecolor("white")
        finite_values = []
        for y, (method_key, _, group) in zip(y_positions, method_order):
            rec = by_panel.get((panel_label, method_key))
            if rec is None:
                continue
            value = float(rec["normalized_bias_error_plotted"])
            finite_values.append(value)
            style = styles[rec["group"]]
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
        xmax = max(finite_values) if finite_values else 1.0
        ax.set_xlim(0, xmax * 1.36 if xmax else 1.0)
        label_offset = max(xmax * 0.045, 0.025)
        for y, (method_key, _, _) in zip(y_positions, method_order):
            rec = by_panel.get((panel_label, method_key))
            if rec is None:
                ax.text(0, y, "N/A", ha="left", va="center", fontsize=8.8, color="#9AA4B2")
                continue
            value = float(rec["normalized_bias_error_plotted"])
            ax.text(
                value + label_offset,
                y,
                format_value(value),
                ha="left",
                va="center",
                fontsize=9.4,
                fontweight="medium",
                color="#1F2937",
                clip_on=False,
            )
        ax.set_title(
            panel_label,
            pad=8,
            fontsize=15,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(facecolor="#EEF2F7", edgecolor="#CBD5E1", boxstyle="round,pad=0.25", linewidth=0.6),
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A97A8")
        ax.spines["bottom"].set_color("#8A97A8")
        ax.tick_params(axis="x", colors="#374151")
        ax.tick_params(axis="y", length=0, colors="#374151")
        ax.set_xlabel("Bias error", fontsize=12.5, fontweight="semibold")
        ax.set_yticks(y_positions)
        if ax_idx % 3 == 0:
            ax.set_yticklabels([label for _, label, _ in method_order])
            for label in ax.get_yticklabels():
                label.set_fontweight("medium")
        else:
            ax.set_yticklabels([])
        ax.set_ylim(y_positions[-1] + 1.1, y_positions[0] - 1.1)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=styles["base"]["color"], markeredgecolor="#27313A", label="Base model", markersize=7),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=styles["icl"]["color"], markeredgecolor="#27313A", label="ICL", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=styles["baseline"]["color"], markeredgecolor="#27313A", label="Baselines", markersize=7.5),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=styles["diy_tune"]["color"], markeredgecolor="#27313A", label="DIY IT", markersize=8),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=styles["diy_twopass"]["color"], markeredgecolor="#27313A", label="DIY Two Pass (No IT)", markersize=8),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=styles["diy_combo"]["color"], markeredgecolor="#27313A", label="DIY Two Pass (IT)", markersize=8.5),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=6, frameon=True, fancybox=True, framealpha=0.95, edgecolor="#B9C2CF")
    fig.suptitle("Qwen DIY compared with existing debiasing baselines", y=0.99, fontsize=18, fontweight="bold", color="#111827")
    fig.text(
        0.01,
        0.035,
        "Metrics and normalization match the debiasing-performance figure; lower bias error is better. N/A marks unavailable Qwen metric files.",
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0.105, 0.06, 1, 0.875), w_pad=1.35, h_pad=1.3)
    fig.savefig(OUTDIR / "pdf/baseline_comparison_lollipop.pdf", bbox_inches="tight")
    plt.close(fig)


def reasoning_path(method_key: str, dataset_key: str, shot: int = 0, tag: str = "all_allversions") -> Path | None:
    if method_key == "base":
        path = (
            M6_ROOT
            / "reasoning_first_pass"
            / dataset_key
            / f"qwen35_firstpass_base_qwen35_27b_{dataset_key}"
            / f"{dataset_key}_metrics_overall_qwen35_firstpass_base_qwen35_27b_{dataset_key}.csv"
        )
        return path
    if method_key == "icl":
        return (
            ICL_REASONING_ROOT
            / "reasoning"
            / dataset_key
            / f"m4_baseicl_zero_qwen35_27b_allstrat_{dataset_key}"
            / f"{dataset_key}_metrics_overall_m4_baseicl_zero_qwen35_27b_allstrat_{dataset_key}.csv"
        )
    if method_key == "diy_it":
        return (
            IT_ROOT
            / "reasoning"
            / dataset_key
            / f"m3_qwen35_27b_ms500_{tag}_{dataset_key}"
            / f"{dataset_key}_metrics_overall_m3_qwen35_27b_ms500_{tag}_{dataset_key}.csv"
        )
    if method_key == "diy_twopass_no_it":
        shot_word = "zero" if shot == 0 else "one" if shot == 1 else "two"
        return (
            M6_ROOT
            / "eval"
            / f"m6_qwen35_27b_two_pass_{shot_word}__base_all_strategies__all_strategies"
            / dataset_key
            / f"{dataset_key}_metrics_overall.csv"
        )
    if method_key == "diy_twopass_it":
        shot_word = "zero" if shot == 0 else "one" if shot == 1 else "two"
        return (
            M6_ROOT
            / "eval"
            / f"m6_qwen35_27b_two_pass_{shot_word}__all_allversions__all_strategies"
            / dataset_key
            / f"{dataset_key}_metrics_overall.csv"
        )
    return None


def qwen_shot_word(shot: int) -> str:
    return "zero" if shot == 0 else "one" if shot == 1 else "two"


def reasoning_two_pass_path(dataset_key: str, shot: int, checkpoint: str, strategy: str) -> Path:
    return (
        M6_ROOT
        / "eval"
        / f"m6_qwen35_27b_two_pass_{qwen_shot_word(shot)}__{checkpoint}__{strategy}"
        / dataset_key
        / f"{dataset_key}_metrics_overall.csv"
    )


def reasoning_config_path(
    method_key: str,
    dataset_key: str,
    shot: int,
    tag: str,
    strategy_key: str,
) -> Path | None:
    if method_key == "base":
        return reasoning_path("base", dataset_key)
    if method_key == "icl":
        tag = "allstrat" if strategy_key == "all_strategies" else next((short for short, key, _, _ in INTERVENTIONS if key == strategy_key), strategy_key)
        return (
            ICL_REASONING_ROOT
            / "reasoning"
            / dataset_key
            / f"m4_baseicl_zero_qwen35_27b_{tag}_{dataset_key}"
            / f"{dataset_key}_metrics_overall_m4_baseicl_zero_qwen35_27b_{tag}_{dataset_key}.csv"
        )
    if method_key == "it":
        return reasoning_path("diy_it", dataset_key, tag=tag)
    if method_key.startswith("no_it_"):
        checkpoint = f"base_{strategy_key}"
        strategy = strategy_key
        return reasoning_two_pass_path(dataset_key, shot, checkpoint, strategy)
    if method_key.startswith("it_"):
        checkpoint = tag
        strategy = "all_strategies" if strategy_key == "all_strategies" else strategy_key
        return reasoning_two_pass_path(dataset_key, shot, checkpoint, strategy)
    return None


def read_accuracy(path: Path) -> float:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    value = safe_float(rows[-1].get("accuracy"))
    if value is None:
        raise KeyError(f"accuracy missing from {path}")
    return value


def collect_reasoning_records() -> list[dict[str, str]]:
    records = []
    for dataset_key, benchmark_label in REASONING:
        for method_key, method_label in REASONING_METHODS:
            path = reasoning_path(method_key, dataset_key, shot=0)
            if path is None or not path.exists():
                record_missing("reasoning_performance", method_label, dataset_key, benchmark_label, "reasoning metric file not available")
                continue
            try:
                acc = read_accuracy(path)
            except (KeyError, RuntimeError, ValueError) as exc:
                record_missing("reasoning_performance", method_label, dataset_key, benchmark_label, str(exc))
                continue
            records.append(
                {
                    "dataset_key": dataset_key,
                    "benchmark_label": benchmark_label,
                    "method_key": method_key,
                    "method_label": method_label.replace("\n", " "),
                    "metric": "accuracy",
                    "accuracy": f"{acc:.8g}",
                    "accuracy_percent_plotted": f"{acc * 100.0:.8g}",
                    "source_file": rel(path),
                }
            )
    return records


def baseline_reasoning_path(method_key: str, dataset_key: str) -> Path | None:
    lookup_keys = [method_key]
    if method_key == "selfdebias_reprompting":
        lookup_keys.append("self_debiasing")
    method_dirs = [root / key for root in BASELINE_RUN_ROOTS for key in lookup_keys if (root / key).exists()]
    for method_dir in method_dirs:
        path = first_glob(method_dir / dataset_key, [f"**/{dataset_key}_metrics_overall*.csv"])
        if path is not None:
            return path
    return None


def collect_baseline_reasoning_records() -> list[dict[str, str]]:
    records = []
    for dataset_key, benchmark_label in REASONING:
        for method_key, method_label in BASELINE_METHODS:
            path = baseline_reasoning_path(method_key, dataset_key)
            if path is None or not path.exists():
                record_missing("baseline_reasoning_performance", method_label, dataset_key, benchmark_label, "baseline reasoning metric file not available")
                continue
            try:
                acc = read_accuracy(path)
            except (KeyError, RuntimeError, ValueError) as exc:
                record_missing("baseline_reasoning_performance", method_label, dataset_key, benchmark_label, str(exc))
                continue
            records.append(
                {
                    "dataset_key": dataset_key,
                    "benchmark_label": benchmark_label,
                    "method_key": method_key,
                    "method_label": method_label,
                    "metric": "accuracy",
                    "accuracy": f"{acc:.8g}",
                    "accuracy_percent_plotted": f"{acc * 100.0:.8g}",
                    "source_file": rel(path),
                }
            )
    return records


def plot_reasoning(records: list[dict[str, str]]) -> None:
    set_style("bars")
    values = {(r["dataset_key"], r["method_key"]): float(r["accuracy_percent_plotted"]) for r in records}
    fig, ax = plt.subplots(figsize=(10.6, 5.15))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    x = np.arange(len(REASONING))
    width = 0.15
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(REASONING_METHODS))
    for offset, (method_key, method_label) in zip(offsets, REASONING_METHODS):
        ys = [values.get((dataset_key, method_key), np.nan) for dataset_key, _ in REASONING]
        bars = ax.bar(
            x + offset,
            ys,
            width=width,
            label=method_label.replace("\n", " "),
            color=CORE_COLORS[method_key],
            edgecolor="#20252A",
            linewidth=1.05,
            zorder=3,
        )
        for bar in bars:
            bar.set_hatch(CORE_HATCHES[method_key])
        for bar, value in zip(bars, ys):
            if not np.isfinite(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.85, f"{value:.1f}", ha="center", va="bottom", fontsize=10.4, fontweight="bold", color="#111827")
    finite = [v for v in values.values() if np.isfinite(v)]
    ymin = max(0, min(finite) - 6) if finite else 0
    ymax = min(100, max(finite) + 6) if finite else 100
    ax.set_ylim(ymin, ymax)
    ax.set_title("Qwen reasoning benchmark performance", pad=10, fontsize=16.5, fontweight="bold", color="#111827")
    ax.set_ylabel("Accuracy (%)", fontsize=12.8, fontweight="semibold", color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in REASONING], fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=9, colors="#1F2937")
    ax.tick_params(axis="y", colors="#374151")
    ax.grid(axis="y", color="#D7DCE5", linewidth=0.85, linestyle="--", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    handles = [
        Patch(facecolor=CORE_COLORS[method_key], edgecolor="#20252A", hatch=CORE_HATCHES[method_key], linewidth=1.05, label=method_label.replace("\n", " "))
        for method_key, method_label in REASONING_METHODS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, frameon=True, fancybox=True, framealpha=0.97, edgecolor="#B9C2CF", handlelength=1.35, columnspacing=1.0, borderpad=0.55)
    fig.text(
        0.01,
        0.026,
        "Metric: raw benchmark accuracy; higher is better. DIY IT uses the 500-example all-intervention Qwen checkpoint; two-pass settings use all interventions with zero demonstrations.",
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0.015, 0.082, 1, 0.875))
    fig.savefig(OUTDIR / "pdf/reasoning_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def mean_by_method(records: list[dict[str, str]], value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[row["method_key"]].append(float(row[value_key]))
    return {key: float(np.mean(vals)) for key, vals in grouped.items() if vals}


def collect_pareto_records(
    bias_records: list[dict[str, str]], baseline_records: list[dict[str, str]], reasoning_records: list[dict[str, str]]
) -> list[dict[str, str]]:
    bias_by_panel: dict[tuple[str, str], float] = {}
    bias_panels: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    groups: dict[str, str] = {}

    group_defaults = {
        "base": "base",
        "icl": "icl",
        "diy_it": "diy_tune",
        "diy_twopass_no_it": "diy_twopass",
        "diy_twopass_it": "diy_combo",
    }
    label_defaults = {key: label.replace("\n", " ") for key, label in REASONING_METHODS}

    # Use both sources: baseline_records has existing baselines, while
    # bias_records has partial Qwen two-pass DIY rows that are absent from the
    # baseline-comparison table until every panel has finished.
    for row in [*baseline_records, *bias_records]:
        key = row["method_key"]
        panel = row["panel_label"]
        bias_by_panel[(key, panel)] = float(row["normalized_bias_error_plotted"])
        bias_panels[key].add(row["panel_label"])
        labels[key] = row.get("method_label", label_defaults.get(key, key)).replace("\n", " ")
        groups[key] = row.get("group", group_defaults.get(key, "baseline"))
    reason_grouped: dict[str, list[float]] = defaultdict(list)
    for row in reasoning_records:
        reason_grouped[row["method_key"]].append(float(row["accuracy_percent_plotted"]))
    records = []
    for method_key in sorted(set(labels) | set(reason_grouped), key=lambda k: (groups.get(k, ""), labels.get(k, k))):
        bvals = [value for (key, _), value in bias_by_panel.items() if key == method_key]
        rvals = list(reason_grouped.get(method_key, []))
        reasoning_source_scope = "model_specific" if rvals else ""
        n_bias = len(bias_panels.get(method_key, set()))
        n_reasoning = len(rvals)
        records.append(
            {
                "method_key": method_key,
                "method_label": labels.get(method_key, dict(REASONING_METHODS).get(method_key, method_key).replace("\n", " ")),
                "group": groups.get(method_key, "diy" if method_key.startswith("diy") else "baseline"),
                "mean_bias_error": f"{float(np.mean(bvals)):.8g}" if bvals else "",
                "mean_reasoning_accuracy": f"{float(np.mean(rvals)):.8g}" if rvals else "",
                "n_bias_panels": str(n_bias),
                "n_reasoning_benchmarks": str(n_reasoning),
                "reasoning_source_scope": reasoning_source_scope,
                "plotted": str(bool(bvals and rvals)),
            }
        )
    return records


def plot_pareto(records: list[dict[str, str]]) -> None:
    set_style("small")
    points = [
        r
        for r in records
        if r.get("mean_bias_error")
        and r.get("mean_reasoning_accuracy")
        and r.get("group") in {"baseline", "icl", "diy_tune", "diy_twopass", "diy_combo"}
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    if len(points) >= 2:
        sorted_points = sorted(points, key=lambda r: float(r["mean_bias_error"]))
        ax.plot(
            [float(r["mean_bias_error"]) for r in sorted_points],
            [float(r["mean_reasoning_accuracy"]) for r in sorted_points],
            color="#111827",
            linestyle=(0, (1.4, 2.4)),
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )
    for row in points:
        group = row["group"]
        color, marker, size = PARETO_STYLES.get(group, PARETO_STYLES["baseline"])
        x = float(row["mean_bias_error"])
        y = float(row["mean_reasoning_accuracy"])
        ax.scatter(x, y, s=size, marker=marker, facecolor=color, edgecolor="#20252A", linewidth=1.05, alpha=0.96, zorder=5)
        dx, dy = PARETO_LABEL_OFFSETS.get(row["method_label"], (0.16, 0.24))
        label = PARETO_SHORT_LABELS.get(row["method_label"], row["method_label"])
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="bottom" if dy > 0 else "top" if dy < 0 else "center",
            fontsize=6.8,
            color="#374151",
            linespacing=0.95,
            annotation_clip=False,
        )

    if points:
        xs = [float(r["mean_bias_error"]) for r in points]
        ys = [float(r["mean_reasoning_accuracy"]) for r in points]
        ax.set_xlim(max(0, min(xs) - 0.55), max(xs) + 0.75)
        ax.set_ylim(min(ys) - 2.0, max(ys) + 1.5)
    else:
        ax.text(0.5, 0.55, "No Qwen methods have both bias and reasoning metrics.", ha="center", va="center", transform=ax.transAxes, fontsize=10.5, color="#374151")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_xlabel("Mean bias error", fontweight="semibold")
    ax.set_ylabel("Reasoning accuracy (%)", fontweight="semibold")
    ax.grid(color="#E4E8F0", linestyle=":", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")
    ax.tick_params(colors="#374151")
    fig.text(0.5, 0.982, "Bias and reasoning performance", ha="center", va="top", fontsize=10.2, fontweight="semibold", color="#111827")
    fig.text(
        0.01,
        0.036,
        "Bias error averages available normalized bias metrics; reasoning averages ARC-Challenge, ARC-Easy, and Balanced COPA.\n"
        "Lower bias and higher reasoning accuracy are better. Existing-baseline reasoning uses Qwen baseline utility runs when available.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#374151",
        linespacing=1.2,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PARETO_STYLES["baseline"][0],
            markeredgecolor="#5B6472",
            markeredgewidth=1.0,
            markersize=5.6,
            label="Baselines",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=PARETO_STYLES["icl"][0],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.3,
            label="ICL",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=PARETO_STYLES["diy_tune"][0],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.3,
            label="DIY IT",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=PARETO_STYLES["diy_twopass"][0],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.3,
            label="DIY Two Pass (No IT)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=PARETO_STYLES["diy_combo"][0],
            markeredgecolor="#20252A",
            markeredgewidth=1.2,
            markersize=6.3,
            label="DIY Two Pass (IT)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#B9C2CF",
        borderpad=0.42,
        columnspacing=0.8,
        handlelength=1.22,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.75)
    fig.savefig(OUTDIR / "pdf/bias_reasoning_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def collect_intervention_records() -> list[dict[str, str]]:
    records = []
    for dataset_key, panel_label, metric, metric_label, ideal in PANELS:
        base = read_bias_source(dataset_key, panel_label, metric, metric_label, ideal, "base", "Base Model", base_bias_path(dataset_key), "intervention_ablation")
        if base:
            records.append({**base, "method": "Base Model", "intervention_key": "__base__", "intervention_label": "Base Model"})
        for tag, intervention_key, _, intervention_label in INTERVENTIONS:
            row = read_bias_source(
                dataset_key,
                panel_label,
                metric,
                metric_label,
                ideal,
                f"icl_{intervention_key}",
                "ICL",
                icl_strategy_bias_path(dataset_key, intervention_key),
                "intervention_ablation",
            )
            if row:
                records.append({**row, "method": "ICL", "intervention_key": intervention_key, "intervention_label": intervention_label})
            row = read_bias_source(
                dataset_key,
                panel_label,
                metric,
                metric_label,
                ideal,
                f"diy_it_{intervention_key}",
                "DIY IT",
                it_bias_path(dataset_key, tag),
                "intervention_ablation",
            )
            if row:
                records.append({**row, "method": "DIY IT", "intervention_key": intervention_key, "intervention_label": intervention_label})
            for method, method_key in (
                ("DIY Two Pass (No IT)", "diy_twopass_no_it"),
                ("DIY Two Pass (IT)", "diy_twopass_it"),
            ):
                with_it = method_key == "diy_twopass_it"
                tag_for_method = tag if with_it else "all_allversions"
                row = read_bias_source(
                    dataset_key,
                    panel_label,
                    metric,
                    metric_label,
                    ideal,
                    f"{method_key}_{intervention_key}",
                    method,
                    m6_strategy_path(dataset_key, 0, tag_for_method, intervention_key, with_it=with_it),
                    "intervention_ablation",
                )
                if row:
                    records.append({**row, "method": method, "intervention_key": intervention_key, "intervention_label": intervention_label})
    return records


def plot_intervention(records: list[dict[str, str]]) -> None:
    set_style("bars")
    values = {
        (r["method"], r["intervention_key"], r["panel_label"]): float(r["normalized_bias_error_plotted"])
        for r in records
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.9), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    methods = ["ICL", "DIY IT", "DIY Two Pass (No IT)", "DIY Two Pass (IT)"]
    x = np.arange(len(methods))
    width = 0.13
    offsets = np.linspace(-2 * width, 2 * width, len(INTERVENTIONS))
    for ax, (_, panel_label, _, _, _) in zip(axes, PANELS):
        ax.set_facecolor("#FCFCFD")
        base_error = values.get(("Base Model", "__base__", panel_label))
        panel_values = []
        for offset, (_, intervention_key, _, intervention_label) in zip(offsets, INTERVENTIONS):
            color, hatch = INTERVENTION_STYLES[intervention_label]
            bar_values = []
            for method in methods:
                value = values.get((method, intervention_key, panel_label))
                bar_values.append(base_error - value if base_error is not None and value is not None else np.nan)
            panel_values.extend([v for v in bar_values if np.isfinite(v)])
            bars = ax.bar(x + offset, bar_values, width=width, color=color, edgecolor="#111827", linewidth=0.75, hatch=hatch, zorder=3)
        ymin = min(0.0, min(panel_values) if panel_values else 0.0)
        ymax = max(panel_values) if panel_values else 1.0
        pad = 0.16 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - pad * 0.25, ymax + pad)
        ax.set_title(panel_label, pad=8, fontweight="bold", color="#111827")
        ax.set_xticks(x)
        ax.set_xticklabels(["ICL", "DIY IT", "DIY Two Pass\n(No IT)", "DIY Two Pass\n(IT)"])
        ax.grid(axis="y", color="#E5E7EB", linestyle=":", linewidth=0.75, alpha=0.95)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="x", length=0, pad=5, colors="#374151")
        ax.tick_params(axis="y", colors="#374151")
    handles = [
        Patch(facecolor=color, edgecolor="#111827", hatch=hatch, linewidth=0.9, label=label)
        for _, _, _, label in INTERVENTIONS
        for color, hatch in [INTERVENTION_STYLES[label]]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=True, fancybox=True, framealpha=0.96, edgecolor="#B9C2CF", borderpad=0.45, columnspacing=1.0, handlelength=1.6)
    fig.text(0.5, 0.985, "Qwen bias error reduction by intervention strategy", ha="center", va="top", fontsize=16.5, fontweight="bold", color="#111827")
    fig.text(
        0.012,
        0.028,
        "Bars show absolute reduction from the base model on each normalized bias metric; higher is better. Missing Qwen cells are left blank.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#374151",
    )
    fig.supxlabel("DIY method", y=0.07, fontsize=11, fontweight="semibold")
    fig.supylabel("Bias error reduction", x=0.028, fontsize=11, fontweight="semibold")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.78, wspace=0.24, hspace=0.42)
    fig.savefig(OUTDIR / "pdf/intervention_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def collect_shot_records(tag: str, figure: str, title: str = "") -> list[dict[str, str]]:
    records = []
    config = next((cfg for cfg in CONFIGS if cfg.it_tag == tag), None)
    strategy_key = config.strategy_key if config is not None else "all_strategies"
    for dataset_key, panel_label, metric, metric_label, ideal in PANELS:
        sources = {
            "base": ("Base Model Inference", base_bias_path(dataset_key)),
            "it": ("DIY IT", it_bias_path(dataset_key, tag)),
        }
        for method in SHOT_METHODS:
            key = str(method["key"])
            if key in sources:
                label, path = sources[key]
                row = read_bias_source(dataset_key, panel_label, metric, metric_label, ideal, key, label, path, figure)
                if row:
                    row["setting"] = str(method["setting"])
                    row["shot"] = str(method["shot"])
                    row["figure_title"] = title
                    records.append(row)
            elif key == "icl":
                if strategy_key == "all_strategies":
                    row = read_icl_bias_source(dataset_key, panel_label, metric, metric_label, ideal, figure)
                else:
                    row = read_bias_source(
                        dataset_key,
                        panel_label,
                        metric,
                        metric_label,
                        ideal,
                        key,
                        "ICL",
                        icl_strategy_bias_path(dataset_key, strategy_key),
                        figure,
                    )
                if row:
                    row["setting"] = str(method["setting"])
                    row["shot"] = str(method["shot"])
                    row["figure_title"] = title
                    records.append(row)
            elif key == "icl_1":
                row = read_m4_bbq_strategy_mean_source(dataset_key, panel_label, metric, metric_label, ideal, key, "ICL 1", figure)
                if row:
                    row["setting"] = str(method["setting"])
                    row["shot"] = str(method["shot"])
                    row["figure_title"] = title
                    records.append(row)
                    continue
                row = read_bias_source(
                    dataset_key,
                    panel_label,
                    metric,
                    metric_label,
                    ideal,
                    key,
                    "ICL 1",
                    icl_bias_path(dataset_key, shot=1),
                    figure,
                )
                if row:
                    row["setting"] = str(method["setting"])
                    row["shot"] = str(method["shot"])
                    row["figure_title"] = title
                    records.append(row)
            elif key.startswith("ft_icl_"):
                shot = int(key.rsplit("_", 1)[1])
                label = str(method["label"]).replace("\n", " ")
                if shot == 1 and tag == "all_allversions":
                    row = read_m4_bbq_strategy_mean_source(dataset_key, panel_label, metric, metric_label, ideal, key, label, figure)
                    if row:
                        row["setting"] = str(method["setting"])
                        row["shot"] = str(method["shot"])
                        row["figure_title"] = title
                        records.append(row)
                        continue
                row = read_bias_source(
                    dataset_key,
                    panel_label,
                    metric,
                    metric_label,
                    ideal,
                    key,
                    label,
                    ft_icl_bias_path(dataset_key, shot, tag),
                    figure,
                )
                if row:
                    row["setting"] = str(method["setting"])
                    row["shot"] = str(method["shot"])
                    row["figure_title"] = title
                    records.append(row)
            elif key.startswith("no_it_"):
                shot = int(key.rsplit("_", 1)[1])
                label = str(method["label"]).replace("\n", " ")
                row = read_bias_source(
                    dataset_key,
                    panel_label,
                    metric,
                    metric_label,
                    ideal,
                    key,
                    label,
                    m6_strategy_path(dataset_key, shot, tag, strategy_key, with_it=False),
                    figure,
                )
                if row:
                    row["setting"] = str(method["setting"])
                    row["shot"] = str(method["shot"])
                    row["figure_title"] = title
                    records.append(row)
            elif key.startswith("it_"):
                shot = int(key.rsplit("_", 1)[1])
                label = str(method["label"]).replace("\n", " ")
                row = read_bias_source(
                    dataset_key,
                    panel_label,
                    metric,
                    metric_label,
                    ideal,
                    key,
                    label,
                    m6_strategy_path(dataset_key, shot, tag, strategy_key, with_it=True),
                    figure,
                )
                if row:
                    row["setting"] = str(method["setting"])
                    row["shot"] = str(method["shot"])
                    row["figure_title"] = title
                    records.append(row)
            else:
                record_missing(figure, str(method["label"]).replace("\n", " "), dataset_key, panel_label, "Qwen two-pass bias metric file not available")
    return records


def plot_shot_matrix(records: list[dict[str, str]], filename: str, title: str, subtitle: str) -> None:
    set_style("bars")
    method_keys = [str(m["key"]) for m in SHOT_METHODS]
    values = {(r["panel_label"], r["method_key"]): float(r["normalized_bias_error_plotted"]) for r in records}
    matrix = np.full((len(PANELS), len(method_keys)), np.nan)
    for i, (_, panel_label, _, _, _) in enumerate(PANELS):
        for j, key in enumerate(method_keys):
            matrix[i, j] = values.get((panel_label, key), np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.7), sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    x = np.arange(len(SHOT_METHODS))
    labels = [str(m["label"]) for m in SHOT_METHODS]
    colors = [str(m["color"]) for m in SHOT_METHODS]
    hatches = [str(m["hatch"]) for m in SHOT_METHODS]
    for i, (ax, (_, panel_label, _, _, _)) in enumerate(zip(axes, PANELS)):
        row_values = matrix[i]
        ax.set_facecolor("#FCFCFD")
        ax.axvspan(2.55, 5.45, color="#ECF9F0", alpha=0.62, zorder=0)
        ax.axvspan(5.55, 8.45, color="#FFF3E4", alpha=0.72, zorder=0)
        ax.axvline(2.5, color="#9AA4B2", linewidth=0.9, alpha=0.8)
        ax.axvline(5.5, color="#9AA4B2", linewidth=0.9, alpha=0.8)
        bars = ax.bar(x, row_values, width=0.68, color=colors, edgecolor="#252A2E", linewidth=0.95, zorder=3)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        finite = row_values[np.isfinite(row_values)]
        ymax = float(finite.max()) if len(finite) else 1.0
        ax.set_ylim(0, ymax * 1.25 if ymax else 1.0)
        ax.set_title(panel_label, pad=9, fontsize=14, fontweight="bold", color="#1F2937", bbox=dict(facecolor="#F1F5F9", edgecolor="#CBD5E1", boxstyle="round,pad=0.24", linewidth=0.7))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if i % 3 == 0:
            ax.set_ylabel("Bias error", fontsize=11.5, color="#374151", fontweight="semibold")
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
        for bar, value in zip(bars, row_values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + offset, format_value(float(value)), ha="center", va="bottom", fontsize=9.5, color="#1F2937")
        annotate_missing_bars(ax, x, row_values)
        ax.text(4.0, ymax * 0.975, "Two Pass (No IT)", ha="center", va="top", fontsize=10.2, color="#17623A", fontweight="semibold")
        ax.text(7.0, ymax * 0.975, "Two Pass (IT)", ha="center", va="top", fontsize=10.2, color="#8A4A12", fontweight="semibold")
    fig.suptitle(title, y=0.978, fontsize=18, fontweight="bold", color="#111827")
    fig.text(0.01, 0.049, subtitle, ha="left", va="bottom", fontsize=9.3, color="#374151")
    fig.text(
        0.01,
        0.028,
        "Metrics and normalization match the main debiasing plot. Shot labels 0/1/2 refer to two-pass demonstrations; N/A marks unavailable Qwen bias files.",
        ha="left",
        va="bottom",
        fontsize=9.3,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.91), w_pad=1.25, h_pad=1.35)
    fig.savefig(OUTDIR / f"pdf/{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def collect_reasoning_shot_records(config: PlotConfig, figure: str) -> list[dict[str, str]]:
    records = []
    for dataset_key, benchmark_label in REASONING:
        for method in REASONING_SHOT_METHODS:
            method_key = str(method["key"])
            shot = int(method["shot"]) if str(method["shot"]).isdigit() else 0
            path = reasoning_config_path(method_key, dataset_key, shot, config.it_tag, config.strategy_key)
            method_label = str(method["label"]).replace("\n", " ")
            if path is None or not path.exists():
                record_missing(figure, method_label, dataset_key, benchmark_label, "reasoning metric file not available")
                continue
            try:
                acc = read_accuracy(path)
            except (KeyError, RuntimeError, ValueError) as exc:
                record_missing(figure, method_label, dataset_key, benchmark_label, str(exc))
                continue
            records.append(
                {
                    "dataset_key": dataset_key,
                    "benchmark_label": benchmark_label,
                    "method_key": method_key,
                    "method_label": method_label,
                    "setting": str(method["setting"]),
                    "shot": str(method["shot"]),
                    "strategy_setting": config.strategy_label,
                    "it_checkpoint": config.ft_label,
                    "metric": "accuracy",
                    "accuracy": f"{acc:.8g}",
                    "accuracy_percent_plotted": f"{acc * 100.0:.8g}",
                    "source_file": rel(path),
                }
            )
    return records


def plot_reasoning_shot_matrix(records: list[dict[str, str]], filename: str, title: str, subtitle: str) -> None:
    set_style("bars")
    method_keys = [str(m["key"]) for m in REASONING_SHOT_METHODS]
    values = {(r["benchmark_label"], r["method_key"]): float(r["accuracy_percent_plotted"]) for r in records}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.45), sharey=False)
    fig.patch.set_facecolor("white")
    x = np.arange(len(REASONING_SHOT_METHODS))
    labels = [str(m["label"]) for m in REASONING_SHOT_METHODS]
    colors = [str(m["color"]) for m in REASONING_SHOT_METHODS]
    hatches = [str(m["hatch"]) for m in REASONING_SHOT_METHODS]

    for ax, (_, benchmark_label) in zip(axes, REASONING):
        row_values = np.array([values.get((benchmark_label, key), np.nan) for key in method_keys])
        ax.set_facecolor("#FCFCFD")
        ax.axvspan(2.55, 5.45, color="#ECF9F0", alpha=0.62, zorder=0)
        ax.axvspan(5.55, 8.45, color="#FFF3E4", alpha=0.72, zorder=0)
        ax.axvline(2.5, color="#9AA4B2", linewidth=0.9, alpha=0.8)
        ax.axvline(5.5, color="#9AA4B2", linewidth=0.9, alpha=0.8)
        bars = ax.bar(x, row_values, width=0.68, color=colors, edgecolor="#252A2E", linewidth=0.95, zorder=3)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        finite = row_values[np.isfinite(row_values)]
        ymin = max(0.0, float(finite.min()) - 5.0) if len(finite) else 0.0
        ymax = min(100.0, float(finite.max()) + 6.0) if len(finite) else 100.0
        if ymax <= ymin:
            ymax = ymin + 10.0
        ax.set_ylim(ymin, ymax)
        ax.set_title(
            benchmark_label,
            pad=9,
            fontsize=14,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(facecolor="#F1F5F9", edgecolor="#CBD5E1", boxstyle="round,pad=0.24", linewidth=0.7),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy (%)", fontsize=11.5, color="#374151", fontweight="semibold")
        ax.grid(axis="y", color="#DCE2EA", linewidth=0.85, linestyle="--", alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")
        ax.tick_params(axis="x", length=0, pad=5, colors="#374151")
        ax.tick_params(axis="y", colors="#374151")
        ymin, ymax = ax.get_ylim()
        offset = 0.025 * (ymax - ymin)
        for bar, value in zip(bars, row_values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + offset,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color="#1F2937",
                )
        annotate_missing_bars(ax, x, row_values)
        ax.text(4.0, ymax - 0.035 * (ymax - ymin), "Two Pass (No IT)", ha="center", va="top", fontsize=10.2, color="#17623A", fontweight="semibold")
        ax.text(7.0, ymax - 0.035 * (ymax - ymin), "Two Pass (IT)", ha="center", va="top", fontsize=10.2, color="#8A4A12", fontweight="semibold")

    fig.suptitle(title, y=0.985, fontsize=17, fontweight="bold", color="#111827")
    fig.text(0.01, 0.052, subtitle, ha="left", va="bottom", fontsize=9.3, color="#374151")
    fig.text(
        0.01,
        0.029,
        "Metric: raw benchmark accuracy; higher is better. Shot labels 0/1/2 refer to two-pass demonstrations; N/A marks unavailable Qwen reasoning files.",
        ha="left",
        va="bottom",
        fontsize=9.3,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.91), w_pad=1.25)
    fig.savefig(OUTDIR / f"pdf/{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_missing_report() -> None:
    if not MISSING:
        return
    # Deduplicate while preserving order.
    seen = set()
    rows = []
    for row in MISSING:
        key = tuple(row.items())
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    write_csv(OUTDIR / "csv/missing_data_report.csv", rows)


def main() -> None:
    ensure_dirs()

    core_bias = collect_core_bias_records()
    write_csv(OUTDIR / "csv/debiasing_method_bars_data.csv", core_bias)
    plot_debiasing_method_bars(core_bias)

    baseline_records = collect_baseline_records()
    write_csv(OUTDIR / "csv/baseline_comparison_lollipop_data.csv", baseline_records)
    plot_baseline_lollipop(baseline_records)

    reasoning_records = collect_reasoning_records()
    write_csv(OUTDIR / "csv/reasoning_performance_data.csv", reasoning_records)
    plot_reasoning(reasoning_records)

    baseline_reasoning_records = collect_baseline_reasoning_records()
    write_csv(OUTDIR / "csv/baseline_reasoning_performance_data.csv", baseline_reasoning_records)

    pareto_records = collect_pareto_records(core_bias, baseline_records, reasoning_records + baseline_reasoning_records)
    write_csv(OUTDIR / "csv/bias_reasoning_pareto_data.csv", pareto_records)
    plot_pareto(pareto_records)

    intervention_records = collect_intervention_records()
    write_csv(OUTDIR / "csv/intervention_ablation_data.csv", intervention_records)
    plot_intervention(intervention_records)

    shot_records = collect_shot_records("all_allversions", "debiasing_method_bars_by_shot", "All interventions, all-version IT checkpoint")
    write_csv(OUTDIR / "csv/debiasing_method_bars_by_shot_data.csv", shot_records)
    plot_shot_matrix(
        shot_records,
        "debiasing_method_bars_by_shot",
        "Qwen debiasing performance by two-pass shot setting",
        "Strategy setting: all interventions. IT checkpoint: all-version.",
    )

    config_rows = []
    for config in CONFIGS:
        records = collect_shot_records(config.it_tag, f"debiasing_method_bars_by_shot_{config.slug}", config.title)
        config_rows.extend(records)
        plot_shot_matrix(
            records,
            f"debiasing_method_bars_by_shot_{config.slug}",
            f"Qwen {config.title}",
            f"Strategy setting: {config.strategy_label}. IT checkpoint: {config.ft_label}.",
        )
    write_csv(OUTDIR / "csv/debiasing_method_bars_by_shot_configs_data.csv", config_rows)

    reasoning_shot_config = CONFIGS[0]
    reasoning_shot_records = collect_reasoning_shot_records(reasoning_shot_config, "reasoning_performance_by_shot")
    write_csv(OUTDIR / "csv/reasoning_performance_by_shot_data.csv", reasoning_shot_records)
    plot_reasoning_shot_matrix(
        reasoning_shot_records,
        "reasoning_performance_by_shot",
        "Qwen reasoning performance by two-pass shot setting",
        "Strategy setting: all interventions. IT checkpoint: all-version.",
    )

    reasoning_config_rows = []
    for config in CONFIGS:
        records = collect_reasoning_shot_records(config, f"reasoning_performance_by_shot_{config.slug}")
        reasoning_config_rows.extend(records)
        plot_reasoning_shot_matrix(
            records,
            f"reasoning_performance_by_shot_{config.slug}",
            f"Qwen reasoning: {config.title}",
            f"Strategy setting: {config.strategy_label}. IT checkpoint: {config.ft_label}.",
        )
    write_csv(OUTDIR / "csv/reasoning_performance_by_shot_configs_data.csv", reasoning_config_rows)
    write_missing_report()


if __name__ == "__main__":
    main()
