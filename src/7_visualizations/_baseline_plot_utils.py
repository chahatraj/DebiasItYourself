#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams


METHOD_DISPLAY_ORDER = [
    "CI",
    "SR",
    "PT",
    "IND",
    "PC",
    "OPINION",
    "ACTION",
    "EVENT",
    "ALL STRATEGIES",
]


def _strip_eval_prefixes(text: str) -> str:
    s = str(text).lower()
    prefixes = (
        "crows_pairs_metrics_by_bias_",
        "crows_pairs_metrics_overall_",
        "crows_pairs_scored_",
        "stereoset_metrics_",
        "stereoset_predictions_",
        "stereoset_sentence_scores_",
        "bbq_eval_",
    )
    for p in prefixes:
        if s.startswith(p):
            return s[len(p) :]
    return s


def _strategy_abbrev(text: str) -> str | None:
    s = _strip_eval_prefixes(text)
    if "allstrategies" in s or "all_strategies" in s:
        return "ALL STRATEGIES"
    if "counter_imaging" in s:
        return "CI"
    if "stereotype_replacement" in s:
        return "SR"
    if "perspective_taking" in s:
        return "PT"
    if "individuating" in s:
        return "IND"
    if "process_counter_stereotypic" in s or "process_counterstereotypic" in s:
        return "PC"

    # short strategy tags embedded in run names, e.g. "sr500_..."
    m = re.search(r"(?:^|[_-])(ci|sr|pt|ind|pc)\d+(?:[_-]|$)", s)
    if m:
        return m.group(1).upper()
    return None


def infer_method(source: str) -> str:
    src = str(source)
    if "/" in src:
        prefix = src.split("/", 1)[0]
        if prefix and prefix != "3_baselines":
            return prefix

    fname = Path(src).name.lower()
    m = re.search(r"_llama_[^_]+_(.+?)(?:_all)?\.csv$", fname)
    if m:
        method = m.group(1)
    else:
        method = Path(src).stem

    if method == "selfdebiasing":
        return "self_debiasing"
    return method


def _normalize_bbq_dimension(label: str) -> str:
    text = str(label).strip()
    text = re.sub(r"\.jsonl$", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()

    mapping = {
        "age": "Age",
        "disability status": "Disability Status",
        "gender identity": "Gender Identity",
        "nationality": "Nationality",
        "physical appearance": "Physical Appearance",
        "race ethnicity": "Race Ethnicity",
        "race x ses": "Race X SES",
        "race x gender": "Race X Gender",
        "religion": "Religion",
        "ses": "SES",
        "sexual orientation": "Sexual Orientation",
        "all": "Overall",
        "overall": "Overall",
    }
    return mapping.get(text, text.title())


def load_transposed_long(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "dimension" not in df.columns:
        raise ValueError(f"`dimension` column not found in {csv_path}")

    long_df = df.melt(
        id_vars=["dimension"],
        var_name="source_metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])

    parsed = long_df["source_metric"].str.rsplit("|", n=1, expand=True)
    long_df["source"] = parsed[0]
    long_df["metric"] = parsed[1]
    long_df["method"] = long_df["source"].map(infer_method)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])

    is_bbq = "bbq" in str(csv_path).lower()
    if is_bbq:
        long_df["dimension"] = long_df["dimension"].map(_normalize_bbq_dimension)

    return long_df


def set_plot_style() -> None:
    sns.set_theme(style="white", context="paper")
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#e7e7e7",
            "axes.linewidth": 0.8,
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "figure.dpi": 240,
            "axes.grid": False,
        }
    )


def _pastel_heatmap_cmap(center: float | None):
    if center is not None:
        return sns.blend_palette(["#9ecae1", "#f8fafc", "#f6c5d1"], as_cmap=True)
    return sns.blend_palette(["#fdf2f8", "#f0f9ff", "#ecfeff"], as_cmap=True)


def _pastel_bar_palette(n: int):
    base = ["#b8e1ff", "#d0f4de", "#fde2e4", "#cddafd", "#ffe5b4", "#d9c2ff", "#c7f9cc", "#fbcfe8"]
    if n <= len(base):
        return base[:n]
    reps = (n + len(base) - 1) // len(base)
    return (base * reps)[:n]


def prettify_label(label: str) -> str:
    text = str(label).replace("_", " ").replace("-", " ")
    words = [w for w in text.split() if w]
    return " ".join(
        w.capitalize() if w.lower() not in {"lm", "icat", "ss", "ses"} else w.upper()
        for w in words
    )


def write_interpretation_file(
    out_path: Path,
    dataset_name: str,
    metric: str,
    interpretation: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Dataset: {dataset_name}",
        f"Metric: {metric}",
        "",
        "Interpretation:",
        interpretation.strip(),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _add_interpretation_box(fig: plt.Figure, interpretation: str) -> None:
    fig.text(
        0.01,
        0.01,
        f"Interpretation: {interpretation}",
        ha="left",
        va="bottom",
        fontsize=10.5,
        wrap=True,
        color="#374151",
    )


def plot_metric_heatmap(
    long_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    interpretation: str,
    cmap: str | None = None,
    center: float | None = None,
    method_order: list[str] | None = None,
    prettify_method_labels: bool = True,
    prettify_dimension_labels: bool = True,
) -> None:
    subset = long_df[long_df["metric"] == metric].copy()
    if subset.empty:
        return

    dimension_order = list(long_df["dimension"].drop_duplicates())
    if method_order:
        present = set(subset["method"].unique())
        ordered = [m for m in method_order if m in present]
        extras = sorted([m for m in present if m not in set(ordered)])
        method_order = ordered + extras
    else:
        method_order = sorted(subset["method"].unique())

    pivot = subset.pivot_table(
        index="dimension",
        columns="method",
        values="value",
        aggfunc="mean",
    )
    pivot = pivot.reindex(
        index=[d for d in dimension_order if d in pivot.index],
        columns=method_order,
    )

    if cmap is None:
        cmap = _pastel_heatmap_cmap(center)

    max_abs = float(np.nanmax(np.abs(pivot.to_numpy()))) if not pivot.empty else 0.0
    fmt = ".2f"
    if max_abs < 1.0:
        fmt = ".3f"
    if max_abs < 0.1:
        fmt = ".4f"

    fig_w = max(8, 0.9 * len(method_order) + 3)
    fig_h = max(4.5, 0.58 * len(pivot.index) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=center,
        annot=True,
        fmt=fmt,
        annot_kws={"fontsize": 10.5, "color": "#111111"},
        linewidths=0.0,
        cbar_kws={"label": prettify_label(metric), "shrink": 0.9},
        ax=ax,
    )
    ax.set_title(title, pad=16)
    ax.set_xlabel("Method")
    ax.set_ylabel("Dimension")
    if prettify_method_labels:
        ax.set_xticklabels([prettify_label(c) for c in list(pivot.columns)], rotation=24, ha="right")
    else:
        ax.set_xticklabels([str(c) for c in list(pivot.columns)], rotation=24, ha="right")
    if prettify_dimension_labels:
        ax.set_yticklabels([prettify_label(i) for i in list(pivot.index)], rotation=0)
    else:
        ax.set_yticklabels([str(i) for i in list(pivot.index)], rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _add_interpretation_box(fig, interpretation)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_metric_bar(
    long_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    interpretation: str,
    method_order: list[str] | None = None,
    prettify_method_labels: bool = True,
) -> None:
    subset = long_df[long_df["metric"] == metric].copy()
    if subset.empty:
        return

    agg = subset.groupby("method", as_index=False)["value"].mean()
    if method_order:
        rank = {name: idx for idx, name in enumerate(method_order)}
        agg["_rank"] = agg["method"].map(lambda x: rank.get(x, 10**9))
        agg = agg.sort_values(["_rank", "method"], ascending=[True, True]).drop(columns=["_rank"])
    else:
        agg = agg.sort_values("value", ascending=True)

    fig_h = max(5, 0.58 * len(agg) + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    palette = _pastel_bar_palette(len(agg))
    sns.barplot(
        data=agg,
        y="method",
        x="value",
        hue="method",
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title(title, pad=16)
    ax.set_xlabel(f"Mean {prettify_label(metric)}")
    ax.set_ylabel("Method")
    ticks = ax.get_yticks()
    if prettify_method_labels:
        labels = [prettify_label(v) for v in agg["method"].tolist()]
    else:
        labels = [str(v) for v in agg["method"].tolist()]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#d9dee8")
    sns.despine(ax=ax, left=True, bottom=False)

    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(row["value"], i, f"  {row['value']:.2f}", va="center", fontsize=11, color="#111111")

    _add_interpretation_box(fig, interpretation)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _format_count_tag(n: int) -> str:
    if n < 1000:
        return str(n)
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{int(round(n / 1000.0))}k"


def _size_tag_and_rank(method_key: str) -> tuple[str, int]:
    key = _strip_eval_prefixes(method_key)
    m = re.match(r"^n(\d+)_", key)
    if m:
        n = int(m.group(1))
        return f"({_format_count_tag(n)})", n
    m = re.search(r"(?:^|[_-])(?:ci|sr|pt|ind|pc)(\d+)(?:[_-]|$)", key)
    if m:
        n = int(m.group(1))
        return f"({_format_count_tag(n)})", n
    m = re.search(r"(?:^|[_-])ms[-_]?(\d+)(?:[_-]|$)", key)
    if m:
        n = int(m.group(1))
        return f"({_format_count_tag(n)})", n
    return "", 10**9


def _base_label(method_key: str) -> str:
    key = _strip_eval_prefixes(method_key)
    strat = _strategy_abbrev(key)
    if strat:
        return strat
    if "_strat_ci_" in key:
        return "CI"
    if "_strat_sr_" in key:
        return "SR"
    if "_strat_pt_" in key:
        return "PT"
    if "_strat_ind_" in key:
        return "IND"
    if "_strat_pc_" in key:
        return "PC"
    if "_ver_opinion_" in key:
        return "OPINION"
    if "_ver_action_" in key:
        return "ACTION"
    if "_ver_event_" in key:
        return "EVENT"
    if "_allstrat_allver" in key:
        return "ALL STRATEGIES"
    return key


def _pretty_label(method_key: str) -> str:
    base = _base_label(method_key)
    tag, _ = _size_tag_and_rank(method_key)
    return f"{base} {tag}".strip()


def _order_key(method_key: str) -> tuple[int, int, str]:
    base = _base_label(method_key)
    _tag, size_rank = _size_tag_and_rank(method_key)
    try:
        base_rank = METHOD_DISPLAY_ORDER.index(base)
    except ValueError:
        base_rank = 999
    return (size_rank, base_rank, str(method_key))


def rename_and_order_methods(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "method" not in df.columns:
        return df.copy(), []

    out = df.copy()
    raw_methods = [m for m in out["method"].dropna().unique().tolist()]
    sorted_raw = sorted(raw_methods, key=_order_key)

    mapping = {m: _pretty_label(m) for m in sorted_raw}
    out["method"] = out["method"].map(lambda x: mapping.get(x, x))

    ordered_labels: list[str] = []
    seen: set[str] = set()
    for raw in sorted_raw:
        label = mapping[raw]
        if label not in seen:
            seen.add(label)
            ordered_labels.append(label)

    return out, ordered_labels
