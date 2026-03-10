#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_LABELS = {
    "stereotype_replacement": "SR",
    "counter_imaging": "CI",
    "individuating": "IND",
    "perspective_taking": "PT",
    "positive_contact": "PC",
}

STRATEGY_ORDER = [
    "stereotype_replacement",
    "counter_imaging",
    "individuating",
    "perspective_taking",
    "positive_contact",
]

MODE_ORDER = ["learnable", "fixed"]


def _mode_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_crowspairs(results_root: Path) -> pd.DataFrame:
    rows = []
    base = results_root / "crowspairs"
    for mode_dir in _mode_dirs(base):
        mode = mode_dir.name
        for strat_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()]):
            strategy = strat_dir.name
            files = sorted(strat_dir.glob("crows_pairs_metrics_overall_*.csv"))
            if not files:
                continue
            df = pd.read_csv(files[0])
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row["mode"] = mode
            row["strategy_name"] = strategy
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["strategy_label"] = out["strategy_name"].map(lambda x: METRIC_LABELS.get(x, x))
    return out


def load_stereoset(results_root: Path) -> pd.DataFrame:
    rows = []
    base = results_root / "stereoset"
    for mode_dir in _mode_dirs(base):
        mode = mode_dir.name
        for strat_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()]):
            strategy = strat_dir.name
            files = sorted(strat_dir.glob("stereoset_metrics_*.csv"))
            if not files:
                continue
            df = pd.read_csv(files[0])
            if df.empty:
                continue
            if {"split", "domain"}.issubset(df.columns):
                pick = df[(df["split"] == "overall") & (df["domain"] == "overall")]
                if pick.empty:
                    pick = df.head(1)
            else:
                pick = df.head(1)
            row = pick.iloc[0].to_dict()
            row["mode"] = mode
            row["strategy_name"] = strategy
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["strategy_label"] = out["strategy_name"].map(lambda x: METRIC_LABELS.get(x, x))
    return out


def load_bbq(results_root: Path) -> pd.DataFrame:
    rows = []
    base = results_root / "bbq"
    for mode_dir in _mode_dirs(base):
        mode = mode_dir.name
        for csv_path in sorted(mode_dir.glob("*.csv")):
            strategy = csv_path.stem
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            if "input_file" in df.columns and (df["input_file"] == "__overall__").any():
                pick = df[df["input_file"] == "__overall__"].iloc[0]
            else:
                numeric_cols = [
                    c for c in [
                        "Accuracy",
                        "Accuracy_ambig",
                        "Accuracy_disambig",
                        "Bias_score_disambig",
                        "Bias_score_ambig",
                    ]
                    if c in df.columns
                ]
                agg: Dict[str, float] = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in numeric_cols}
                pick = pd.Series(agg)

            row = pick.to_dict()
            row["mode"] = mode
            row["strategy_name"] = strategy
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["strategy_label"] = out["strategy_name"].map(lambda x: METRIC_LABELS.get(x, x))
        if "Bias_score_disambig" in out.columns:
            out["Abs_Bias_score_disambig"] = out["Bias_score_disambig"].abs()
        if "Bias_score_ambig" in out.columns:
            out["Abs_Bias_score_ambig"] = out["Bias_score_ambig"].abs()
    return out


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["mode"] = pd.Categorical(d["mode"], categories=MODE_ORDER, ordered=True)
    d["strategy_name"] = pd.Categorical(d["strategy_name"], categories=STRATEGY_ORDER, ordered=True)
    d = d.sort_values(["strategy_name", "mode"]).reset_index(drop=True)
    return d


def _grouped_bars(df: pd.DataFrame, metrics: List[str], title: str, out_path: Path) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5), squeeze=False)
    axes = axes[0]

    palette = {"learnable": "#2E8B57", "fixed": "#7B68EE"}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric not in df.columns:
            ax.set_axis_off()
            continue

        plot_df = df[["strategy_label", "mode", metric]].copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric])

        sns.barplot(
            data=plot_df,
            x="strategy_label",
            y=metric,
            hue="mode",
            palette=palette,
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("Strategy")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        if i > 0:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _heatmap(df: pd.DataFrame, metric: str, title: str, out_path: Path, center=None) -> None:
    if metric not in df.columns:
        return
    p = df[["strategy_label", "mode", metric]].copy()
    p[metric] = pd.to_numeric(p[metric], errors="coerce")
    p = p.dropna(subset=[metric])
    if p.empty:
        return

    pivot = p.pivot(index="strategy_label", columns="mode", values=metric)
    pivot = pivot.reindex(index=[METRIC_LABELS.get(s, s) for s in STRATEGY_ORDER if METRIC_LABELS.get(s, s) in pivot.index])
    pivot = pivot.reindex(columns=[m for m in MODE_ORDER if m in pivot.columns])

    plt.figure(figsize=(5.8, 3.6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu_r", center=center, cbar_kws={"shrink": 0.85})
    plt.title(title)
    plt.xlabel("Mode")
    plt.ylabel("Strategy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot soft-prompt experiment results")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("/scratch/craj/diy/results/8_soft_prompt/sp_embed_20260228_1"),
    )
    parser.add_argument(
        "--figures_root",
        type=Path,
        default=Path("/scratch/craj/diy/figures/8_soft_prompt/sp_embed_20260228_1"),
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")

    crows = _prep(load_crowspairs(args.results_root))
    stereo = _prep(load_stereoset(args.results_root))
    bbq = _prep(load_bbq(args.results_root))

    args.figures_root.mkdir(parents=True, exist_ok=True)

    if not crows.empty:
        crows.to_csv(args.figures_root / "summary_crowspairs.csv", index=False)
        _grouped_bars(
            crows,
            ["stereotype_preference_pct", "mean_stereo_prob_norm", "mean_anti_prob_norm"],
            "Soft Prompt Results: CrowS-Pairs",
            args.figures_root / "crowspairs_grouped_bars.png",
        )
        _heatmap(
            crows,
            "stereotype_preference_pct",
            "CrowS-Pairs stereotype_preference_pct",
            args.figures_root / "crowspairs_heatmap_stereotype_preference_pct.png",
            center=50.0,
        )

    if not stereo.empty:
        stereo.to_csv(args.figures_root / "summary_stereoset.csv", index=False)
        _grouped_bars(
            stereo,
            ["LM Score", "SS Score", "ICAT Score"],
            "Soft Prompt Results: StereoSet (overall)",
            args.figures_root / "stereoset_grouped_bars.png",
        )
        _heatmap(
            stereo,
            "ICAT Score",
            "StereoSet ICAT (overall)",
            args.figures_root / "stereoset_heatmap_icat.png",
            center=None,
        )

    if not bbq.empty:
        bbq.to_csv(args.figures_root / "summary_bbq.csv", index=False)
        _grouped_bars(
            bbq,
            ["Accuracy", "Bias_score_disambig", "Bias_score_ambig", "Abs_Bias_score_disambig", "Abs_Bias_score_ambig"],
            "Soft Prompt Results: BBQ (overall)",
            args.figures_root / "bbq_grouped_bars.png",
        )
        _heatmap(
            bbq,
            "Abs_Bias_score_disambig",
            "BBQ |Bias_score_disambig| (overall)",
            args.figures_root / "bbq_heatmap_abs_bias_disambig.png",
            center=None,
        )

    print(f"Saved soft-prompt plots to: {args.figures_root}")


if __name__ == "__main__":
    main()
