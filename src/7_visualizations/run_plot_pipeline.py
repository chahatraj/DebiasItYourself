#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _pd():
    import pandas as _pd_mod

    return _pd_mod


def _plot_utils():
    from _baseline_plot_utils import (
        load_transposed_long,
        plot_metric_bar,
        plot_metric_heatmap,
        rename_and_order_methods,
        set_plot_style,
    )

    return (
        load_transposed_long,
        plot_metric_bar,
        plot_metric_heatmap,
        rename_and_order_methods,
        set_plot_style,
    )


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    input_subdir: str
    output_name: str
    metrics: tuple[str, ...]
    dimension_columns: tuple[str, ...]
    dimension_strategy: str  # "first-match" or "join-all"
    match_patterns: tuple[str, ...]


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "crowspairs": DatasetConfig(
        name="crowspairs",
        input_subdir="crowspairs",
        output_name="combined_all_finetuning_crowspairs.csv",
        metrics=(
            "stereotype_preference_pct",
            "mean_stereo_prob_norm",
            "mean_anti_prob_norm",
        ),
        dimension_columns=("bias_type",),
        dimension_strategy="first-match",
        match_patterns=("crows_pairs_metrics_by_bias_*.csv",),
    ),
    "stereoset": DatasetConfig(
        name="stereoset",
        input_subdir="stereoset",
        output_name="combined_all_finetuning_stereoset.csv",
        metrics=("LM Score", "SS Score", "ICAT Score"),
        dimension_columns=("split", "domain"),
        dimension_strategy="join-all",
        match_patterns=("stereoset_metrics_*.csv",),
    ),
    "bbq": DatasetConfig(
        name="bbq",
        input_subdir="bbq",
        output_name="combined_all_finetuning_bbq.csv",
        metrics=("Bias_score_disambig", "Bias_score_ambig"),
        dimension_columns=("Category", "Model"),
        dimension_strategy="first-match",
        match_patterns=("bbq_eval_*.csv", "*bbq*eval*.csv"),
    ),
}


DATASET_LABEL = {
    "crowspairs": "CrowS-Pairs",
    "stereoset": "StereoSet",
    "bbq": "BBQ",
}

DATASET_INTERPRETATION = {
    "stereoset": {
        "LM Score": "Higher is better. This reflects language modeling quality (fluency/coherence likelihood).",
        "SS Score": (
            "Closer to 50 is better. Values far from 50 indicate stronger stereotypical or anti-stereotypical skew."
        ),
        "ICAT Score": "Higher is better. ICAT rewards strong language modeling quality with lower stereotype bias.",
    },
    "crowspairs": {
        "stereotype_preference_pct": (
            "Closer to 50 is better (less stereotype preference). "
            "Values above 50 indicate stronger stereotype preference; below 50 indicate anti-stereotype preference."
        ),
        "mean_stereo_prob_norm": (
            "Lower is better. Smaller normalized probability assigned to stereotypical continuation indicates less stereotype bias."
        ),
        "mean_anti_prob_norm": (
            "Higher is better. Larger normalized probability assigned to anti-stereotypical continuation indicates less stereotype bias."
        ),
    },
    "bbq": {
        "Bias_score_disambig": (
            "Lower absolute magnitude is better (values closer to 0 indicate less measured bias). "
            "Positive values indicate one bias direction; negative values indicate the opposite direction."
        ),
        "Bias_score_ambig": (
            "Lower absolute magnitude is better (values closer to 0 indicate less measured bias under ambiguous settings). "
            "Sign indicates direction of bias."
        ),
    },
}

DEFAULT_INTERPRETATION = {
    "stereoset": "Compare methods relative to the same dimension; prioritize lower bias and stronger language quality.",
    "crowspairs": (
        "Interpret this metric by comparing methods within the same dimension; "
        "lower stereotype signal is generally preferred."
    ),
    "bbq": "Lower absolute magnitude is generally preferred for bias scores (closer to 0 indicates less bias).",
}

BASELINE_ORDER = {
    "crowspairs": [
        "bba",
        "biasedit",
        "biasfreebench",
        "cal",
        "debias_llms",
        "debias_nlg",
        "dpo",
        "fairsteer",
        "lftf",
        "mbias",
        "peft",
        "reduce_social_bias",
    ],
    "stereoset": [
        "bba",
        "biasedit",
        "biasfreebench",
        "cal",
        "debias_llms",
        "debias_nlg",
        "dpo",
        "fairsteer",
        "lftf",
        "mbias",
        "peft",
        "reduce_social_bias",
    ],
    "bbq": [
        "3_baselines",
        "bba",
        "biasedit",
        "biasfreebench",
        "cal",
        "debias_llms",
        "debias_nlg",
        "dpo",
        "fairsteer",
        "lftf",
        "mbias",
        "peft",
        "reduce_social_bias",
        "self_debiasing",
    ],
}


def _find_dimension_columns(
    fieldnames: list[str],
    file_path: Path,
    preferred_columns: tuple[str, ...],
    dimension_strategy: str,
) -> tuple[str, ...]:
    cols = tuple(c for c in preferred_columns if c in fieldnames)
    if not cols:
        raise ValueError(
            f"No dimension columns found in {file_path}. "
            f"Tried: {list(preferred_columns)}"
        )
    if dimension_strategy == "first-match":
        return (cols[0],)
    if dimension_strategy == "join-all":
        missing = tuple(c for c in preferred_columns if c not in fieldnames)
        if missing:
            raise ValueError(
                f"Missing required dimension columns in {file_path}: {list(missing)}"
            )
        return tuple(preferred_columns)
    raise ValueError(f"Unknown dimension strategy '{dimension_strategy}'")


def _read_one_file(
    file_path: Path,
    metrics: tuple[str, ...],
    dimension_columns: tuple[str, ...],
    dimension_strategy: str,
) -> dict[tuple[str, str], str]:
    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {file_path}")

        dim_cols = _find_dimension_columns(
            reader.fieldnames,
            file_path,
            dimension_columns,
            dimension_strategy,
        )
        needed = {*dim_cols, *metrics}
        missing = needed - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {file_path}: {sorted(missing)}")

        row_data: dict[tuple[str, str], str] = {}
        for rec in reader:
            dim_parts = [(rec.get(c) or "").strip() for c in dim_cols]
            if not all(dim_parts):
                continue
            dim = " / ".join(dim_parts)
            for metric in metrics:
                row_data[(dim, metric)] = rec.get(metric, "")

        return row_data


def _list_matching_csvs(input_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not input_dir.exists():
        return []
    files = sorted(input_dir.rglob("*.csv"))
    if not patterns:
        return files
    return [p for p in files if any(fnmatch(p.name, pat) for pat in patterns)]


def combine_dataset(input_dir: Path, config: DatasetConfig, output_file: Path) -> bool:
    csv_files = _list_matching_csvs(input_dir, config.match_patterns)
    if not csv_files:
        print(f"[SKIP] {config.name}: no matching files in {input_dir}")
        return False

    all_rows: dict[str, dict[tuple[str, str], str]] = {}
    all_dims: set[str] = set()

    for file_path in csv_files:
        label = file_path.relative_to(input_dir).as_posix()
        try:
            row_data = _read_one_file(
                file_path=file_path,
                metrics=config.metrics,
                dimension_columns=config.dimension_columns,
                dimension_strategy=config.dimension_strategy,
            )
        except ValueError as exc:
            print(f"[WARN] {config.name}: skipping incompatible file {file_path}: {exc}")
            continue

        if not row_data:
            continue
        all_rows[label] = row_data
        all_dims.update(dim for dim, _metric in row_data.keys())

    if not all_rows:
        print(f"[SKIP] {config.name}: no compatible metric files in {input_dir}")
        return False

    sorted_dims = sorted(all_dims)
    source_files = sorted(all_rows.keys())
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["dimension"]
        for source_file in source_files:
            for metric in config.metrics:
                header.append(f"{source_file}|{metric}")
        writer.writerow(header)

        for dim in sorted_dims:
            row = [dim]
            for source_file in source_files:
                values = all_rows[source_file]
                for metric in config.metrics:
                    row.append(values.get((dim, metric), ""))
            writer.writerow(row)

    print(f"[OK] {config.name}: wrote transposed -> {output_file}")
    return True


def _resolve_dataset_input_dir(run_root: Path, input_subdir: str) -> Path:
    nested = run_root / "results" / input_subdir
    flat = run_root / input_subdir
    if nested.exists():
        return nested
    if flat.exists():
        return flat
    return nested


def prepare_finetuning_results(run_root: Path, output_root: Path, datasets: list[str]) -> None:
    for dataset_name in datasets:
        cfg = DATASET_CONFIGS[dataset_name]
        input_dir = _resolve_dataset_input_dir(run_root=run_root, input_subdir=cfg.input_subdir)
        output_path = output_root / cfg.output_name

        combine_dataset(input_dir=input_dir, config=cfg, output_file=output_path)


def _empty_df() -> pd.DataFrame:
    pd = _pd()
    return pd.DataFrame(columns=["dimension", "source", "metric", "method", "value"])


def _default_baseline_csv(dataset: str) -> Path:
    return Path(f"/scratch/craj/diy/results/3_baselines/combined_all_baselines_{dataset}_transposed.csv")


def _default_finetuning_csv(dataset: str) -> Path:
    root = Path("/scratch/craj/diy/results/6_instructiontuning")
    return root / f"combined_all_finetuning_{dataset}.csv"


def _default_output_dir(dataset: str, mode: str) -> Path:
    if mode == "baselines":
        return Path(f"/scratch/craj/diy/figures/{dataset}")
    root = Path("/scratch/craj/diy/figures/6_instructiontuning")
    if mode == "finetuning":
        return root / dataset / "finetuning"
    return root / dataset / "combined"


def _metric_center(dataset: str, safe_metric: str) -> float | None:
    if dataset == "stereoset":
        return 50.0 if "score" in safe_metric and "icat" not in safe_metric else None
    if dataset == "crowspairs":
        return 50.0 if "stereotype_preference_pct" in safe_metric else 0.0
    if dataset == "bbq":
        return 0.0 if "bias_score" in safe_metric else None
    return None


def _metric_label(dataset: str, metric: str) -> str:
    if dataset == "bbq":
        return metric.replace("_", " ")
    return metric


def _interpretation(dataset: str, metric: str) -> str:
    return DATASET_INTERPRETATION.get(dataset, {}).get(metric, DEFAULT_INTERPRETATION[dataset])


def _load_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_df()
    load_transposed_long, *_ = _plot_utils()
    return load_transposed_long(path)


def _render(
    dataset: str,
    mode: str,
    df: pd.DataFrame,
    output_dir: Path,
    method_order: list[str] | None,
    prettify_method_labels: bool,
) -> None:
    _, plot_metric_bar, plot_metric_heatmap, _, _ = _plot_utils()
    dataset_name = DATASET_LABEL[dataset]
    metrics = list(df["metric"].drop_duplicates())

    for metric in metrics:
        safe_metric = metric.lower().replace(" ", "_")
        metric_label = _metric_label(dataset, metric)
        interpretation = _interpretation(dataset, metric)
        center = _metric_center(dataset, safe_metric)

        if mode == "baselines":
            heatmap_title = f"{dataset_name} Baselines by Dimension ({metric_label})"
            bar_title = f"{dataset_name} Baseline Mean by Method ({metric_label})"
        elif mode == "finetuning":
            heatmap_title = f"{dataset_name} Finetuning by Dimension ({metric_label})"
            bar_title = f"{dataset_name} Finetuning Mean by Model ({metric_label})"
        else:
            heatmap_title = f"{dataset_name} Baselines + Finetuning by Dimension ({metric_label})"
            bar_title = f"{dataset_name} Baselines + Finetuning Mean by Method ({metric_label})"

        plot_metric_heatmap(
            df,
            metric=metric,
            out_path=output_dir / f"{dataset}_heatmap_{safe_metric}.png",
            title=heatmap_title,
            interpretation=interpretation,
            center=center,
            method_order=method_order,
            prettify_method_labels=prettify_method_labels,
        )
        plot_metric_bar(
            df,
            metric=metric,
            out_path=output_dir / f"{dataset}_mean_{safe_metric}.png",
            title=bar_title,
            interpretation=interpretation,
            method_order=method_order,
            prettify_method_labels=prettify_method_labels,
        )

    print(f"[DONE] {dataset_name} {mode} plots written to: {output_dir}")


def _plot_baselines(dataset: str, input_csv: Path, output_dir: Path) -> None:
    if not input_csv.exists():
        print(f"[SKIP] no input CSV found: {input_csv}")
        return
    load_transposed_long, _plot_bar, _plot_heatmap, _rename, set_plot_style = _plot_utils()
    set_plot_style()
    df = load_transposed_long(input_csv)
    _render(
        dataset=dataset,
        mode="baselines",
        df=df,
        output_dir=output_dir,
        method_order=None,
        prettify_method_labels=True,
    )


def _plot_finetuning(dataset: str, input_paths: list[Path], output_dir: Path) -> None:
    existing_paths = [p for p in input_paths if p.exists()]
    if not existing_paths:
        print(f"[SKIP] no input CSV found. Checked: {[str(p) for p in input_paths]}")
        return

    pd = _pd()
    load_transposed_long, _plot_bar, _plot_heatmap, rename_and_order_methods, set_plot_style = _plot_utils()
    set_plot_style()
    df = pd.concat([load_transposed_long(p) for p in existing_paths], ignore_index=True)
    df, method_order = rename_and_order_methods(df)
    _render(
        dataset=dataset,
        mode="finetuning",
        df=df,
        output_dir=output_dir,
        method_order=method_order,
        prettify_method_labels=False,
    )


def _plot_baselines_plus_finetuning(
    dataset: str,
    baseline_csv: Path,
    finetuning_paths: list[Path],
    output_dir: Path,
) -> None:
    pd = _pd()
    _, _, _, rename_and_order_methods, set_plot_style = _plot_utils()
    base_df = _load_or_empty(baseline_csv)
    ft_parts = [_load_or_empty(p) for p in finetuning_paths]
    ft_parts = [d for d in ft_parts if not d.empty]
    ft_df = pd.concat(ft_parts, ignore_index=True) if ft_parts else _empty_df()
    ft_df, ft_order = rename_and_order_methods(ft_df)

    if base_df.empty and ft_df.empty:
        print("[SKIP] no baseline or finetuning input found")
        return

    df = pd.concat([d for d in (base_df, ft_df) if not d.empty], ignore_index=True)
    base_methods = set(base_df["method"].unique())
    preferred = BASELINE_ORDER[dataset]
    baseline_present = [m for m in preferred if m in base_methods]
    baseline_extras = sorted([m for m in base_methods if m not in set(preferred)])
    method_order = baseline_present + baseline_extras + ft_order

    set_plot_style()
    _render(
        dataset=dataset,
        mode="baselines_plus_finetuning",
        df=df,
        output_dir=output_dir,
        method_order=method_order,
        prettify_method_labels=False,
    )


def _finetuning_csv(results_root: Path, dataset: str) -> Path:
    return results_root / f"combined_all_finetuning_{dataset}.csv"


def _mode_enabled(mode: str, target: str) -> bool:
    if mode == "all":
        return True
    if target == "baselines_plus_finetuning":
        return mode in {"baselines_plus_finetuning", "combined"}
    return mode == target


def _run_single_dataset_plot(args: argparse.Namespace) -> None:
    if args.mode in {"combined", "all"}:
        raise ValueError("When using --dataset, --mode must be one of: baselines, finetuning, baselines_plus_finetuning")

    dataset = args.dataset
    output_dir = args.output_dir or _default_output_dir(dataset, args.mode)

    if args.mode == "baselines":
        input_csv = args.input_csv or _default_baseline_csv(dataset)
        _plot_baselines(dataset=dataset, input_csv=input_csv, output_dir=output_dir)
        return

    if args.mode == "finetuning":
        input_paths = args.input_csvs if args.input_csvs else [args.input_csv or _default_finetuning_csv(dataset)]
        _plot_finetuning(dataset=dataset, input_paths=input_paths, output_dir=output_dir)
        return

    baseline_csv = args.baseline_csv or _default_baseline_csv(dataset)
    finetuning_paths = (
        args.finetuning_csvs if args.finetuning_csvs else [args.finetuning_csv or _default_finetuning_csv(dataset)]
    )
    _plot_baselines_plus_finetuning(
        dataset=dataset,
        baseline_csv=baseline_csv,
        finetuning_paths=finetuning_paths,
        output_dir=output_dir,
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    if args.prepare_only:
        prepare_finetuning_results(run_root=args.run_root, output_root=args.results_root, datasets=args.datasets)
        print("[DONE] Finetuning result preparation complete.")
        return

    if _mode_enabled(args.mode, "baselines"):
        for dataset in args.datasets:
            _plot_baselines(
                dataset=dataset,
                input_csv=_default_baseline_csv(dataset),
                output_dir=_default_output_dir(dataset, "baselines"),
            )

    if _mode_enabled(args.mode, "finetuning") or _mode_enabled(args.mode, "baselines_plus_finetuning"):
        prepare_finetuning_results(run_root=args.run_root, output_root=args.results_root, datasets=args.datasets)

    if _mode_enabled(args.mode, "finetuning"):
        for dataset in args.datasets:
            _plot_finetuning(
                dataset=dataset,
                input_paths=[_finetuning_csv(args.results_root, dataset)],
                output_dir=args.figures_root / dataset / "finetuning",
            )

    if _mode_enabled(args.mode, "baselines_plus_finetuning"):
        for dataset in args.datasets:
            _plot_baselines_plus_finetuning(
                dataset=dataset,
                baseline_csv=_default_baseline_csv(dataset),
                finetuning_paths=[_finetuning_csv(args.results_root, dataset)],
                output_dir=args.figures_root / dataset / "combined",
            )

    print(f"[DONE] Visualization pipeline complete (mode={args.mode}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified visualization CLI for preparing finetuning tables and generating "
            "baseline/finetuning/combined plots."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["baselines", "finetuning", "baselines_plus_finetuning", "combined", "all"],
        default="all",
    )
    parser.add_argument(
        "--dataset",
        choices=["crowspairs", "stereoset", "bbq"],
        default=None,
        help="Single dataset mode (replaces old plot_results.py usage).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["crowspairs", "stereoset", "bbq"],
        default=["crowspairs", "stereoset", "bbq"],
        help="Pipeline mode datasets (ignored when --dataset is provided).",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/scratch/craj/diy/outputs/6_instructiontuning"),
        help="Run directory containing either <run_root>/results/<dataset>/... or <run_root>/<dataset>/...",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/scratch/craj/diy/results/6_instructiontuning"),
        help="Directory where finetuning CSVs are written/read.",
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=Path("/scratch/craj/diy/figures/6_instructiontuning"),
        help="Directory where generated finetuning and combined figures are written.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build finetuning CSVs (pipeline mode).",
    )

    # Single-dataset plotting options (replaces old plot_results.py args).
    parser.add_argument("--input_csv", type=Path, default=None)
    parser.add_argument(
        "--input_csvs",
        type=Path,
        nargs="+",
        default=None,
        help="Optional list of finetuning CSVs to combine.",
    )
    parser.add_argument("--baseline_csv", type=Path, default=None)
    parser.add_argument("--finetuning_csv", type=Path, default=None)
    parser.add_argument(
        "--finetuning_csvs",
        type=Path,
        nargs="+",
        default=None,
        help="Optional list of finetuning CSVs to combine.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset:
        _run_single_dataset_plot(args)
        return

    _run_pipeline(args)


if __name__ == "__main__":
    main()
