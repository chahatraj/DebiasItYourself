#!/usr/bin/env python3
"""Combine baseline BBQ result CSVs into one wide CSV.

Output shape:
- Rows: one per input file
- Columns: two-level header
  - Level 1: bias dimension (from `Category` or `Model`)
  - Level 2: `Bias_score_disambig`, `Bias_score_ambig`
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import argparse
import csv


DEFAULT_INPUT_DIR = Path("/scratch/craj/diy/results/3_baselines")
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "combined_bias_scores.csv"
DEFAULT_TRANSPOSED_OUTPUT_PATH = DEFAULT_INPUT_DIR / "combined_bias_scores_transposed.csv"
DEFAULT_METRICS = ("Bias_score_disambig", "Bias_score_ambig")
DEFAULT_DIMENSION_COLUMNS = ("Category", "Model")


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
    raise ValueError(
        f"Unknown dimension strategy '{dimension_strategy}'"
    )


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
            reader.fieldnames, file_path, dimension_columns, dimension_strategy
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


def combine(
    input_dir: Path,
    output_path: Path,
    metrics: tuple[str, ...],
    dimension_columns: tuple[str, ...],
    match_patterns: tuple[str, ...],
    dimension_strategy: str,
) -> None:
    csv_files = sorted(input_dir.rglob("*.csv"))
    excluded_names = {output_path.name, DEFAULT_TRANSPOSED_OUTPUT_PATH.name}
    csv_files = [p for p in csv_files if p.name not in excluded_names]
    if match_patterns:
        csv_files = [
            p
            for p in csv_files
            if any(fnmatch(p.name, pat) for pat in match_patterns)
        ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_dir}")

    all_rows: dict[str, dict[tuple[str, str], str]] = {}
    all_dims: set[str] = set()

    for file_path in csv_files:
        label = f"{file_path.parent.name}/{file_path.name}"
        try:
            row_data = _read_one_file(
                file_path,
                metrics,
                dimension_columns,
                dimension_strategy,
            )
        except ValueError as e:
            print(f"Skipping incompatible file {file_path}: {e}")
            continue
        if not row_data:
            continue
        all_rows[label] = row_data
        all_dims.update(dim for dim, _metric in row_data.keys())

    if not all_rows:
        raise ValueError(
            "No compatible CSV files found with expected columns "
            f"({list(dimension_columns)} + {list(metrics)})."
        )

    sorted_dims = sorted(all_dims)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Two header rows for subcolumns.
        header_top = ["source_file"]
        header_sub = [""]
        for dim in sorted_dims:
            header_top.extend([dim] * len(metrics))
            header_sub.extend(list(metrics))

        writer.writerow(header_top)
        writer.writerow(header_sub)

        for source_file in sorted(all_rows.keys()):
            row = [source_file]
            values = all_rows[source_file]
            for dim in sorted_dims:
                for metric in metrics:
                    row.append(values.get((dim, metric), ""))
            writer.writerow(row)

    print(f"Wrote {len(all_rows)} rows to: {output_path}")


def transpose_combined(
    input_path: Path, transposed_output_path: Path, metrics: tuple[str, ...]
) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        raise ValueError(
            f"Expected at least 3 rows in {input_path} (2 headers + data rows)."
        )

    top = rows[0]  # source_file, Age, Age, Disability_status, ...
    sub = rows[1]  # , Bias_score_disambig, Bias_score_ambig, ...
    data = rows[2:]
    source_files = [r[0] for r in data]

    # Build ordered unique dimension list from header row.
    dimensions: list[str] = []
    seen: set[str] = set()
    for col_idx in range(1, len(top)):
        dim = top[col_idx].strip()
        if dim and dim not in seen:
            seen.add(dim)
            dimensions.append(dim)

    # Header: one column per (source_file, metric).
    header = ["dimension"]
    for source in source_files:
        for metric in metrics:
            header.append(f"{source}|{metric}")

    # Map (dimension, metric) -> column index in the combined CSV.
    col_lookup: dict[tuple[str, str], int] = {}
    for col_idx in range(1, len(top)):
        dim = top[col_idx].strip()
        metric = sub[col_idx].strip()
        if dim and metric:
            col_lookup[(dim, metric)] = col_idx

    transposed_rows: list[list[str]] = [header]
    for dim in dimensions:
        out_row = [dim]
        for src_row in data:
            for metric in metrics:
                col_idx = col_lookup.get((dim, metric))
                value = (
                    src_row[col_idx]
                    if col_idx is not None and col_idx < len(src_row)
                    else ""
                )
                out_row.append(value)
        transposed_rows.append(out_row)

    with transposed_output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(transposed_rows)

    print(f"Wrote transposed table to: {transposed_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing per-method CSVs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--transposed-output",
        type=Path,
        default=DEFAULT_TRANSPOSED_OUTPUT_PATH,
        help=(
            "Transposed CSV path "
            f"(default: {DEFAULT_TRANSPOSED_OUTPUT_PATH})"
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to combine.",
    )
    parser.add_argument(
        "--dimension-columns",
        nargs="+",
        default=list(DEFAULT_DIMENSION_COLUMNS),
        help="Dimension columns (used in order, joined by ' / ' if multiple).",
    )
    parser.add_argument(
        "--match-patterns",
        nargs="*",
        default=[],
        help="Optional filename patterns (fnmatch), e.g. '*bbq_eval*.csv'.",
    )
    parser.add_argument(
        "--dimension-strategy",
        choices=("first-match", "join-all"),
        default="first-match",
        help="How to use --dimension-columns.",
    )
    args = parser.parse_args()

    metrics = tuple(args.metrics)
    dimension_columns = tuple(args.dimension_columns)
    match_patterns = tuple(args.match_patterns)

    combine(
        args.input_dir,
        args.output,
        metrics=metrics,
        dimension_columns=dimension_columns,
        match_patterns=match_patterns,
        dimension_strategy=args.dimension_strategy,
    )
    transpose_combined(args.output, args.transposed_output, metrics=metrics)


if __name__ == "__main__":
    main()
