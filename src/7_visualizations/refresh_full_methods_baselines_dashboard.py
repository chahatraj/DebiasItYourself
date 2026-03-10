#!/usr/bin/env python3
"""Backfill dashboard table cells from completed result files.

Default behavior only fills missing cells ("", "--") in:
  - results/new_results/full_methods_baselines_dataset_table.csv
Then rewrites the embedded CSV block in:
  - results/new_results/full_methods_baselines_dashboard.html
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path("/scratch/craj/diy")
TABLE_CSV = ROOT / "results/new_results/full_methods_baselines_dataset_table.csv"
DASHBOARD_HTML = ROOT / "results/new_results/full_methods_baselines_dashboard.html"

METHOD_RESULTS_ROOT = ROOT / "results/new_results"
BASELINE_RESULTS_ROOT = ROOT / "results/3_baselines"
ADDL_BASELINES_ROOT = ROOT / "results/10_additional_benchmarks/baselines"
ADDL_BASELINE_METHODS_ROOT = ROOT / "results/10_additional_benchmarks/baseline_methods"

MISSING_TOKENS = {"", "--"}

METRIC_TO_DATASET = {
    "crowspairs_metric_score": "crowspairs",
    "stereoset_icat_overall": "stereoset",
    "bbq_mean_abs_bias_disambig": "bbq",
    "bold_toxicity_prop": "bold",
    "honest_score": "honest",
    "winobias_abs_pro_anti_gap": "winobias",
    "winogender_mean_abs_occupation_gender_bias_score": "winogender",
    "bias_in_bios_mean_abs_tpr_gap": "bias_in_bios",
    "unqover_gender_mu_bias_intensity": "unqover_gender",
    "unqover_race_mu_bias_intensity": "unqover_race",
    "unqover_religion_mu_bias_intensity": "unqover_religion",
    "unqover_nationality_mu_bias_intensity": "unqover_nationality",
}


def _latest_path(paths: Iterable[Path]) -> Optional[Path]:
    items = [p for p in paths if p.exists() and p.is_file()]
    if not items:
        return None
    return sorted(items, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _glob_latest(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    return _latest_path(matches)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_missing(value: Optional[str]) -> bool:
    return (value or "").strip() in MISSING_TOKENS


def _first_non_missing_value(rows: list[dict[str, str]], key: str) -> Optional[str]:
    for row in rows:
        val = (row.get(key) or "").strip()
        if not _is_missing(val):
            return val
    return None


def _extract_crowspairs(path: Path) -> Optional[str]:
    rows = _read_csv_rows(path)
    return _first_non_missing_value(rows, "metric_score")


def _extract_stereoset(path: Path) -> Optional[str]:
    rows = _read_csv_rows(path)
    for row in rows:
        split = (row.get("split") or "").strip().lower()
        domain = (row.get("domain") or "").strip().lower()
        if split == "overall" and domain == "overall":
            val = (row.get("ICAT Score") or "").strip()
            if not _is_missing(val):
                return val
    return _first_non_missing_value(rows, "ICAT Score")


def _extract_bbq(path: Path) -> Optional[str]:
    rows = _read_csv_rows(path)
    for row in rows:
        cat = (row.get("Category") or row.get("category") or "").strip().lower()
        model = (row.get("Model") or row.get("model") or "").strip().lower()
        input_file = (row.get("input_file") or "").strip().lower()
        if cat == "overall" or model == "overall" or input_file == "__overall__":
            raw = (row.get("Bias_score_disambig") or row.get("bias_score_disambig") or "").strip()
            if _is_missing(raw):
                continue
            try:
                return str(abs(float(raw)))
            except ValueError:
                return raw
    raw = _first_non_missing_value(rows, "Bias_score_disambig")
    if raw is None:
        raw = _first_non_missing_value(rows, "bias_score_disambig")
    if raw is None:
        return None
    try:
        return str(abs(float(raw)))
    except ValueError:
        return raw


def _extract_overall_metric(path: Path, key: str) -> Optional[str]:
    rows = _read_csv_rows(path)
    return _first_non_missing_value(rows, key)


def _latest_baseline_base_dir() -> Optional[Path]:
    if not ADDL_BASELINES_ROOT.exists():
        return None
    dirs = [d for d in ADDL_BASELINES_ROOT.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]


def _method_file(row_name: str, dataset_key: str) -> Optional[Path]:
    method_root = METHOD_RESULTS_ROOT / row_name
    if not method_root.exists():
        return None
    mapping = {
        "crowspairs": ["**/crows_pairs_metrics_overall*.csv", "**/crowspairs_metrics_overall*.csv"],
        "stereoset": ["**/stereoset_metrics*.csv"],
        "bbq": ["**/bbq_metrics*.csv"],
        "bold": ["**/bold_metrics_overall*.csv"],
        "honest": ["**/honest_metrics_overall*.csv"],
        "winobias": ["**/winobias_metrics_overall*.csv"],
        "winogender": ["**/winogender_metrics_overall*.csv"],
        "bias_in_bios": ["**/bias_in_bios_metrics_overall*.csv"],
        "unqover_gender": ["**/unqover_gender_metrics_overall*.csv"],
        "unqover_race": ["**/unqover_race_metrics_overall*.csv"],
        "unqover_religion": ["**/unqover_religion_metrics_overall*.csv"],
        "unqover_nationality": ["**/unqover_nationality_metrics_overall*.csv"],
    }
    return _glob_latest(method_root, mapping[dataset_key])


def _baseline_additional_root(row_name: str) -> Optional[Path]:
    if not ADDL_BASELINE_METHODS_ROOT.exists():
        return None
    candidates = [p for p in ADDL_BASELINE_METHODS_ROOT.glob(f"*/{row_name}") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda d: d.stat().st_mtime, reverse=True)[0]


def _baseline_file(row_name: str, dataset_key: str) -> Optional[Path]:
    if row_name == "llama_8b_base":
        base_dir = _latest_baseline_base_dir()
        if base_dir is None:
            return None
        mapping = {
            "crowspairs": ["**/crows_pairs_metrics_overall*.csv", "**/crowspairs_metrics_overall*.csv"],
            "stereoset": ["**/stereoset_metrics*.csv"],
            "bbq": ["**/bbq_metrics*.csv"],
            "bold": ["**/bold_metrics_overall*.csv"],
            "honest": ["**/honest_metrics_overall*.csv"],
            "winobias": ["**/winobias_metrics_overall*.csv"],
            "winogender": ["**/winogender_metrics_overall*.csv"],
            "bias_in_bios": ["**/bias_in_bios_metrics_overall*.csv"],
            "unqover_gender": ["**/unqover_gender_metrics_overall*.csv"],
            "unqover_race": ["**/unqover_race_metrics_overall*.csv"],
            "unqover_religion": ["**/unqover_religion_metrics_overall*.csv"],
            "unqover_nationality": ["**/unqover_nationality_metrics_overall*.csv"],
        }
        return _glob_latest(base_dir, mapping[dataset_key])

    if dataset_key in {"crowspairs", "stereoset", "bbq"}:
        base_dir = BASELINE_RESULTS_ROOT / row_name
        if not base_dir.exists():
            return None
        mapping = {
            "crowspairs": ["**/crowspairs_metrics_overall*.csv", "**/crows_pairs_metrics_overall*.csv"],
            "stereoset": ["**/stereoset*.csv"],
            "bbq": ["**/bbq*eval*.csv", "**/bbq_metrics*.csv"],
        }
        return _glob_latest(base_dir, mapping[dataset_key])

    addl_root = _baseline_additional_root(row_name)
    if addl_root is None:
        return None
    mapping = {
        "bold": ["**/bold_metrics_overall*.csv"],
        "honest": ["**/honest_metrics_overall*.csv"],
        "winobias": ["**/winobias_metrics_overall*.csv"],
        "winogender": ["**/winogender_metrics_overall*.csv"],
        "bias_in_bios": ["**/bias_in_bios_metrics_overall*.csv"],
        "unqover_gender": ["**/unqover_gender_metrics_overall*.csv"],
        "unqover_race": ["**/unqover_race_metrics_overall*.csv"],
        "unqover_religion": ["**/unqover_religion_metrics_overall*.csv"],
        "unqover_nationality": ["**/unqover_nationality_metrics_overall*.csv"],
    }
    return _glob_latest(addl_root, mapping[dataset_key])


def _extract_metric(metric_name: str, file_path: Path) -> Optional[str]:
    dataset_key = METRIC_TO_DATASET[metric_name]
    if dataset_key == "crowspairs":
        return _extract_crowspairs(file_path)
    if dataset_key == "stereoset":
        return _extract_stereoset(file_path)
    if dataset_key == "bbq":
        return _extract_bbq(file_path)
    if dataset_key == "bold":
        return _extract_overall_metric(file_path, "toxicity_prop")
    if dataset_key == "honest":
        return _extract_overall_metric(file_path, "honest_score")
    if dataset_key == "winobias":
        return _extract_overall_metric(file_path, "abs_pro_anti_gap")
    if dataset_key == "winogender":
        return _extract_overall_metric(file_path, "mean_abs_occupation_gender_bias_score")
    if dataset_key == "bias_in_bios":
        return _extract_overall_metric(file_path, "mean_abs_tpr_gap")
    if dataset_key.startswith("unqover_"):
        return _extract_overall_metric(file_path, "mu_bias_intensity")
    return None


def _embed_csv_in_dashboard(html_path: Path, csv_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8").strip("\n")
    pattern = re.compile(
        r'(<script id="csv-data" type="text/plain">\n)(.*?)(\n\s*</script>)',
        flags=re.DOTALL,
    )
    replacement = r"\1" + csv_text + r"\3"
    new_html, n = pattern.subn(replacement, html, count=1)
    if n != 1:
        raise RuntimeError("Could not locate unique <script id=\"csv-data\"> block in dashboard HTML.")
    html_path.write_text(new_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=TABLE_CSV)
    parser.add_argument("--html", type=Path, default=DASHBOARD_HTML)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace non-missing cells with latest available values.",
    )
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    updated_cells = 0
    for row in rows:
        row_type = (row.get("type") or "").strip()
        row_name = (row.get("name") or "").strip()
        for metric in METRIC_TO_DATASET:
            current = (row.get(metric) or "").strip()
            if not args.overwrite_existing and not _is_missing(current):
                continue
            if row_type == "method":
                fp = _method_file(row_name, METRIC_TO_DATASET[metric])
            else:
                fp = _baseline_file(row_name, METRIC_TO_DATASET[metric])
            if fp is None:
                continue
            new_val = _extract_metric(metric, fp)
            if _is_missing(new_val):
                continue
            if current != new_val:
                row[metric] = new_val
                updated_cells += 1

    if updated_cells > 0 or args.overwrite_existing:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        _embed_csv_in_dashboard(args.html, args.csv)

    filled = 0
    missing = 0
    for row in rows:
        for metric in METRIC_TO_DATASET:
            if _is_missing(row.get(metric)):
                missing += 1
            else:
                filled += 1

    print(f"updated_cells={updated_cells}")
    print(f"filled={filled}")
    print(f"missing={missing}")


if __name__ == "__main__":
    main()
