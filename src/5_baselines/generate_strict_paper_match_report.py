#!/usr/bin/env python3
"""
Generate a strict paper-vs-local baseline comparison report.

Strict means:
- same baseline method
- same dataset
- same model family (Llama-3.1-8B in this case)
- same metric definition

Rows that do not meet strict criteria are excluded from the numeric comparison table.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Dict, List


ROOT = Path("/scratch/craj/diy")
RESULTS_DIR = ROOT / "results" / "new_results"

OUR_BBA_STEREOSET = ROOT / "results" / "3_baselines" / "bba" / "stereoset_eval_llama_8b_bba.csv"
OUR_BBA_BBQ = ROOT / "results" / "3_baselines" / "bba" / "bbq_eval_llama_8b_bba.csv"
OUR_DECAP_BBQ_GLOB = "bbq_eval_llama_8b_decap_*.csv"

OUT_CSV = RESULTS_DIR / "paper_match_strict_comparison.csv"
OUT_HTML = RESULTS_DIR / "paper_match_strict_comparison.html"


def load_bba_stereoset() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with OUR_BBA_STEREOSET.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "intrasentence":
                continue
            domain = row["domain"]
            out[domain] = {
                "LM Score": float(row["LM Score"]),
                "SS Score": float(row["SS Score"]),
                "ICAT Score": float(row["ICAT Score"]),
            }
    return out


def load_bba_bbq_overall_percent() -> Dict[str, float]:
    with OUR_BBA_BBQ.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Category"] == "overall":
                return {
                    "Accuracy_ambig": float(row["Accuracy_ambig"]) * 100.0,
                    "Accuracy_disambig": float(row["Accuracy_disambig"]) * 100.0,
                    "Accuracy_avg": float(row["Accuracy"]) * 100.0,
                }
    raise RuntimeError("Could not find overall BBQ row in BBA results.")


def status_from_diff(diff: float) -> str:
    if diff <= 1.0:
        return "match"
    if diff <= 5.0:
        return "close"
    return "mismatch"


def build_rows() -> List[Dict[str, str]]:
    stereoset = load_bba_stereoset()
    bbq = load_bba_bbq_overall_percent()
    decap = load_decap_bbq_overall_percent()

    # Paper values extracted from BBA paper tables:
    # - StereoSet: Table 5, Llama-3.1, BBA row
    # - BBQ: Table 6, Llama-3.1, BBA row
    specs = [
        ("stereoset", "intrasentence", "gender", "SS Score", 69.84, "Table 5"),
        ("stereoset", "intrasentence", "gender", "LM Score", 98.44, "Table 5"),
        ("stereoset", "intrasentence", "gender", "ICAT Score", 59.38, "Table 5"),
        ("stereoset", "intrasentence", "profession", "SS Score", 71.58, "Table 5"),
        ("stereoset", "intrasentence", "profession", "LM Score", 95.56, "Table 5"),
        ("stereoset", "intrasentence", "profession", "ICAT Score", 54.32, "Table 5"),
        ("stereoset", "intrasentence", "religion", "SS Score", 49.31, "Table 5"),
        ("stereoset", "intrasentence", "religion", "LM Score", 92.41, "Table 5"),
        ("stereoset", "intrasentence", "religion", "ICAT Score", 91.14, "Table 5"),
        ("bbq", "overall", "overall", "Accuracy_ambig", 74.46, "Table 6"),
        ("bbq", "overall", "overall", "Accuracy_disambig", 88.95, "Table 6"),
        ("bbq", "overall", "overall", "Accuracy_avg", 81.70, "Table 6"),
        # DeCAP paper values from Table 3(a), Llama3-instruct(8B), DeCAP (ours):
        # Acc = 83.51, BS = 3.58
        ("decap_bbq", "overall", "overall", "Accuracy_avg", 83.51, "Table 3(a)"),
        ("decap_bbq", "overall", "overall", "BS_bbq", 3.58, "Table 3(a)"),
    ]

    rows: List[Dict[str, str]] = []
    for dataset, split, domain, metric, paper_val, paper_table in specs:
        if dataset == "stereoset":
            our_val = stereoset[domain][metric]
            baseline = "bba"
            paper_source = "BBA (arXiv:2602.04398)"
        elif dataset == "bbq":
            our_val = bbq[metric]
            baseline = "bba"
            paper_source = "BBA (arXiv:2602.04398)"
        elif dataset == "decap_bbq":
            our_val = decap[metric]
            baseline = "decap"
            paper_source = "DeCAP (arXiv:2503.19426)"
        else:
            raise RuntimeError(f"Unknown dataset tag: {dataset}")
        diff = abs(our_val - paper_val)
        status = status_from_diff(diff)
        rows.append(
            {
                "baseline": baseline,
                "dataset": "bbq" if dataset == "decap_bbq" else dataset,
                "split": split,
                "domain": domain,
                "metric": metric,
                "our_score": f"{our_val:.4f}",
                "paper_score": f"{paper_val:.4f}",
                "abs_diff": f"{diff:.4f}",
                "status": status,
                "paper_source": paper_source,
                "paper_table": paper_table,
            }
        )
    return rows


def load_decap_bbq_overall_percent() -> Dict[str, float]:
    """
    Aggregate per-category DeCAP BBQ files into an overall paper-style summary.
    Returns:
      - Accuracy_avg: weighted overall accuracy in %
      - BS_bbq: (|BS_ambig| + |BS_disambig|) / 2, where each BS is weighted by sample count
    """
    files = sorted((ROOT / "results" / "3_baselines").glob(OUR_DECAP_BBQ_GLOB))
    if not files:
        raise RuntimeError("No DeCAP BBQ files found for aggregation.")

    rows = []
    for path in files:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            rows.append(row)

    def wavg(metric_key: str, count_key: str) -> float:
        num = sum(float(r[metric_key]) * int(r[count_key]) for r in rows)
        den = sum(int(r[count_key]) for r in rows)
        return num / den

    acc = wavg("Accuracy", "N_total") * 100.0
    bs_amb = wavg("Bias_score_ambig", "N_ambig")
    bs_dis = wavg("Bias_score_disambig", "N_disambig")
    bs_bbq = (abs(bs_amb) + abs(bs_dis)) / 2.0
    return {
        "Accuracy_avg": acc,
        "BS_bbq": bs_bbq,
    }


def write_csv(rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "baseline",
        "dataset",
        "split",
        "domain",
        "metric",
        "our_score",
        "paper_score",
        "abs_diff",
        "status",
        "paper_source",
        "paper_table",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_html(rows: List[Dict[str, str]]) -> str:
    match_n = sum(1 for r in rows if r["status"] == "match")
    close_n = sum(1 for r in rows if r["status"] == "close")
    mismatch_n = sum(1 for r in rows if r["status"] == "mismatch")
    total_n = len(rows)

    excluded = [
        (
            "biasedit",
            "No directly comparable Llama-3.1-8B aggregate row in our current output format.",
            "Re-run BIASEDIT with paper-matching model + export per-domain SS/LM/ICAT and CrowS SS in one summary file.",
            "needs rerun",
        ),
        (
            "biasfreebench",
            "Paper uses Bias-Free Score framing, not our consolidated columns.",
            "Add official BFS metric computation and store BFS alongside current benchmark metrics.",
            "code change only",
        ),
        (
            "cal",
            "CAL paper reports different benchmark framing than our current summary columns.",
            "Create CAL-paper metric adapter (BBQ/UNQOVER prompt setup + metric parser), then compare on those fields.",
            "code change + rerun",
        ),
        (
            "debias_llms",
            "Paper reports GAS/GLD/ADD metrics, not CrowS/StereoSet/BBQ columns.",
            "Implement GAS/GLD/ADD eval for our checkpoints and compare directly on those metrics.",
            "code change only",
        ),
        (
            "debias_nlg",
            "Paper benchmark table is GPT-2 based; our run is Llama-3.1-8B.",
            "Either run paper GPT-2 setup for strict replication or mark comparison as cross-model only.",
            "needs rerun",
        ),
        (
            "dpo",
            "Citation in script points to an unrelated arXiv entry, so strict paper extraction is blocked.",
            "Fix citation to the exact SBM paper version and extract benchmark table values.",
            "metadata fix",
        ),
        (
            "fairsteer",
            "Paper and local run differ in setup details despite overlapping datasets.",
            "Add a relaxed-comparison table with explicit caveats (model/prompt/eval differences).",
            "code change only",
        ),
        (
            "lftf",
            "Paper benchmark family differs from current consolidated metrics.",
            "Run LFTF on paper benchmark suite and add parallel reporting columns.",
            "needs rerun",
        ),
        (
            "mbias",
            "Paper focuses safety/context-retention metrics absent from our table.",
            "Implement paper metrics (bias/toxicity/KR/faithfulness/relevancy) for our checkpoints.",
            "code change only",
        ),
        (
            "reduce_social_bias",
            "Paper reports stereotype engagement rates rather than our aggregate metric layout.",
            "Add paper-style stereotype-rate computation and compare that field directly.",
            "code change only",
        ),
        (
            "self_debiasing",
            "Paper uses GPT-3.5 ambiguous-BBQ setting; local run uses different model/setup.",
            "Run ambiguous-only protocol and keep model-specific caveat in comparison.",
            "needs rerun",
        ),
    ]

    row_html = []
    for r in rows:
        status_cls = f"status-{r['status']}"
        row_html.append(
            "<tr>"
            f"<td>{html.escape(r['baseline'])}</td>"
            f"<td>{html.escape(r['dataset'])}</td>"
            f"<td>{html.escape(r['split'])}</td>"
            f"<td>{html.escape(r['domain'])}</td>"
            f"<td>{html.escape(r['metric'])}</td>"
            f"<td class='num'>{html.escape(r['our_score'])}</td>"
            f"<td class='num'>{html.escape(r['paper_score'])}</td>"
            f"<td class='num'>{html.escape(r['abs_diff'])}</td>"
            f"<td><span class='pill {status_cls}'>{html.escape(r['status'])}</span></td>"
            f"<td>{html.escape(r['paper_table'])}</td>"
            "</tr>"
        )

    excluded_html = []
    for name, reason, fix, effort in excluded:
        excluded_html.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{html.escape(fix)}</td>"
            f"<td>{html.escape(effort)}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strict Paper Comparison</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --ink: #0f172a;
      --muted: #475569;
      --line: #d8dee8;
      --card: #ffffff;
      --match: #1e8e5a;
      --close: #c67700;
      --mismatch: #c62828;
      --pill-bg: #eef2f7;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 16px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 1.25rem;
    }}
    p {{
      margin: 0 0 10px 0;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 8px;
      margin: 10px 0 14px 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
    }}
    .card .k {{
      font-size: 0.76rem;
      color: var(--muted);
    }}
    .card .v {{
      font-size: 1.05rem;
      font-weight: 700;
      margin-top: 2px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      font-size: 0.86rem;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f1f5f9;
      font-weight: 700;
    }}
    td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      background: var(--pill-bg);
    }}
    .status-match {{ color: var(--match); }}
    .status-close {{ color: var(--close); }}
    .status-mismatch {{ color: var(--mismatch); }}
    .section {{
      margin-top: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Strict Paper Comparison (Comparable Rows Only)</h1>
    <p>Compared only where baseline, dataset, metric definition, and model family are directly aligned.</p>
    <p>Primary source extracted: BBA paper (arXiv:2602.04398), Table 5 and Table 6.</p>
    <div class="cards">
      <div class="card"><div class="k">Total rows</div><div class="v">{total_n}</div></div>
      <div class="card"><div class="k">Match (|diff| ≤ 1.0)</div><div class="v" style="color:var(--match)">{match_n}</div></div>
      <div class="card"><div class="k">Close (≤ 5.0)</div><div class="v" style="color:var(--close)">{close_n}</div></div>
      <div class="card"><div class="k">Mismatch (> 5.0)</div><div class="v" style="color:var(--mismatch)">{mismatch_n}</div></div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Baseline</th>
          <th>Dataset</th>
          <th>Split</th>
          <th>Domain</th>
          <th>Metric</th>
          <th>Our score</th>
          <th>Paper score</th>
          <th>|Diff|</th>
          <th>Status</th>
          <th>Paper table</th>
        </tr>
      </thead>
      <tbody>
        {"".join(row_html)}
      </tbody>
    </table>

    <div class="section">
      <h1>Excluded Baselines (Not Strictly Comparable)</h1>
      <table>
        <thead>
          <tr>
            <th>Baseline</th>
            <th>Reason excluded from strict numeric match</th>
            <th>How to make it comparable</th>
            <th>Effort</th>
          </tr>
        </thead>
        <tbody>
          {"".join(excluded_html)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    OUT_HTML.write_text(render_html(rows))
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_HTML}")


if __name__ == "__main__":
    main()
