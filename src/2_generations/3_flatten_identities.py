#!/usr/bin/env python3
import json
import re
from pathlib import Path
import pandas as pd
from typing import List

# ========= Paths =========
INPUT_FILE = Path("/scratch/craj/diy/outputs/1_generations/identities/bias_identities_llama.jsonl")
OUTPUT_CSV  = INPUT_FILE.with_name("bias_identities_flat.csv")
OUTPUT_JSONL = INPUT_FILE.with_name("bias_identities_flat.jsonl")

# ========= Text cleanup helpers =========
def strip_code_fences(t: str) -> str:
    """Remove ```json / ``` code fences if present."""
    if not isinstance(t, str):
        return ""
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t

def normalize_quotes(t: str) -> str:
    """Collapse CSV-style doubled quotes into normal quotes."""
    return t.replace('""', '"')

def remove_json_comments(t: str) -> str:
    """Remove // line comments and /* */ block comments."""
    t = re.sub(r"//.*?$", "", t, flags=re.MULTILINE)
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
    return t

def remove_trailing_commas(t: str) -> str:
    """Remove trailing commas before ] or } which break json.loads."""
    t = re.sub(r",\s*(\])", r"\1", t)
    t = re.sub(r",\s*(\})", r"\1", t)
    return t

# ========= Core extraction =========
IDENTITY_PLACEHOLDERS = {"...", "…"}  # filter out placeholder entries

def extract_string_items_from_array_chunk(chunk: str) -> List[str]:
    """Given the inside of [...] return a list of decoded strings."""
    chunk = remove_json_comments(chunk)
    chunk = remove_trailing_commas(chunk)
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', chunk, flags=re.DOTALL)
    cleaned = []
    seen = set()
    for raw in items:
        try:
            val = json.loads(f'"{raw}"')
        except Exception:
            val = raw
        val = val.strip()
        if not val or val in IDENTITY_PLACEHOLDERS:
            continue
        if val not in seen:
            seen.add(val)
            cleaned.append(val)
    return cleaned

def find_best_identities_in_text(raw: str) -> List[str]:
    """Find all identities arrays in the text and return the largest one."""
    if not isinstance(raw, str) or not raw.strip():
        return []

    t = normalize_quotes(raw)
    t = strip_code_fences(t)
    t = remove_json_comments(t)

    best = []

    # 1) Regex match every "identities": [ ... ] block
    pattern = re.compile(r'"identities"\s*:\s*\[(.*?)\]', re.DOTALL | re.IGNORECASE)
    for m in pattern.finditer(t):
        chunk = m.group(1)
        items = extract_string_items_from_array_chunk(chunk)
        if len(items) > len(best):
            best = items

    if best:
        return best

    # 2) Fallback: try to json.loads { ... } blocks
    obj_pat = re.compile(r'\{[^{}]*"identities"\s*:\s*\[[\s\S]*?\][^{}]*\}', re.DOTALL | re.IGNORECASE)
    for m in obj_pat.finditer(t):
        block = remove_trailing_commas(m.group(0))
        try:
            obj = json.loads(block)
            ids = obj.get("identities", [])
            if isinstance(ids, list):
                seen = set()
                items = []
                for it in ids:
                    if isinstance(it, str):
                        v = it.strip()
                        if v and v not in IDENTITY_PLACEHOLDERS and v not in seen:
                            seen.add(v)
                            items.append(v)
                if len(items) > len(best):
                    best = items
        except Exception:
            continue

    return best

# ========= Read JSONL and flatten =========
flat_rows = []
no_ids_dims = []

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        dim = str(obj.get("dimension", "")).strip()
        raw = obj.get("raw_output", "")
        ids = find_best_identities_in_text(raw)

        if not ids:
            no_ids_dims.append(dim)
            continue

        for ident in ids:
            flat_rows.append({"bias_dimension": dim, "identity": ident})

# ========= Save outputs =========
df = pd.DataFrame(flat_rows)
df.to_csv(OUTPUT_CSV, index=False)
with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
    for row in flat_rows:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

dims = df["bias_dimension"].nunique() if not df.empty else 0
print(f"✅ Flattened {len(df)} rows across {dims} dimensions.")
if no_ids_dims:
    print(f"⚠️ No identities parsed for: {no_ids_dims}")
print(f"- CSV:  {OUTPUT_CSV}")
print(f"- JSONL:{OUTPUT_JSONL}")