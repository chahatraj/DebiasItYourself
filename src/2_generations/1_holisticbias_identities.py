import json
import csv
from collections.abc import Mapping

# Load the two JSON files
with open("/scratch/craj/diy/data/holisticbias/descriptors_1.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

with open("/scratch/craj/diy/data/holisticbias/descriptors.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# Recursive union merge
def deep_union(d1, d2):
    if isinstance(d1, Mapping) and isinstance(d2, Mapping):
        merged = dict(d1)
        for k, v in d2.items():
            if k in merged:
                merged[k] = deep_union(merged[k], v)
            else:
                merged[k] = v
        return merged
    elif isinstance(d1, list) and isinstance(d2, list):
        combined = d1 + d2
        seen = []
        for item in combined:
            if item not in seen:
                seen.append(item)
        return seen
    else:
        if d1 == d2:
            return d1
        return [d1, d2]

# Perform the union
merged_data = deep_union(data1, data2)

# Save merged JSON
json_out = "/scratch/craj/diy/data/holisticbias/merged_descriptors.json"
with open(json_out, "w", encoding="utf-8") as out:
    json.dump(merged_data, out, indent=4, ensure_ascii=False)
print(f"✅ Merged JSON saved at {json_out}")

# --- Save as CSV with hierarchy split into columns ---
csv_out = "/scratch/craj/diy/data/holisticbias/merged_descriptors.csv"

def flatten_json(obj, parent_keys=None):
    rows = []
    if parent_keys is None:
        parent_keys = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            rows.extend(flatten_json(v, parent_keys + [k]))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            rows.extend(flatten_json(item, parent_keys + [str(i)]))
    else:
        rows.append(parent_keys + [obj])
    return rows

flat_rows = flatten_json(merged_data)

# --- Ensure consistent 5 columns: bias_dimension, level_2, level_3, level_4, identity ---
normalized_rows = []
for r in flat_rows:
    # last element is always identity
    identity = r[-1]
    levels = r[:-1]

    # pad levels up to 4 slots
    while len(levels) < 4:
        levels.append("")

    # keep only 4 levels (in case deeper nesting exists)
    levels = levels[:4]

    # construct row: bias_dimension, level_2, level_3, level_4, identity
    normalized_rows.append(levels + [identity])

# Remove rows where level_4 == "preference"
filtered_rows = [r for r in normalized_rows if r[3] != "preference"]

# Headers
headers = ["bias_dimension", "level_2", "level_3", "level_4", "identity"]

# Write CSV
with open(csv_out, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(filtered_rows)

print(f"✅ Merged CSV saved at {csv_out} (rows with 'preference' in level_4 removed, headers fixed)")
