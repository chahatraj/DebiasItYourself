import pandas as pd
import ast
import os

# === Paths ===
INPUT_FILE = "/scratch/craj/diy/outputs/1_generations/bias_concepts_llama.csv"
OUTPUT_FILE = "/scratch/craj/diy/outputs/1_generations/bias_concepts_llama_clean.csv"

# === Load CSV ===
df = pd.read_csv(INPUT_FILE)

# === Safely parse the list-like string in 'concept_templates' ===
def parse_templates(x):
    if pd.isna(x):
        return []
    try:
        # Sometimes it's a string representation of a list or JSON
        return ast.literal_eval(x)
    except Exception:
        return []

df["concept_templates_list"] = df["concept_templates"].apply(parse_templates)

# === Expand templates into separate columns ===
max_len = max(df["concept_templates_list"].apply(len))
template_cols = [f"template_{i+1}" for i in range(max_len)]

for i, col in enumerate(template_cols):
    df[col] = df["concept_templates_list"].apply(lambda x: x[i] if i < len(x) else "")

# === Remove any leading/trailing quotes from expanded template columns ===
for col in template_cols:
    df[col] = df[col].astype(str).str.strip(" '\"")


# === Drop helper column ===
df.drop(columns=["concept_templates_list"], inplace=True)

# === Save formatted CSV ===
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved formatted CSV to {OUTPUT_FILE}")
print(f"Created {len(template_cols)} template columns.")
