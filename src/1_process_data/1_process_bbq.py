#!/usr/bin/env python3
import os
import json
import pandas as pd

def process_bbq_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            add_meta = ex.get("additional_metadata", {})

            prompt = (
                f"Context: {ex.get('context')}\n"
                f"Question: {ex.get('question')}\n"
                f"A. {ex.get('ans0')}\n"
                f"B. {ex.get('ans1')}\n"
                f"C. {ex.get('ans2')}\n"
                f"Answer:"
            )

            data.append({
                # --- core identifiers ---
                "example_id": ex.get("example_id"),
                "question_index": ex.get("question_index"),
                "question_polarity": ex.get("question_polarity"),
                "context_condition": ex.get("context_condition"),
                "category": ex.get("category"),

                # --- question and answer content ---
                "context": ex.get("context"),
                "question": ex.get("question"),
                "ans0": ex.get("ans0"),
                "ans1": ex.get("ans1"),
                "ans2": ex.get("ans2"),
                "label": ex.get("label"),

                # --- metadata info ---
                "answer_info": ex.get("answer_info"),
                "subcategory": add_meta.get("subcategory"),
                "stereotyped_groups": add_meta.get("stereotyped_groups"),
                "version": add_meta.get("version"),
                "source": add_meta.get("source"),

                # --- convenience ---
                "prompt": prompt,
                "source_file": os.path.basename(path),
            })
    return data


if __name__ == "__main__":
    data_dir = "/scratch/craj/diy/data/BBQ/data"
    all_data = []

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".jsonl"):
            path = os.path.join(data_dir, fname)
            print(f"🔄 Processing {fname}")
            all_data.extend(process_bbq_file(path))

    df = pd.DataFrame(all_data)
    out_path = "/scratch/craj/diy/data/processed_bbq_all.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} rows → {out_path}")
    print(f"✅ Columns saved: {df.columns.tolist()}")