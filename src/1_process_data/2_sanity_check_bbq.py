import pandas as pd
import ast

def sanity_check_bbq(df):
    print("✅ Starting sanity checks...\n")

    # === 0. Fix stringified lists ===
    df["stereotyped_groups"] = df["stereotyped_groups"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df["answer_info"] = df["answer_info"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # === 1. Basic shape ===
    print(f"🔹 Total examples: {len(df)}")
    print(f"🔹 Source files: {df['source_file'].nunique()}")
    print(f"🔹 Source file breakdown:\n{df['source_file'].value_counts()}\n")

    # === 2. Label values ===
    label_counts = df["label"].value_counts()
    print(f"🔹 Label value counts:\n{label_counts}\n")
    assert all(l in [0, 1, 2] for l in label_counts.index), "❌ Invalid label values found!"

    # === 3. Context condition check ===
    context_conditions = df["context_condition"].unique()
    print(f"🔹 Unique context conditions: {context_conditions}\n")
    assert set(context_conditions).issubset({"ambig", "disambig"}), "❌ Unexpected context conditions!"

    # === 4. Missing values check ===
    essential_cols = ["context", "question", "ans0", "ans1", "ans2", "prompt", "label"]
    missing = df[essential_cols].isnull().sum()
    print("🔹 Missing value check (should be 0):")
    print(missing)
    assert missing.sum() == 0, "❌ Missing values found!"

    # === 5. Prompt integrity check ===
    bad_prompts = df[~df["prompt"].str.endswith("Answer:")]
    print(f"\n🔹 Prompt format issues: {len(bad_prompts)}")
    if len(bad_prompts) > 0:
        print(bad_prompts[['example_id', 'prompt']].head())

    # === 6. Stereotyped groups format ===
    bad_stereotypes = df[~df["stereotyped_groups"].apply(lambda x: isinstance(x, list))]
    print(f"\n🔹 Non-list stereotyped_groups entries: {len(bad_stereotypes)}")

    # === 7. Duplicate answers check ===
    dup_options = df[(df['ans0'] == df['ans1']) | (df['ans0'] == df['ans2']) | (df['ans1'] == df['ans2'])]
    print(f"\n🔹 Examples with duplicate answer options: {len(dup_options)}")

    # === 8. Valid answer_info structure ===
    def is_valid_answer_info(entry):
        if not isinstance(entry, dict):
            return False
        for k in ['ans0', 'ans1', 'ans2']:
            if k not in entry or not isinstance(entry[k], list) or len(entry[k]) != 2:
                return False
        return True

    bad_answer_info = df[~df['answer_info'].apply(is_valid_answer_info)]
    print(f"🔹 Malformed answer_info entries: {len(bad_answer_info)}")

    # === 9. Label points to valid option ===
    bad_labels = df[~df['label'].isin([0, 1, 2])]
    print(f"🔹 Examples with invalid label indices: {len(bad_labels)}")

    print("\n✅ Sanity check completed.\n")


if __name__ == "__main__":
    df = pd.read_csv("/scratch/craj/diy/data/processed_bbq_all.csv")
    sanity_check_bbq(df)
