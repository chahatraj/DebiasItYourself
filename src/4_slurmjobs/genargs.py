# generate_args.py
strategies = [
    "stereotype_replacement",
    "counter_imaging",
    "individuating",
    "perspective_taking",
    "positive_contact"
]

shots = ["zero"] #, "one", "two", "five"]
format_modes = ["strategy_first", "testing_first"]
prompt_versions = ["define"] #["long", "define"]
source_files = [
    "Age.jsonl",
    "Disability_status.jsonl",
    "Gender_identity.jsonl",
    "Nationality.jsonl",
    "Physical_appearance.jsonl",
    "Race_ethnicity.jsonl",
    "Religion.jsonl",
    "SES.jsonl",
    "Sexual_orientation.jsonl"
]

with open("s2.txt", "w") as f:
    for strategy in strategies:
        for shot in shots:
            for format_mode in format_modes:
                for prompt_version in prompt_versions:
                    for source_file in source_files:
                        line = (f"--model llama_8b "
                                f"--format_mode {format_mode} "
                                f"--shot {shot} "
                                f"--prompt_version {prompt_version} "
                                f"--strategy {strategy} "
                                f"--source_file {source_file}\n")
                        f.write(line)

print("✅ Created all_combinations.txt.")
