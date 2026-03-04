# How To Run `7_finetune_llama.py`

This guide documents the current CLI for:

- `/scratch/craj/diy/src/3_experiments/7_finetune_llama.py`

## Setup

Run from the experiments folder with your project venv:

```bash
cd /scratch/craj/diy/src/3_experiments
/home/craj/nanotron-env/bin/python 7_finetune_llama.py ...
```

## Quick Preview (No Training)

Use preview mode to inspect what is fed into the model (sys prompt, instruction, input, debiased response):

```bash
/home/craj/nanotron-env/bin/python 7_finetune_llama.py \
  --strategies stereotype_replacement \
  --versions all \
  --preview_examples 5 \
  --alpaca_ratio 0 \
  --skip_push
```

Preview mode exits before model loading/training.

## Selection Behavior

- If only `--versions` is provided, all strategies are used.
- If only `--strategies` is provided, all versions are used.
- If both are provided, exactly that strategy/version cross-product is used.
- Data is combined from existing files under:
  - `/scratch/craj/diy/outputs/1_generations/debiased_instances`
- Instruction is per-example (based on each example's strategy), unless forced with `--prompt_strategy`.

## Common Run Patterns

Single strategy, all versions:

```bash
/home/craj/nanotron-env/bin/python 7_finetune_llama.py \
  --strategies stereotype_replacement \
  --versions all \
  --max_debias_samples 1000 \
  --loss_mode response_only \
  --skip_push
```

All strategies, single version:

```bash
/home/craj/nanotron-env/bin/python 7_finetune_llama.py \
  --versions action \
  --max_debias_samples 1000 \
  --loss_mode response_only \
  --skip_push
```

All strategies, all versions:

```bash
/home/craj/nanotron-env/bin/python 7_finetune_llama.py \
  --strategies all \
  --versions all \
  --max_debias_samples 1000 \
  --loss_mode response_only \
  --skip_push
```

Filtered by fixed scenario and bias dimension:

```bash
/home/craj/nanotron-env/bin/python 7_finetune_llama.py \
  --versions event \
  --bias_dimension race_ethnicity \
  --scenario technology \
  --preview_examples 5 \
  --alpaca_ratio 0 \
  --skip_push
```

## Full CLI Reference

- `--file_key`
  - Values: `sr`, `ci`, `ind`, `pt`, `pc`
  - Default: `sr`

- `--input_file`
  - Values: path to `.jsonl` or `.csv`
  - Default: `None`

- `--strategies`
  - Values: comma-separated keys/full names, or `all`
  - Keys: `sr,ci,ind,pt,pc`
  - Full names: `stereotype_replacement,counter_imaging,individuating,perspective_taking,positive_contact`
  - Default: `None`

- `--versions`
  - Values: comma-separated `opinion,action,event` (also accepts `opinion_version,action_version,event_version`), or `all`
  - Default: `None`

- `--bias_dimension`
  - Values: `age,gender,nationality,physical_appearance,physical_disability,race_ethnicity,religion,sexual_orientation,socioeconomic_status`
  - Default: `None`

- `--bias_dimensions`
  - Values: comma-separated bias dimensions above, or `all`
  - Default: `None`

- `--scenarios`
  - Values: comma-separated fixed scenarios below, or `all`
  - Fixed values:
    - `art_and_leisure`
    - `economics`
    - `education`
    - `environment`
    - `healthcare`
    - `law_and_policy`
    - `media`
    - `sports`
    - `technology`
    - `workplace`
  - Default: `None`

- `--scenario`
  - Values: one fixed scenario from list above
  - Default: `None`

- `--identities`
  - Values: comma-separated identity strings, or `all`
  - Default: `None`

- `--concept_templates`
  - Values: comma-separated concept template strings, or `all`
  - Default: `None`

- `--concept_template`
  - Values: single concept template string
  - Default: `None`

- `--num_scenarios`
  - Values: integer `> 0`
  - Default: `None`

- `--num_identities`
  - Values: integer `> 0`
  - Default: `None`

- `--num_concepts`
  - Values: integer `> 0`
  - Default: `None`

- `--selection_seed`
  - Values: integer
  - Default: `42`

- `--cap_sampling_seed`
  - Values: integer
  - Default: `None` (auto-derived)

- `--pair`
  - Values: `opinion`, `action`, `event`, `all`
  - Default: `opinion`

- `--model`
  - Values: `llama_8b`, `llama_70b`, `aya_8b`, `qwen_32b`
  - Default: `llama_8b`

- `--prompt_strategy` (alias `--strategy`)
  - Values: `auto`, `stereotype_replacement`, `counter_imaging`, `individuating`, `perspective_taking`, `positive_contact`
  - Default: `auto`

- `--shot`
  - Values: `zero`, `one`, `two`, `five`
  - Default: `zero`

- `--output_dir`
  - Values: path
  - Default: `None` (auto-generated)

- `--epochs`
  - Values: integer `<= 3`
  - Default: `3`

- `--batch_size`
  - Values: integer
  - Default: `2`

- `--preview_examples`
  - Values: integer (`>0` prints examples and exits)
  - Default: `0`

- `--loss_mode`
  - Values: `full_sequence`, `response_only`
  - Default: `full_sequence`

- `--max_debias_samples`
  - Values: integer
  - Default: `None`

- `--alpaca_ratio`
  - Values: float
  - Default: `0.2`

- `--wandb_group`
  - Values: string
  - Default: `None`

- `--skip_push`
  - Flag (boolean)
  - Default: `False`

- `--resume_if_available`
  - Flag (boolean)
  - Default: `False`
