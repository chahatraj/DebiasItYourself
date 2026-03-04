# Baseline Methods, Metrics, and Interpretation

This table summarizes the baseline paper methods implemented under `/scratch/craj/diy/src/5_baselines`, what each method does in this repo, and how to interpret its reported metrics.

## Shared Baseline Entrypoints

- DeCAP: `/scratch/craj/diy/src/5_baselines/1_decap_evaluate.py`
- BIASEDIT train: `/scratch/craj/diy/src/5_baselines/2_biasedit_train.py`
- BIASEDIT evaluate: `/scratch/craj/diy/src/5_baselines/2_biasedit_evaluate.py`
- DPO train: `/scratch/craj/diy/src/5_baselines/3_dpo_train.py`
- DPO evaluate: `/scratch/craj/diy/src/5_baselines/3_dpo_evaluate.py`
- FairSteer train: `/scratch/craj/diy/src/5_baselines/4_fairsteer_train.py`
- FairSteer evaluate: `/scratch/craj/diy/src/5_baselines/4_fairsteer_evaluate.py`
- LFTF train: `/scratch/craj/diy/src/5_baselines/5_lftf_train.py`
- LFTF evaluate: `/scratch/craj/diy/src/5_baselines/5_lftf_evaluate.py`
- PEFT train: `/scratch/craj/diy/src/5_baselines/6_peft_train.py`
- PEFT evaluate: `/scratch/craj/diy/src/5_baselines/6_peft_evaluate.py`
- BiasFreeBench evaluate: `/scratch/craj/diy/src/5_baselines/7_biasfreebench_evaluate.py`
- BBA train: `/scratch/craj/diy/src/5_baselines/8_bba_train.py`
- BBA evaluate: `/scratch/craj/diy/src/5_baselines/8_bba_evaluate.py`
- CAL train: `/scratch/craj/diy/src/5_baselines/9_cal_train.py`
- CAL evaluate: `/scratch/craj/diy/src/5_baselines/9_cal_evaluate.py`
- Debias-NLG train: `/scratch/craj/diy/src/5_baselines/10_debias_nlg_train.py`
- Debias-NLG evaluate: `/scratch/craj/diy/src/5_baselines/10_debias_nlg_evaluate.py`
- MBIAS train: `/scratch/craj/diy/src/5_baselines/11_mbias_train.py`
- MBIAS evaluate: `/scratch/craj/diy/src/5_baselines/11_mbias_evaluate.py`
- Debias-LLMs train: `/scratch/craj/diy/src/5_baselines/12_debias_llms_train.py`
- Debias-LLMs evaluate: `/scratch/craj/diy/src/5_baselines/12_debias_llms_evaluate.py`
- Reduce Social Bias evaluate: `/scratch/craj/diy/src/5_baselines/13_reduce_social_bias_evaluate.py`
- Dataset folders (`bbq/`, `crowspairs/`, `stereoset/`) keep thin compatibility wrappers so existing script paths still work.

| Method | Implemented Technique in This Repo | Benchmark Metrics Reported | Metric Interpretation Scale | Variant(s) Run | Version(s) Actually Run and Results Found |
|---|---|---|---|---|---|
| BIASEDIT | LoRA fine-tuning that minimizes stereotype vs anti-stereotype likelihood gap (squared gap objective); dataset-specific pair conversion. | CrowS-Pairs: `stereotype_preference_pct`, `mean_stereo_prob_norm`, `mean_anti_prob_norm`<br>StereoSet: `LM Score`, `SS Score`, `ICAT Score`<br>BBQ: `Accuracy`, `Accuracy_ambig`, `Accuracy_disambig`, `Bias_score_disambig`, `Bias_score_ambig` | CrowS-Pairs: `stereotype_preference_pct` closer to 50 is better; `mean_stereo_prob_norm` lower is better; `mean_anti_prob_norm` higher is better.<br>StereoSet: `LM Score` higher is better; `SS Score` closer to 50 is better; `ICAT Score` higher is better.<br>BBQ: Accuracy metrics higher is better; bias scores are best near 0 (smaller absolute value is better). | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_biasedit_crowspairs_all`<br>StereoSet: `llama_8b_biasedit_stereoset_all`<br>BBQ file exists (`bbq_eval_llama_8b_biasedit_all.csv`), but `Model` column is `Age` (tagging issue in output). |
| DPO (LLM-SBM style) | Preference-aware DPO: chosen = anti/correct, rejected = stereotype/biased target; weighted by confidence `delta` and reference log-prob margins. | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_dpo_crowspairs_all`<br>StereoSet: `llama_8b_dpo_stereoset_all`<br>BBQ: `llama_8b_dpo` (`bbq_eval_llama_8b_dpo_all.csv`). |
| FairSteer | Inference-time activation steering pipeline: train BAD classifiers per layer, select layer, compute DSV steering vector, apply debiasing at eval time. | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS/StereoSet: `llama_8b_fairsteer_steered`<br>BBQ: `llama_8b_fairsteer` (`bbq_eval_llama_8b_fairsteer_all.csv`). |
| LFTF | Two-stage method: locate sensitive layer with BMI, then fine-tune selected layer with anti-stereotype preference objective (dataset-adapted). | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks with `llama_8b_lftf` (`*_lftf*.csv`). |
| PEFT (Bias-aware) | Bias-aware LoRA training: anti/chosen LM loss + balancing losses (`L_bal1`, `L_bal2`) with target-layer selection heuristics. | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_peft_crowspairs_all`<br>StereoSet: `llama_8b_peft_stereoset_all`<br>BBQ file exists (`bbq_eval_llama_8b_peft_all.csv`), but `Model` column is `Age` (tagging issue in output). |
| BiasFreeBench | Prompt-only debiasing evaluation with variants (`vanilla`, `self-aware`, `self-help`, `self-reflection`, `cot-debias`). | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `self-reflection` | Found in all 3 benchmarks, only `self-reflection` variant was run.<br>Model tag in outputs: `llama_8b_biasfreebench_self_reflection`. |
| BBA | Pairwise LoRA training via shared pairwise trainer with BBA-style weighting (`gap_mse_weight` emphasized). | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_bba_crowspairs_all`<br>StereoSet: `llama_8b_bba_stereoset_all`<br>BBQ: `llama_8b_bba_bbq_all`. |
| CAL | Contrastive pairwise LoRA variant with stronger pair preference and positive margin (`pair_pref_weight`, `margin`). | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_cal_crowspairs_all`<br>StereoSet: `llama_8b_cal_stereoset_all`<br>BBQ: `llama_8b_cal_bbq_all`. |
| Debias-NLG (CDA) | Counterfactual-data-augmentation-style pairwise LoRA, enabled via nonzero `cda_weight` (gender swap augmentation in shared trainer). | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_debias_nlg_crowspairs_all`<br>StereoSet: `llama_8b_debias_nlg_stereoset_all`<br>BBQ: `llama_8b_debias_nlg_bbq_all`. |
| MBIAS | Hybrid pairwise LoRA objective combining anti LM, preference, margin, gap-MSE, and CDA components. | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_mbias_crowspairs_all`<br>StereoSet: `llama_8b_mbias_stereoset_all`<br>BBQ: `llama_8b_mbias_bbq_all`. |
| Debias-LLMs | Template/pairwise LoRA baseline with anti LM + pairwise preference terms (no CDA/gap-MSE in default strategy). | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `default` | Found in all 3 benchmarks.<br>CrowS: `llama_8b_debias_llms_crowspairs_all`<br>StereoSet: `llama_8b_debias_llms_stereoset_all`<br>BBQ: `llama_8b_debias_llms_bbq_all`. |
| Reduce Social Bias in LLMs | Prompt-based debiasing (`system1`, `system2`, `cot`) without additional fine-tuning. | CrowS-Pairs, StereoSet, BBQ (same metric families as above) | Same interpretation as above for each benchmark family. | `system2` | Found in all 3 benchmarks, only `system2` variant was run.<br>Model tag in outputs: `llama_8b_reduce_social_bias_system2`. |
| DeCAP | Context-adaptive prompt generation for BBQ QA, including ambiguity-sensitive prompting and demonstration retrieval. | BBQ: `Accuracy`, `Accuracy_ambig`, `Accuracy_disambig`, `Bias_score_disambig`, `Bias_score_ambig` | Accuracy metrics higher is better; bias scores best near 0 (smaller absolute value is better). | `all` | Found for BBQ only: `bbq_eval_llama_8b_decap_all.csv` with model tag `decap_all`. |
| ReGiFT | Reasoning-guided fine-tuning: extract correct reasoning traces from a reasoning model, then LoRA-fine-tune target model on traces; evaluate on BBQ. | BBQ (ReGiFT eval script): `overall_acc`, `ambig_acc`, `disambig_acc` (percent) | Higher is better for all three reported accuracies. | `not run` | No ReGiFT result CSV found under `/scratch/craj/diy/results/3_baselines` at audit time. Model/traces artifacts exist under `/scratch/craj/diy/outputs/3_baselines/regift/`. |
| Self-Debiasing | Zero-shot multi-turn self-correction (`baseline`, `explanation`, `reprompting`) adapted to BBQ. | BBQ: `Accuracy`, `Accuracy_ambig`, `Accuracy_disambig`, `Bias_score_disambig`, `Bias_score_ambig` | Accuracy metrics higher is better; bias scores best near 0 (smaller absolute value is better). | `all (aggregated)` | Found for BBQ only: `bbq_eval_llama_8b_selfdebiasing_all.csv` (aggregated “all” run). File has `Model=Age` and only ambiguous-count rows (`N_disambig=0`), indicating output-format/tagging issues. |

## Benchmark Metric Reference

- CrowS-Pairs metrics are produced by `compute_metrics_from_scored` / `compute_crows_metrics` in baseline scripts.
- StereoSet metrics are produced by `stereoset_score` (`LM Score`, `SS Score`, `ICAT Score`).
- BBQ metrics are produced by `_compute_metrics_for_group` / equivalent method-specific metric functions.
- For BBQ bias scores, sign indicates direction of bias; magnitude indicates strength. Closer to `0` means less bias.

## Variant Run Summary

| Method | Variant(s) Run |
|---|---|
| BIASEDIT | `default` |
| DPO (LLM-SBM style) | `default` |
| FairSteer | `default` |
| LFTF | `default` |
| PEFT (Bias-aware) | `default` |
| BiasFreeBench | `self-reflection` |
| BBA | `default` |
| CAL | `default` |
| Debias-NLG (CDA) | `default` |
| MBIAS | `default` |
| Debias-LLMs | `default` |
| Reduce Social Bias in LLMs | `system2` |
| DeCAP | `all` |
| ReGiFT | `not run` |
| Self-Debiasing | `all (aggregated)` |

## Hyperparameters (Run Snapshot)

| Method | Hyperparameters (What Was Run) | Source |
|---|---|---|
| BIASEDIT | CrowS config: `epochs=3`, `batch_size=8`, `lr=1e-4`, `lambda_r=0.0`, `lora_r=8`, `lora_alpha=16`, `target_layers=[30,31]`.<br>StereoSet saved config shows `epochs=0`, `batch_size=2` (run artifact), with same LoRA/lr family.<br>BBQ per-category configs (e.g., `config_Age.json`): `epochs=3`, `batch_size=8`, `lr=1e-4`, `lambda_r=0.0`, `lora_r=8`, `lora_alpha=16`, `target_layers=[30,31]`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/biasedit/...` |
| DPO (LLM-SBM style) | `epochs=3`, `batch_size=4`, `lr=5e-5`, `beta=0.1`, `alpha=2.0`.<br>CrowS config includes `max_length=256`; StereoSet config same family; BBQ training script uses same core defaults. | Saved configs for CrowS/StereoSet; script defaults in `/scratch/craj/diy/src/5_baselines/3_dpo_train.py` |
| FairSteer | Inference-time steering (no gradient finetune epochs). Core settings: `num_examples_bad=800`, `num_pairs_dsv=110`, `seed=42`, `use_4bit=True`.<br>Selected steering layer from saved runs: CrowS `optimal_layer=6`, StereoSet `optimal_layer=20`, BBQ generic config `optimal_layer=21`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/fairsteer/...`; script defaults in `/scratch/craj/diy/src/5_baselines/4_fairsteer_train.py` |
| LFTF | `num_epochs=2`, `batch_size=16`, `learning_rate=2e-5`, `lora_r=8`, `lora_alpha=16`.<br>Saved optimal layers: CrowS `31`, StereoSet `0` (run artifact), generic/BBQ config `31`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/lftf/...` |
| PEFT (Bias-aware) | CrowS config: `epochs=3`, `batch_size=16`, `lr=5e-5`, `alpha=1.0`, `beta=1.0`, `gamma=1.0`, `target_layer=16`, `max_length=256`.<br>StereoSet saved config shows `epochs=0`, `batch_size=4` (run artifact) with same loss weights/target layer.<br>BBQ training script uses same core defaults (`epochs=3`, `batch_size=16`, `lr=5e-5`, `alpha=beta=gamma=1.0`). | Saved configs for CrowS/StereoSet under `/scratch/craj/diy/outputs/3_baselines/peft/...`; defaults in `/scratch/craj/diy/src/5_baselines/6_peft_train.py` |
| BiasFreeBench | Prompt-only; variant run was `self-reflection`.<br>Batch sizes in scripts: CrowS/StereoSet `batch_size=4`, BBQ `batch_size=8`. | Evaluate script: `/scratch/craj/diy/src/5_baselines/7_biasfreebench_evaluate.py` |
| BBA | `epochs=3`, `batch_size=4`, `lr=5e-5`, `max_length=256` (CrowS/StereoSet), `max_length=320` (BBQ), `lora_r=8`, `lora_alpha=16`, objective weights: `pair_pref_weight=1.0`, `gap_mse_weight=0.5`, `margin=0.0`, `cda_weight=0.0`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/bba/...` |
| CAL | `epochs=3`, `batch_size=4`, `lr=5e-5`, `max_length=256` (CrowS/StereoSet), `max_length=320` (BBQ), `lora_r=8`, `lora_alpha=16`, objective weights: `pair_pref_weight=1.5`, `gap_mse_weight=0.2`, `margin=0.2`, `cda_weight=0.0`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/cal/...` |
| Debias-NLG (CDA) | `epochs=3`, `batch_size=4`, `lr=5e-5`, `max_length=256` (CrowS/StereoSet), `max_length=320` (BBQ), `lora_r=8`, `lora_alpha=16`, objective weights: `pair_pref_weight=0.8`, `gap_mse_weight=0.0`, `margin=0.0`, `cda_weight=0.5`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/debias_nlg/...` |
| MBIAS | `epochs=3`, `batch_size=4`, `lr=5e-5`, `max_length=256` (CrowS/StereoSet), `max_length=320` (BBQ), `lora_r=8`, `lora_alpha=16`, objective weights: `pair_pref_weight=1.2`, `gap_mse_weight=0.1`, `margin=0.1`, `cda_weight=0.2`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/mbias/...` |
| Debias-LLMs | `epochs=3`, `batch_size=4`, `lr=5e-5`, `max_length=256` (CrowS/StereoSet), `max_length=320` (BBQ), `lora_r=8`, `lora_alpha=16`, objective weights: `pair_pref_weight=1.0`, `gap_mse_weight=0.0`, `margin=0.0`, `cda_weight=0.0`. | Saved config JSONs under `/scratch/craj/diy/outputs/3_baselines/debias_llms/...` |
| Reduce Social Bias in LLMs | Prompt-only; variant run was `system2`.<br>Batch sizes in scripts: CrowS/StereoSet `batch_size=4`, BBQ `batch_size=8`. | Evaluate script: `/scratch/craj/diy/src/5_baselines/13_reduce_social_bias_evaluate.py` |
| DeCAP | Inference constants: `SEED=42`, `TEMPERATURE=0.6`, `MAX_NEW_TOKENS=64`, `TOP_K_RETRIEVAL=5`, `ROUGE_THRESHOLD=0.35`.<br>Run mode recorded as `source_file=all`. | `/scratch/craj/diy/src/5_baselines/1_decap_evaluate.py` |
| ReGiFT | Not run end-to-end in baseline results, but training defaults in code: `num_traces=500`, `num_epochs=3`, `batch_size=2`, `learning_rate=2e-4`, `lora_r=16`, `lora_alpha=32`, `seed=42`, `lora_dropout=0.05`, targets `q_proj,k_proj,v_proj,o_proj`.<br>Trace extraction generation defaults: `max_new_tokens=400`, `temperature=0.6`, `top_p=0.9`. | `/scratch/craj/diy/src/5_baselines/14_regift_train.py` and adapter config under `/scratch/craj/diy/outputs/3_baselines/regift/models/adapter_config.json` |
| Self-Debiasing | Prompt-only multi-turn; recorded run mode `method=all` (aggregated).<br>Generation defaults: `temperature=1.0`; baseline answer `max_new_tokens=25`; explanation step `150`; reprompted answer `25`.<br>Parser defaults include `source_file=all`, `include_disambig=False`. | `/scratch/craj/diy/src/5_baselines/15_selfdebias_evaluate.py` |
