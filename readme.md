# Debias It Yourself (DIY)

**Debias It Yourself (DIY)** is a framework for **cognitive debiasing in large language models (LLMs)**.  
It leverages cognitive science to evaluate and mitigate social biases through structured reasoning, prompt-based interventions, and instruction-tuning.

---

## Overview

DIY provides an end-to-end pipeline to
- Apply **cognitive debiasing strategies** (e.g., stereotype replacement, counter-imaging, perspective-taking, individuating, positive contact).
- Evaluate the effects of different strategies across multiple demographic dimensions.
- Generate processed data and analysis outputs for further benchmarking.

---

## 📂 Repository Structure

```
DebiasItYourself/
│
├── data/
│   ├── BBQ/                      # Original BBQ dataset clone
│   ├── holisticbias/             # HolisticBias descriptors
│   ├── processed_bbq_all.csv     # Unified processed dataset
│
├── figures/                      # Visual assets and plots
│
├── outputs/                      # Generated instances and model outputs
│   ├── 1_generations/            # Seed and debiased instance generations
│   ├── 2_base_models/            # Raw outputs from base models
│   ├── 3_baselines/              # Comparison baseline results
│   ├── 4_incontext/              # In-context evaluation results
│   ├── 5_finetuning/             # Fine-tuned model outputs
│   └── 6_instructiontuning/      # Instruction-tuning outputs
│
├── results/                      # Aggregated evaluation scores and metrics
│   ├── 2_base_models/
│   ├── 3_baselines/
│   ├── 4_incontext/
│   ├── 5_finetuning/
│   └── 6_instructiontuning/
│
├── src/                          # Source code for each pipeline stage
│   ├── 1_process_data/           # Data preprocessing scripts
│   ├── 2_generations/            # Debiased data generation scripts
│   ├── 3_experiments/            # Evaluation and fine-tuning scripts
│   └── 4_slurmjobs/              # SLURM job submission scripts
│
├── readme.md                     
```


## 🧩 Prerequisites

- Python ≥ 3.10  
- PyTorch ≥ 2.0  
- Transformers (Hugging Face)  
- Datasets  
- Pandas, NumPy  
- tqdm, wandb (optional for tracking)

## ⚙️ Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/chahatraj/DebiasItYourself.git
   cd DebiasItYourself
   ```

2. **Clone the BBQ dataset**

   ```bash
   git clone https://github.com/nyu-mll/BBQ.git data/BBQ
   ```

3. **Generate the processed BBQ file**

   ```bash
   cd src/1_process_data
   python process_bbq.py
   ```

   This will create `data/processed_bbq_all.csv`.

---

## 🤓 Running DIY

Our supported strategies include:

* `stereotype_replacement`
* `counter_imaging`
* `individuating`
* `perspective_taking`
* `positive_contact`

---

## 📊 Outputs

Evaluation files will be saved under:

```
outputs/
```

Each experiment stores:

* Raw model outputs
* Processed JSON/CSV summaries

---

## 📊 Results

Result files will be saved under:

```
results/
```

Bias metrics and scores.

---

# Models and Hyperparameters

This section describes the models, configurations, and hyperparameters used in the **Debias It Yourself (DIY)** framework across generation, debiasing, and evaluation stages.

---

### 🧩 Base and Debiasing Models

| Model Key   | Hugging Face Model                  | Size | Quantization                 | Cache Directory                                    |
| ----------- | ----------------------------------- | ---- | ---------------------------- | -------------------------------------------------- |
| `llama_8b`  | `meta-llama/Llama-3.1-8B-Instruct`  | 8B   | 4-bit (`BitsAndBytesConfig`) | `/scratch/craj/model_cache/llama-3.1-8b-instruct`  |
| `llama_70b` | `meta-llama/Llama-3.3-70B-Instruct` | 70B  | 4-bit (`BitsAndBytesConfig`) | `/scratch/craj/model_cache/llama-3.3-70b-instruct` |
| `aya_8b`    | `CohereForAI/aya-expanse-8b`        | 8B   |                              | `/scratch/craj/model_cache/aya-expanse-8b`         |
| `qwen_32b`  | `Qwen/QwQ-32B`                      | 32B  |                              | `/scratch/craj/model_cache/qwen-32b`               |

All models are loaded via `AutoTokenizer` and `AutoModelForCausalLM` with **device mapping set to `auto`** for GPU allocation.
Padding tokens are automatically assigned to `eos_token` if not defined.

---

### ⚙️ Quantization Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

Used across inference, generation, and fine-tuning scripts to enable efficient memory usage on GPUs during high-volume evaluations.

---

### 🧩 Generation Stage

**Seed Instance Generation (`5_gen_seed_instances.py`)**
Objective: Generate 3 biased versions per concept–identity pair → Opinion / Action / Event

**Sampling:** `do_sample=True`

**Parameters:**

```python
max_new_tokens = 400
temperature = 0.7
top_p = 0.95
repetition_penalty = 1.2
```

Prompt Output Format: Structured numbered lines with `<END>` marker.

**Debiased Instance Generation (`6_gen_debiased_instances.py`)**
Objective: Apply cognitive debiasing strategies — *Stereotype Replacement, Counter-Imaging, Individuating, Perspective Taking, Positive Contact*
Variants: zero-shot and one-shot (with example)
Same decoding parameters as seed generation.

---

### 🧩 Fine-Tuning Stage

**Fine-Tuning (`7_finetune_llama.py`)**
Method: LoRA fine-tuning using `trl.SFTTrainer`

Objective: Train LLaMA-based models on debiased pairs *(biased → debiased)*

**Prompt Format:**

```text
<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. ...
<</SYS>>

### Instruction:
{debiasing_strategy_prompt}

### Input:
{biased_instance}

### Debiased Response:
[/INST]
{debiased_instance}
```

**Training Configuration:**

| Hyperparameter         | Value                 |
| ---------------------- | --------------------- |
| Epochs                 | 3                     |
| Batch Size             | 4                     |
| Learning Rate          | 2e-4 (default in TRL) |
| Optimizer              | AdamW                 |
| LoRA Rank              | 8                     |
| LoRA Alpha             | 16                    |
| Gradient Checkpointing | Enabled               |
| Mixed Precision        | FP16                  |
| WandB Logging          | Enabled               |
| Seed                   | 42                    |

**Fine-tuned Model Repositories:**

| Strategy               | Repository                                  |
| ---------------------- | ------------------------------------------- |
| Stereotype Replacement | `chahatraj/cognitive_stereotypereplacement` |
| Counter-Imaging        | `chahatraj/cognitive_counterimaging`        |
| Individuating          | `chahatraj/cognitive_individuating`         |
| Perspective Taking     | `chahatraj/cognitive_perspectivetaking`     |
| Positive Contact       | `chahatraj/cognitive_positivecontact`       |
| Combined               | `chahatraj/cognitive_collaborative`         |

---

### 🧪 Inference and Evaluation

**1️⃣ Base Model Evaluation (`1_bbq_inference.py`)**
Dataset: BBQ

Prompt Format:

```text
{context}
{question}
A. {ans0}
B. {ans1}
C. {ans2}
Answer:
```

Prediction Method: `predict_proba()` *(CAPPR library)*
Batch Size: `1`

**Metrics:**

* Accuracy
* Bias Score (`sDIS`, `sAMB`)
* Ambiguous vs. Disambiguated Accuracy

**2️⃣ In-Context Evaluation (`3_bbq_incontext.py`)**
Objective: Evaluate debiasing strategies without fine-tuning
Prompt Modes: `strategy_first`, `testing_first`, `revise`
Shots: `0 / 1 / 2 / 5` examples
Strategies Tested: 5 debiasing strategies
Prefix Variants: short, long, define explanations

**3️⃣ Fine-Tuned Model Evaluation (`ft_inference.py`)**
Uses LoRA-adapted models from Hugging Face Hub
Prediction Method: `predict_proba()` on multiple BBQ bias dimensions
Batch Size: 1
Output: `.csv` with probabilities, predicted label, and metadata

**4️⃣ Metrics Computation (`2_bbq_eval.py`)**
Calculates:

* Accuracy, Accuracy_ambig, Accuracy_disambig
* Bias_score_disambig (`sDIS`)
* Bias_score_ambig (`sAMB`)
* N_total, N_ambig, N_disambig

**Bias Scoring:**

```python
sDIS = ((n_biased / n_total) * 2 - 1)
sAMB = (1 - accuracy) * sDIS
```

---

### 🔬 Random Seed and Reproducibility

All stages set deterministic seeds:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

This ensures consistent generation, fine-tuning, and evaluation results across runs.

---

### 🧾 Summary

| Stage       | Script                                                        | Model(s)           | Purpose                         |
| ----------- | ------------------------------------------------------------- | ------------------ | ------------------------------- |
| Generation  | `5_gen_seed_instances.py`                                     | LLaMA / Aya / Qwen | Create biased templates         |
| Debiasing   | `6_gen_debiased_instances.py`                                 | Same               | Apply reasoning-based debiasing |
| Fine-tuning | `7_finetune_llama.py`                                         | LLaMA 8B / 70B     | Train on debiased pairs         |
| Inference   | `1_bbq_inference.py`, `3_bbq_incontext.py`, `ft_inference.py` | All                | Evaluate bias performance       |
| Evaluation  | `2_bbq_eval.py`                                               | N/A                | Compute bias metrics            |

---

🧩 **In summary**, the framework integrates LLaMA, Aya, and Qwen families in quantized settings, applies structured cognitive debiasing strategies, fine-tunes models with LoRA adapters, and evaluates bias reduction across base, in-context, and fine-tuned conditions.
