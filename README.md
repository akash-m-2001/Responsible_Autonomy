**Responsible Autonomy for Prompt Injection Defense**

  This repository contains the implementation and analysis for the CS562 final project on trust-aware prompt injection defense.
  The project reframes prompt injection defense as a runtime governance problem, extending beyond binary detection to policy-based control under uncertainty.

**Project Overview**

  Large Language Models (LLMs) are vulnerable to prompt injection attacks, where malicious user inputs attempt to override system instructions or safety constraints.
  Most existing defenses rely on binary allow/block decisions, which are brittle in real-world autonomous systems.

  This project introduces a Responsible Autonomy framework that:

  Detects prompt injection using a discriminative classifier

  Converts model confidence into a continuous trust score

  Maps trust to interpretable runtime policies: Allow, Escalate, Block

  Supports both stateless and stateful (multi-turn) governance

  Repository Structure
.
01_dataset_construction_and_normalization.ipynb
02_roberta_prompt_injection_training.ipynb
03_stateless_runtime_governance_roberta.ipynb
04_stateful_runtime_governance_multiturn.ipynb
README.md
.gitignore

**Notebooks**
**1. Dataset Construction**

01_dataset_construction_and_normalization.ipynb

Aggregates multiple public prompt injection datasets

Normalizes schema across sources

Produces final train / validation / test splits
**Prerequisites**

To run the notebooks and reproduce the experiments, the following environment is required.

**Software**

Python 3.9+

Jupyter Notebook or JupyterLab or Any environment supports notebooks

**Core Libraries**

numpy

pandas

scikit-learn

matplotlib

seaborn

torch

transformers

datasets

**Optional (Recommended)**

CUDA-enabled GPU for model training

NVIDIA drivers compatible with PyTorch (if using GPU)

**Installation**
All required packages can be installed using:

pip install numpy pandas scikit-learn matplotlib seaborn torch transformers datasets

**Hardware Notes**

Model training (Notebook 02) benefits significantly from GPU acceleration.

Runtime governance notebooks (Notebooks 03 and 04) can be run on CPU.

**Execution Order**

To ensure correct dependencies between steps, run notebooks in the following order:

01_dataset_construction_and_normalization.ipynb
02_roberta_prompt_injection_training.ipynb
03_stateless_runtime_governance_roberta.ipynb
04_stateful_runtime_governance_multiturn.ipynb

**Environment Warnings**

You may encounter warnings related to HuggingFace tokenizers parallelism when running in notebook environments. These warnings do not affect correctness and can be safely ignored or suppressed by setting:

export TOKENIZERS_PARALLELISM=false

**Outputs:**

final_train_dataset.csv

final_test_dataset.csv

**2. Model Training**

02_roberta_prompt_injection_training.ipynb

Fine-tunes a RoBERTa-base discriminative classifier

Binary classification: benign vs injection

Loss: Binary Cross Entropy

**Produces:**
Classification metrics (accuracy, precision, recall, F1)

3. Stateless Runtime Governance

03_stateless_runtime_governance_roberta.ipynb

Applies the trained classifier to prompts independently

**Computes:**

Injection probability

Trust score = 1 − P(prompt injection)

Policy decision (Allow / Escalate / Block)

**Generates:**

Trust score distributions

Policy distributions

Policy confusion matrices

Calibration plots

**4. Stateful Runtime Governance**

04_stateful_runtime_governance_multiturn.ipynb

Extends stateless governance with multi-turn context

Aggregates trust across recent prompts using conservative rules

Demonstrates reduced attack success under gradual probing

**Compares:**

Stateless vs Stateful policy outcomes

Escalation rates and safety tradeoffs

**Outputs**

The folders below contain generated CSV artifacts with per-prompt predictions and policy decisions:

responsible_autonomy_outputs/

responsible_autonomy_outputs_notebook4/

These files are not included in the repository due to size considerations.
All results can be fully reproduced by running Notebooks 3 and 4.

Representative figures derived from these outputs are included in the final report and poster.

Key Results

Strong generalization across large-scale datasets

Misclassifications cluster near decision boundaries

Escalation enables conservative handling of uncertainty

Stateful governance reduces risk from gradual adversarial probing

**Reproducibility**
To reproduce results:

Run notebooks in order:

01 → 02 → 03 → 04

Due to Github free tier size limits, I am unable to upload datasets and model but can be reproduced by using those notebooks.
Generated CSV outputs will appear in the output folders

Figures in the report are directly derived from these outputs

Ethical Considerations

This work emphasizes explicit uncertainty handling rather than overconfident automation.
Escalation policies enable safer deployment by deferring ambiguous cases to additional safeguards rather than executing potentially harmful actions.

**Course Information**
Course: CS562 – Adv Top in Sec, Priv and ML

Institution: University of Illinois Urbana-Champaign

Term: Fall 2025

Author: Akash Muthukumar
