# Sequence Models and LLMs in Digital Health: A Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/01_problem_eda.ipynb)
[![CI Status](https://github.com/BradSegal/ADH-LLM-Tutorials-2025/actions/workflows/ci.yml/badge.svg)](https://github.com/BradSegal/ADH-LLM-Tutorials-2025/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Mission

This repository provides a robust, hands-on tutorial for master's-level students in Digital Health and related fields. It uses the real-world clinical problem of early sepsis prediction from ICU time-series data to teach the fundamental principles of sequence modeling (RNNs, LSTMs, Transformers) and the practical application of Large Language Models (LLMs) in a healthcare context. The project prioritizes conceptual understanding, practical implementation skills, and a strong foundation in software engineering best practices.

## Learning Objectives

Upon completing this tutorial series, you will be able to:

*   **Implement and Train** various sequence models (GRU, LSTM, Transformer) for a time-series classification task using PyTorch.
*   **Critically Compare** models based on standard performance metrics (AUROC, AUPRC), calibration, and fairness across subgroups.
*   **Visualize and Interpret** model behavior using techniques like attention maps and feature attributions via Captum.
*   **Utilize** word embeddings for core NLP tasks like semantic search and similarity ranking.
*   **Interact** with an LLM using a standard, OpenAI-compatible API to perform generation, structured data extraction, and tool use.

---

## For Students: Getting Started

The easiest way to get started is to run these notebooks directly in Google Colab. The notebooks are self-contained and will automatically clone this repository and install all required dependencies in the Colab environment.

**Just click any "Open in Colab" link in the table below to begin.**

### Tutorial Notebooks

The tutorials are designed to be completed in sequence, building upon concepts from the previous notebooks.

#### Part 1: Sequence Models for Sepsis Prediction

| Notebook | Description | Link |
| :--- | :--- | :--- |
| **01 - Problem & EDA** | Introduces the clinical problem of sepsis and performs Exploratory Data Analysis (EDA) on the PhysioNet 2019 dataset. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/01_problem_eda.ipynb) |
| **02 - RNN/GRU Baseline** | Implements and trains a Gated Recurrent Unit (GRU) model as a strong baseline for the sequence modeling task. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/02_rnn_baseline.ipynb) |
| **03 - LSTM Model** | Implements an LSTM model, comparing its performance and behavior to the GRU baseline. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/03_lstm.ipynb) |
| **04 - Transformer Model** | Implements an encoder-only Transformer model and analyzes its performance and attention mechanisms. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/04_transformer.ipynb) |
| **05 - Comparison & Explainability** | Compares all three models side-by-side and introduces model explainability techniques using Captum. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/05_compare_eval_explain.ipynb) |
| **06 - Bias, Calibration & Limits** | Explores critical issues of model fairness, calibration, and the ethical limitations of clinical AI. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/06_bias_calibration_limits.ipynb) |
| **06b - Hyperparameter Tuning** | Introduces automated hyperparameter optimization using Optuna to systematically improve model performance. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/06b_hyperparameter_tuning.ipynb) |

#### Part 2: LLM Integration and Use

| Notebook | Description | Link |
| :--- | :--- | :--- |
| **07 - Embeddings Basics** | Introduces the concept of embeddings by visualizing clinical terms in a 2D space. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/07_embeddings_basics.ipynb) |
| **08 - Similarity Search** | Builds a simple semantic search engine for a corpus of clinical notes using embeddings. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/08_similarity_search.ipynb) |
| **09 - OpenAI-Compatible LLM** | Uses a local LLM with an OpenAI-compatible API to demonstrate generation, structured output (JSON), and tool use. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/09_llm_openai_compatible.ipynb) |
| **10 - Model Inference** | Demonstrates how to save, load, and use a trained sepsis prediction model for inference on new data. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/ADH-LLM-Tutorials-2025/blob/master/notebooks/10_model_inference_and_saving.ipynb) |

---

## For Developers: Contributor Guide

This project is built with a strong emphasis on code quality, robustness, and maintainability. Our goal is to create a high-quality educational resource that exemplifies software engineering best practices.

All contributions will be held to these standards. Before contributing, please review the following documents:

*   **[CONTRIBUTING.md](./CONTRIBUTING.md):** The mandatory guide for our development workflow, code style, testing requirements, and quality checks.
*   **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md):** A document outlining the key architectural decisions and the overall design of the system.
*   **[docs/TRAINING_ENGINE.md](./docs/TRAINING_ENGINE.md):** How the reusable training configuration and `Trainer` work, including all validated hyperparameters and DataLoader contracts.

### Notebook runtime helper

Every notebook now calls `core.notebook.ensure_project_root()` in its first code cell. This guarantees relative paths such as `configs/gru.yaml` resolve no matter where the notebook is launched (Colab, VS Code, etc.). The helper also sets the environment variable `PYPROJECT_ROOT`, so subsequent cells can call it without redoing filesystem detection. If you author a new notebook, import the helper next to the other `core.*` imports and invoke it before reading configs.

### Regenerating sequence-model checkpoints

The saved weights used by notebooks 02–05 can be refreshed by running:

```bash
python scripts/generate_checkpoints.py
```

Pass `gru`, `lstm`, or `transformer` to train a subset. The script relies on the same configs and trainer that power the notebooks, so it is the authoritative way to validate that checkpoints were produced with the current code.

## Repository Structure

```
.
├── core/                   # The core, reusable Python library (data, models, trainer).
├── notebooks/              # All student-facing Jupyter notebooks.
├── configs/                # YAML files for configuring experiments (models, training, data).
├── tests/                  # Unit tests for the `core` library.
├── docs/                   # Developer documentation (Architecture, Data Pipeline).
├── .github/                # CI/CD workflows (e.g., automated testing and linting).
├── pyproject.toml          # Project dependencies and tool configuration.
└── README.md               # You are here.
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
