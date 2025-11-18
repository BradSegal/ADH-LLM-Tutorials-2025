# Architecture and Design Decisions

## 1. Overview

This document outlines the high-level architecture and key design decisions for the Digital Health Tutorial Repository. Its purpose is to provide a stable, guiding reference for all development, ensuring consistency and adherence to the project's core principles.

The system is designed as a **hybrid library/notebook architecture**. This model separates the robust, reusable "engine" of the project from the user-facing, narrative-driven "interface."

*   **The `core` Library (The Engine):** A production-quality, fully-tested, and well-documented Python library. It contains all logic for data processing, model definitions, training, evaluation, and visualization. It is the single source of truth for all complex operations.

*   **The `notebooks/` (The Interface):** A collection of Jupyter notebooks that serve as the primary educational interface. They are designed to be "thin," containing minimal code. Their primary role is to import functionality from the `core` library, orchestrate experiments, and present the results within a clear, didactic narrative.

This separation is the foundational principle of the entire project.

## 2. Key Architectural Decisions (ADRs)

The following are the critical, high-impact decisions that define the project's structure and behavior.

---

### ADR-001: Hybrid Library/Notebook Structure

*   **Decision:** We will implement a strict separation between a `core` Python library and the user-facing Jupyter notebooks. Notebooks will import from `core` but will not contain complex logic, class definitions, or function definitions (with the exception of simple, notebook-specific helper functions).
*   **Justification:**
    *   **DRY (Don't Repeat Yourself):** This is the primary driver. Logic for data loading, model training, and evaluation is identical across multiple notebooks. Implementing it once in `core` prevents code duplication, making the system vastly easier to maintain, debug, and extend. A single bug fix in the `core` trainer is instantly propagated to all notebooks.
    *   **Testability:** A pure Python library (`core`) is easily and robustly testable with standard tools like `pytest`. Notebooks are notoriously difficult to unit test. By concentrating logic in the library, we can achieve high test coverage and reliability.
    *   **Clarity & Focus:** This separation allows the notebooks to focus on their primary purpose: teaching and storytelling. Cells are clean, readable, and focus on the "what" and "why," while the `core` library handles the complex "how."

---

### ADR-002: Config-Driven Experiments

*   **Decision:** All experiments (model hyperparameters, training settings, data paths) will be defined in simple, human-readable YAML files located in the `configs/` directory. The notebooks will load these configurations and pass them to the `core` library.
*   **Justification:**
    *   **Decoupling:** This decouples the configuration of an experiment from the execution code. Students can easily modify hyperparameters without needing to hunt through Python code.
    *   **Scaffolded Learning:** It provides a "low-code" entry point for experimentation. A beginner can change the learning rate or number of epochs in the YAML file and observe the impact, a key learning objective.
    *   **Reproducibility:** A YAML file is a complete, version-controllable record of the parameters used to achieve a specific result.

---

### ADR-003: Single Real-World Dataset (PhysioNet 2019)

*   **Decision:** The project will exclusively use the PhysioNet/CinC 2019 Sepsis Challenge dataset. The alternative of providing a bundled synthetic dataset has been rejected.
*   **Justification:**
    *   **Realism:** A core goal is to teach the practical challenges of working with real clinical data. The PhysioNet dataset provides this, with its characteristic issues of missingness, irregular sampling, and noise. A synthetic dataset, while convenient, would mask these critical learning opportunities.
    *   **Focus:** Committing to a single, high-quality dataset allows us to focus development effort on building a robust and well-documented data pipeline for it, rather than dividing effort.
    *   **Reduced Scope:** Generating a *pedagogically useful* synthetic dataset is a non-trivial project in its own right. Using an existing, well-benchmarked dataset eliminates this risk and complexity. The friction of requiring a PhysioNet account is an acceptable trade-off for the gain in realism.

---

### ADR-004: Local LLM Interaction via `litellm`

*   **Decision:** For the LLM tutorials, we will use the `litellm` library to provide an OpenAI-compatible interface to a locally-run Hugging Face `transformers` model. We will not use a dedicated serving engine like `vLLM` as the primary tool.
*   **Justification:**
    *   **Pedagogical Purity:** The primary learning objective is to teach the *OpenAI API contract* (the structure of requests, messages, tools), which is the de facto industry standard. `litellm` allows us to teach this pattern directly and simply.
    *   **Robustness & Simplicity:** `litellm` runs in the same process as the notebook. This avoids the significant complexity of managing a separate client-server architecture inside Colab. Debugging is vastly simpler (a standard Python traceback) compared to debugging a failed server process, which is a major source of user friction. This directly supports the "Fail Fast, Fail Loudly" principle.
    *   **Generalizability:** The exact same student code written using `litellm` can be pointed at the actual OpenAI API (or hundreds of others) by changing a single model string and adding an API key. This is a powerful demonstration of API abstraction.

---

### ADR-005: Idempotent Notebook Bootstrap

*   **Decision:** Every notebook will begin with a standardized, idempotent setup cell. This cell will detect if the repository and dependencies are present and will automatically clone/install them if they are not.
*   **Justification:**
    *   **User Experience:** This is critical for the "Open in Colab" workflow. It removes all manual setup friction for the user, ensuring the environment is correctly configured with a single click.
    *   **Reliability:** It makes the notebook environment predictable and reproducible. The idempotency (fast re-runs) ensures that re-running the cell does not cause unnecessary delays. This supports the "Principle of Least Surprise."

## 3. System Component Diagram

The following diagram illustrates the flow of control and information within the system during a typical model training run.

```mermaid
graph TD
    subgraph "Jupyter Notebook (User Interface)"
        A[User Runs Cell] --> B{Load YAML Config};
        B --> C{Instantiate Trainer, Model, Data};
        C --> D[Call trainer.fit()];
        D --> E{Render Plots & Metrics};
    end

    subgraph "core Library (The Engine)"
        C --> F[configs/transformer.yaml];
        C --> G[core.train.Trainer];
        C --> H[core.models.Transformer];
        C --> I[core.data.loaders];
    end

    subgraph "Data Pipeline"
        I --> J{Load Processed Data (.parquet)};
        J -- If not exists --> K{Parse Raw Data (.psv)};
        K -- If not exists --> L[Download from PhysioNet];
    end

    G --> H;
    G --> I;

    style A fill:#D5E8D4,stroke:#82B366
    style E fill:#D5E8D4,stroke:#82B366
```

For a deeper dive into the training stack—including every supported `TrainConfig` field and the `Trainer`'s runtime guarantees—see [docs/TRAINING_ENGINE.md](./TRAINING_ENGINE.md).
