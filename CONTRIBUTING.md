# Contributing to the Digital Health Tutorial Repository

Thank you for your interest in contributing. This project is built to a high standard of engineering quality to provide a robust and maintainable educational resource. This document outlines the mandatory rules and workflow for all contributions.

## The Development Philosophy

All contributions are reviewed against and must adhere to our four core architectural principles. Before you write any code, internalize them.

1.  **DRY (Don't Repeat Yourself):** All logic must be abstracted and implemented once. If you find yourself copying and pasting code, you are doing it wrong. Stop and create a proper abstraction in the `core` library.
2.  **Principle of Least Surprise (POLS):** Code must be clear, predictable, and simple. Avoid clever tricks or implicit side effects. The behavior of a function should be obvious from its name and signature.
3.  **Strict Contracts:** All inputs and outputs must be explicitly defined and statically typed. Use the `typing` module for all function signatures and Pydantic for configuration models. All new code in the `core` library must pass `mypy --strict`.
4.  **Fail Fast, Fail Loudly:** Your code must be robust. Validate inputs, check for expected conditions, and raise explicit, informative exceptions as early as possible. Do not silently swallow errors or continue in an indeterminate state.

## Local Development Environment Setup

You must follow these steps to create a correct and isolated development environment.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/BradSegal/ADH-LLM-Tutorials-2025.git
    cd ADH-LLM-Tutorials-2025
    ```

2.  **Create and Activate a Virtual Environment:**
    We use `venv` for managing dependencies. This is not optional.
    ```bash
    # Create the virtual environment
    python3 -m venv .venv

    # Activate it (Linux/macOS)
    source .venv/bin/activate

    # Activate it (Windows PowerShell)
    # .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies:**
    Install the project in editable mode (`-e`) along with all development dependencies.
    ```bash
    pip install -e .[dev]
    ```
    *Note: The `[dev]` extra is defined in `pyproject.toml` and includes all tools required for testing and quality checks.*

## The Development Workflow: The Golden Path

All changes to the repository, from a minor typo fix to a major feature, MUST follow this exact sequence.

1.  **Create a Branch:** Create a new feature branch from the `main` branch. Name it descriptively, referencing the ticket or issue number.
    ```bash
    git checkout main
    git pull origin main
    git checkout -b feature/TICKET-42-add-new-metric
    ```

2.  **Implement Logic in `core`:** All core logic (data processing, model architecture, training loops, metrics) MUST be implemented within the `core` library (`core/...`). The notebooks are only for orchestration and narrative.

3.  **Write Unit Tests:** Any new function or class added to the `core` library MUST be accompanied by corresponding unit tests in the `tests/` directory. Bug fixes must include a regression test that fails before the fix and passes after.

4.  **Run Local Quality Checks:** Before committing, you MUST run all local quality checks from the root of the repository. **All checks must pass.**
    ```bash
    # 1. Format code with Black
    black .

    # 2. Format with Ruff's formatter
    ruff format .

    # 3. Lint with Ruff
    ruff check . --fix

    # 4. Run static type checking with MyPy
    mypy .

    # 5. Run all unit tests with Pytest
    pytest
    ```

5.  **Update Notebooks:** If your changes in `core` affect how a notebook is run or what it displays, update the relevant notebook(s) accordingly.

6.  **Commit and Push:** Use clear and concise commit messages.
    ```bash
    git add .
    git commit -m "feat(metrics): Add F1 score calculation"
    git push origin feature/TICKET-42-add-new-metric
    ```

7.  **Open a Pull Request:** Open a pull request against the `main` branch. The PR description must be clear and reference the original issue or ticket. All CI checks must pass before the PR will be considered for review.

## Code Style & Quality Mandates

*   **Formatting:** We use `black` and `ruff format` with their default configurations. This is a non-negotiable, automated standard.
*   **Linting:** We use `ruff`. All reported errors must be fixed.
*   **Typing:** All new code in the `core` library **MUST** be fully type-hinted and pass `mypy .` (which is configured to run in strict mode). Notebook code should be typed where it improves clarity, but `mypy` is not strictly enforced on notebooks.
*   **Docstrings:** All public modules, classes, and functions in the `core` library **MUST** have Google-style docstrings.

## Testing Requirements

*   The `core` library **MUST** maintain a minimum of **80% unit test coverage**. This is checked automatically.
*   Every new feature or bug fix in `core` **MUST** be accompanied by a corresponding test.
*   Run the fast test suite with `pytest -m "not api"` before pushing. Tests that hit local servers or external APIs are marked `@pytest.mark.api`; execute them with `pytest -m api` before releasing or when changing those components.
*   Notebooks are not unit tested. They are tested via a CI workflow that ensures they can execute from top to bottom without error.

## Pull Request Process

1.  Ensure your PR has a clear, descriptive title.
2.  The PR description should summarize the changes and link to the relevant issue/ticket.
3.  Your PR will not be reviewed until all automated CI checks (linting, testing, formatting) are passing.
4.  All PRs require at least one approving review from a project maintainer before being merged.
