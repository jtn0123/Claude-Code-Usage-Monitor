# Agent Workflow

This repository includes automated tests and lint checks. Follow the steps below before committing changes.

## Setup

1. Create and activate a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

All tests live in the `tests/` folder. Run them from the project root with `pytest`:

```bash
pytest
```

The test suite should pass with no failures before you commit.

## Lint Rules

We enforce formatting, style and type checks. Configuration lives in `pyproject.toml`.

1. **Black** – formats code with a 100 character line length.
   ```bash
   black .            # auto format
   black --check .    # verify formatting
   ```
2. **Flake8** – runs static style checks (ignores `E203` and `W503`). The
   repository contains a `.venv` directory that should be excluded.
   ```bash
   flake8 --exclude=.venv .
   ```
3. **Mypy** – run type checks.
   ```bash
   mypy ccusage_monitor.py tests
   ```
4. **Pylint** – run additional static analysis.
   ```bash
   pylint ccusage_monitor.py
   ```

All lint commands should finish with no errors. Run them before committing any code.
