# Repository Guidelines

## Project Structure & Module Organization

This repository is an early-stage Python workspace for a 16-week curriculum covering LLM fine-tuning, RAG, and deployment.

- `main.py` is the current entrypoint and still contains a greeting stub.
- `src/` is empty; place reusable implementation modules here as checkpoints develop.
- `plans/llm-finetuning-rag-learning-plan.md` defines checkpoints `ckpt-00` through `ckpt-07`. Keep its HTML counterpart synchronized when editing curriculum content.
- `pyproject.toml` declares dependencies; `uv.lock` records resolved versions. `README.md` is currently blank.
- No tests or dedicated asset directory exist yet. `.claude-flow/` and `ruvector.db` are local tooling state, not application source.

## Build, Test, and Development Commands

Use Python 3.12, matching `.python-version`, and manage the environment with `uv`:

- `uv sync` installs the locked dependencies into `.venv/`.
- `uv run python main.py` runs the current entrypoint.
- `uv add <package>` adds a runtime dependency and updates dependency metadata.
- `uv run python -m compileall main.py src` checks Python syntax without running training jobs.

There is no configured build pipeline, test runner, formatter, or linter. Current dependencies are `transformers[torch]` and `peft`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for Python modules and functions, and `PascalCase` for classes. Keep entrypoints small and move reusable logic into `src/`. Add type hints to new public functions. Preserve checkpoint identifiers such as `ckpt-02` in documentation and experiment records.

## Testing Guidelines

No testing framework or coverage threshold is established. For new automated tests, use a `tests/` directory and `test_*.py` filenames; document the chosen runner and command when introducing it. Validate training and retrieval changes with checkpoint-specific evaluations. Record exact commands, hyperparameters, results, and failures in one Markdown lab notebook per checkpoint.

## Commit & Pull Request Guidelines

History currently contains `Initial commit` and `docs: add MIT license`; no comprehensive convention is established. Prefer concise, imperative subjects with a relevant prefix, such as `feat: add ckpt-01 training script`.

PRs should identify the checkpoint, describe behavior changes, list validation commands and outcomes, and link relevant issues. Include screenshots for rendered-plan changes and evaluation results for model or retrieval changes. Commit dependency metadata together and exclude secrets, virtual environments, and local tooling state.
