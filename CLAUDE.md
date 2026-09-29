# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A self-study workspace for a 16-week curriculum on LLM fine-tuning, RAG, and deployment. The curriculum itself is the source of truth for what gets built here: `plans/llm-finetuning-rag-learning-plan.md` (with a rendered twin, `plans/llm-finetuning-rag-learning-plan.html`). It is organized as three tracks (A: fine-tuning, B: RAG, C: deployment) and eight checkpoints, `ckpt-00` through `ckpt-07`. Each checkpoint ends in a shipped artifact, and `ckpt-07` is a capstone that combines the tuned model, the retriever, and the serving stack.

The code is at the very start: `main.py` is the `uv init` stub, `src/` is empty, and `README.md` is blank. Consult the plan to see which checkpoint a task belongs to before you add structure.

## Environment & commands

- Python 3.12 (`.python-version`), managed with **uv** (`pyproject.toml` + `uv.lock`, venv in `.venv/`).
- Dependencies so far: `transformers[torch]`, `peft`.
- `uv sync` installs deps, `uv add <pkg>` adds one, and `uv run python main.py` runs the entrypoint.
- There is no test suite, linter, or build config yet.

## Conventions from the plan

- **Ship rule:** every checkpoint ends with a runnable artifact committed to the repo, such as a training script, a merged model, or an eval report.
- **Lab notebook:** keep one markdown file per checkpoint that records what was run, the hyperparameters, what broke, and what you would change.
- **Pin versions:** the stack moves fast, so lock versions when each project starts. Here `uv.lock` does that job.
- **Hardware target:** a local RTX 5080 (16 GB, Blackwell sm_120) under WSL2. Every CUDA step runs in WSL2 Linux. Use a current PyTorch build with sm_120 kernels, and keep bitsandbytes on the same CUDA major version as torch (a mismatch is the most common breakage). If Triton or `torch.compile` misbehaves, fall back to SDPA attention with compilation off. An unquantized bf16 8–9B baseline won't fit and needs a rented GPU.
- The two plan files are kept in sync by hand. When you edit one, update the other to match.

## Non-source artifacts

`.claude-flow/` and `ruvector.db` are local tooling state from the claude-flow/ruvector plugins, not project code.
