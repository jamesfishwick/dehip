# dehip Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-08-03

## Active Technologies

- Python 3.12, uv-managed (matches the HIP repo's `uv sync` toolchain; forced by the torch/transformers/peft ecosystem) + HIP repo cloned as sibling (`../humanization-by-iterative-paraphrasing`, provides `hip-run`/`hip-score` CLIs); torch + transformers (MPS backend for local inference and embedding); `openai` client (GPT-5.4-mini judge, prompt reverse-generation); `datasets` (FineWeb streaming sample); numpy/scipy (MMD, token L2); pytest (001-hip-cascade-harness)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.12, uv-managed (matches the HIP repo's `uv sync` toolchain; forced by the torch/transformers/peft ecosystem): Follow standard conventions

## Recent Changes

- 001-hip-cascade-harness: Added Python 3.12, uv-managed (matches the HIP repo's `uv sync` toolchain; forced by the torch/transformers/peft ecosystem) + HIP repo cloned as sibling (`../humanization-by-iterative-paraphrasing`, provides `hip-run`/`hip-score` CLIs); torch + transformers (MPS backend for local inference and embedding); `openai` client (GPT-5.4-mini judge, prompt reverse-generation); `datasets` (FineWeb streaming sample); numpy/scipy (MMD, token L2); pytest

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
