# dehip task runner. Recipes are documented stubs until Phase 0 lands;
# each echoes the command sequence it will eventually run.

hip_repo := "https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing"

default:
    @just --list

# Clone the HIP training/inference/eval code alongside this repo
clone-hip:
    @echo "git clone {{hip_repo}} ../humanization-by-iterative-paraphrasing"

# Fetch a released HIP LoRA adapter (size: 0.6B, 1.7B, 4B, 8B, 14B)
fetch-adapter size="0.6B":
    @echo "hf download YixuanEvenXu/Qwen3-{{size}}-Base-HIP-adapter --local-dir adapters/Qwen3-{{size}}-Base-HIP-adapter"

# Phase 0 smoke test: one prompt through the instruct -> HIP cascade
smoke-test:
    @echo "TODO Phase 0: generate with a Qwen3 instruct model, paraphrase k=2 rounds with the fetched adapter, print before/after"
    @echo "See pipeline/README.md"

# No-op formatter target mirror (real one is in the Makefile for the pre-commit hook)
format:
    @true
