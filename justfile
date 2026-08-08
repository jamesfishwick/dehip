# dehip task runner. Recipes drive the real dehip CLI (src/dehip/cli.py) and
# the HIP sibling checkout. See specs/001-hip-cascade-harness/quickstart.md for
# the narrated walkthrough and the prerequisites each recipe needs (uv, an
# OPENAI_API_KEY for the judge, the HIP checkout, and downloaded models).

hip_repo := "https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing"
hip_dir := "../humanization-by-iterative-paraphrasing"

# Fixed smoke-run layout so smoke-test's steps chain deterministically. A real
# `dehip generate`/`rewrite` run mints a timestamp run_id when --out is omitted;
# pinning --out here lets each later step name the manifest the prior step wrote.
smoke_corpus := "data/corpus/fineweb-smoke.jsonl"
smoke_manifest := "data/corpus/fineweb-smoke.manifest.json"
smoke_run := "results/runs/smoke"

default:
    @just --list

# Lint the package with ruff. The global pre-commit hook detects this recipe
# (`just lint`) and blocks a commit when it fails, so ruff is gated at commit
# time, not just in CI.
lint:
    uv run ruff check .

# The cascade's `dehip rewrite` shells out to `uv run hip-run` in this checkout.
# Clone the HIP sibling checkout next to this repo and install its deps.
clone-hip:
    git clone {{hip_repo}} {{hip_dir}}
    cd {{hip_dir}} && uv sync

# The adapter id matches `dehip rewrite --adapter`; 4B is the harness default.
# Pass the size positionally: `just fetch-adapter 0.6B` or `just fetch-adapter 4B`
# (NOT `size=0.6B` -- just reads that as the literal argument value).
# Fetch a released HIP LoRA adapter by size (0.6B, 1.7B, 4B, 8B, 14B).
fetch-adapter size="4B":
    uv run hf download YixuanEvenXu/Qwen3-{{size}}-Base-HIP-adapter \
        --local-dir adapters/Qwen3-{{size}}-Base-HIP-adapter

# Prove the harness against itself, generate drafts, rewrite k=2, score, compare.
# Runs the quickstart end to end. Needs the prerequisites above (keys, HIP
# checkout, models); the recipe is the real command sequence, not a claim it has
# been run on real models.
# Run the quickstart smoke sequence end to end.
smoke-test:
    uv sync
    uv run dehip --seed 42 build-corpus --tier smoke --corpus fineweb \
        --out {{smoke_corpus}}
    uv run dehip self-check --reference {{smoke_manifest}} --skip-jmq
    uv run dehip --seed 42 generate --corpus {{smoke_corpus}} --out {{smoke_run}}
    uv run dehip rewrite --run {{smoke_run}} --rounds 2 --hip-repo {{hip_dir}}
    uv run dehip score --candidate {{smoke_run}}/draft.manifest.json \
        --reference {{smoke_manifest}} --prompts {{smoke_corpus}} --yes \
        --out results/reports/smoke-draft.json
    uv run dehip score --candidate {{smoke_run}}/rewrite-k2.manifest.json \
        --reference {{smoke_manifest}} --prompts {{smoke_corpus}} --yes \
        --out results/reports/smoke-rewrite-k2.json
    uv run dehip report --draft-report results/reports/smoke-draft.json \
        --rewrite-report results/reports/smoke-rewrite-k2.json --benchmark \
        --out results/reports/smoke-comparison.json

# Render the smoke-run findings to a branded PDF (Palatino body, Menlo code).
# Needs pandoc + a xelatex-capable TeX (macOS: MacTeX/BasicTeX). The `#` in
# "issue #16" is a LaTeX macro char, so the subtitle spells it out.
render-findings:
    pandoc docs/smoke-run-findings.md \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V mainfont="Palatino" -V monofont="Menlo" -V fontsize=11pt \
        -V colorlinks=true -V linkcolor=RoyalBlue -V urlcolor=RoyalBlue \
        -V title="dehip: Smoke Run Findings" \
        -V subtitle="HIP cascade harness (0.6B and 4B), issue 16" \
        -V author="James Fishwick" -V date="2026-08-07" \
        -o docs/smoke-run-findings.pdf

# No-op formatter mirror; real one is in the Makefile for the pre-commit hook.
format:
    @true
