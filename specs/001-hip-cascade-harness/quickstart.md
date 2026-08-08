# Quickstart: HIP Cascade and Evaluation Harness

The fastest path from clean checkout to a scored smoke run. Assumes macOS with uv installed and an `OPENAI_API_KEY` for the judge.

## 1. Setup

```bash
cd ~/Workspace/personal/dehip
uv sync                                          # installs dehip + deps

# HIP sibling checkout (the rewrite engine)
git clone https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing \
    ../humanization-by-iterative-paraphrasing
(cd ../humanization-by-iterative-paraphrasing && uv sync)
```

## 2. Smoke corpus (50 pairs, ~$1 of prompt generation)

```bash
uv run dehip --seed 42 build-corpus --tier smoke --corpus fineweb
```

## 3. Prove the harness before trusting it

```bash
uv run dehip self-check --reference data/corpus/fineweb-smoke.manifest.json --skip-jmq
```

Must pass (MMD ~0, token L2 ~0 on a half-vs-half split) before anything downstream means anything. Run once more without `--skip-jmq` when you are ready to spend judge calls on the full self-check.

## 4. Generate drafts, rewrite through HIP

```bash
uv run dehip --seed 42 generate --corpus data/corpus/fineweb-smoke.jsonl
uv run dehip rewrite  --run results/runs/<run_id>/ --rounds 2
```

First rewrite run downloads the 4B base model + adapter (~8GB). Watch stderr for degeneration flags.

## 5. Score and compare

```bash
uv run dehip score --candidate results/runs/<run_id>/draft.manifest.json \
                   --reference data/corpus/fineweb-smoke.manifest.json \
                   --prompts data/corpus/fineweb-smoke.jsonl --yes
uv run dehip score --candidate results/runs/<run_id>/rewrite-k2.manifest.json \
                   --reference data/corpus/fineweb-smoke.manifest.json \
                   --prompts data/corpus/fineweb-smoke.jsonl --yes
uv run dehip report --draft-report results/reports/<draft>.json \
                    --rewrite-report results/reports/<rewrite>.json --benchmark
```

The comparison markdown shows draft vs rewrite deltas on all three metrics next to the pinned DFT benchmark rows. SC-003's question — closer to human on at least two of three — is answered by this table.

## Expected wall-clock (smoke tier, M-series Mac)

| Step | Rough time |
|---|---|
| build-corpus | minutes (API-bound) |
| self-check (no JMQ) | ~10 min (first embedder load dominates) |
| generate (50 drafts, 4B) | ~30-60 min |
| rewrite k=2 (4B + adapter) | ~1-2 h |
| score x2 (embeddings cached after first) | ~20 min + judge API |

Full protocol scale (2000/400) runs the identical commands with `--tier judged` / `--tier full` on rented CUDA hardware.
