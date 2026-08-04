# CLI Contract: dehip

Single entry point `dehip`, subcommand per pipeline stage. Every stage reads/writes the file formats defined in data-model.md, exits non-zero on validation failure, and prints a JSON result summary to stdout (human-readable progress to stderr). All commands accept `--seed` and record it in outputs.

## dehip build-corpus

Build the primary or personal corpus (FR-010, R8).

```text
dehip build-corpus --tier {smoke|judged|full}     # 50 / 400 / 2000 pairs
                   --corpus {fineweb|personal}
                   [--source PATH_OR_URLS]        # required for personal
                   [--out data/corpus/{name}.jsonl]
```

- Output: Pair JSONL + TextSet manifest (role=human_reference).
- Failure modes: doc shortage after filtering (reports shortfall, non-zero exit); prompt-generation API failure (resumable — already-generated pairs kept).
- Cost gate: prints estimated prompt-generation call count; `--yes` required above spend threshold (FR-009).

## dehip generate

Stage-1 instruct drafts for a corpus (R3).

```text
dehip generate --corpus data/corpus/{name}.jsonl
               [--model Qwen/Qwen3-4B-Instruct-2507] [--temperature 0.7] [--top-p 0.95]
               [--out results/runs/{run_id}/]
```

- Output: draft TextSet + per-pair draft records inside nascent RewriteBundles.
- Resumable per pair.

## dehip rewrite

Cascade stage: k rounds of hip-run with inter-round degeneration checks (FR-005, FR-006, R2).

```text
dehip rewrite --run results/runs/{run_id}/        # continues from generate
              [--draft-file PATH]                 # rewrite-only mode, skips generate
              [--adapter YixuanEvenXu/Qwen3-4B-Base-HIP-adapter]
              [--rounds 2]                        # max 4
              [--hip-repo ../humanization-by-iterative-paraphrasing]
```

- Output: completed RewriteBundles + one rewrite TextSet per round.
- Precondition: HIP sibling checkout exists and `uv run hip-run` works there (checked before any inference).
- Degeneration: hard trips stop that pair at last good round, bundle flagged, run continues.

## dehip score

The harness (FR-001..004, FR-008, FR-009).

```text
dehip score --candidate {set manifest} --reference {set manifest}
            [--metrics mmd,token_l2,jmq]          # default: all three
            [--judge gpt-5.4-mini] [--embedder nvidia/llama-embed-nemotron-8b]
            [--out results/reports/{report_id}.json]
```

- Validation before spend: pairing, counts, min lengths, min N; JMQ cost preflight with `--yes` gate.
- Output: MetricReport JSON + rendered .md; verdicts JSONL persisted before aggregation.
- `--recompute-jmq-from results/runs/{run_id}/verdicts.jsonl` re-aggregates without API calls (FR-008).

## dehip self-check

FR-003 as a command.

```text
dehip self-check --reference {human set manifest} [--skip-jmq]
```

- Splits the human set in half, scores half vs half, asserts noise bounds (MMD ~0, token L2 ~0, JMQ in [0.9, 1.1] i.e. win-rate 45-55%). Non-zero exit outside bounds.

## dehip detect

External AI-text detector scoring — SC-005's measurement instrument.

```text
dehip detect --sets {draft manifest} {rewrite manifest} [...]
             [--detector pangram]                 # pangram default; gptzero optional
             [--out results/reports/{name}-detect.json]
```

- Requires `PANGRAM_API_KEY` (or `GPTZERO_API_KEY`); exit 3 if missing.
- Output: per-set human-probability summary (mean, median, distribution) + per-text scores; the SC-005 check is the delta between a draft set and a rewrite set summary.
- Cost gate: prints text count per set; `--yes` required above spend threshold (FR-009).

## dehip report

Comparison assembly (FR-007, Story 3).

```text
dehip report --draft-report R1.json --rewrite-report R2.json [...more rewrite reports]
             [--benchmark]                        # attach pinned benchmark rows
             [--out results/reports/{name}-comparison.{json,md}]
```

- Deltas per metric; k-trajectory table when multiple rewrite reports given; refuses `--benchmark` if any input report scored a `personal`-corpus set (FR-010).

## Exit codes (all commands)

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | validation failure (bad inputs, mismatched sets, N below floor without override) |
| 3 | external dependency failure (HIP checkout, API auth) |
| 4 | self-check out of bounds |
