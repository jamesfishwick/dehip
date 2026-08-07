# Smoke Run Findings (0.6B cascade, issue #16)

First end-to-end run of the whole pipeline on real instruments: a 50-pair FineWeb
smoke corpus, Qwen3-4B-Instruct drafts, the released Qwen3-0.6B HIP paraphraser for
the rewrite, the real nvidia/llama-embed-nemotron-8b embedder for MMD, GPT-5.4-mini
for JMQ, and Pangram for the external detector. Date: 2026-08-07.

The 0.6B is a deliberately weak smoke paraphraser. It was chosen to validate the
pipeline and the hip-run boundary cheaply, not to produce quality. The quality
verdict waits for the 4B run.

## The pipeline ran end to end

```
build-corpus (50) -> self-check PASS -> generate (50 drafts)
  -> rewrite (50, real hip-run 0.6B) -> score (3 metrics) -> report -> detect
```

Every stage ran against real models and real APIs. The self-check gate passed
first (human-vs-human MMD 0.0003), so the metrics are trustworthy.

## Scorecard

| Criterion | Result | Detail |
|---|---|---|
| SC-001 self-check | PASS | human-vs-human MMD 0.000307, token-L2 0.0221 |
| SC-003 rewrite closer to human on 2 of 3 | PASS | MMD and token-L2 yes, JMQ no |
| SC-004 degeneration at or under 5% | FAIL | 14 of 50 (28%), the weak 0.6B |
| SC-005 detector at least 30 points more human | FAIL (narrow) | delta +27.3 points, threshold 30 |
| SC-006 full run under one day | PASS | a few hours end to end |

### The numbers (36 common pairs that completed both rounds)

| Metric | Draft | Rewrite (0.6B, k2) | Direction |
|---|---|---|---|
| MMD (lower is human) | 0.0132 | 0.0014 | 9x closer to human |
| token-L2 (lower is human) | 0.0358 | 0.0276 | closer to human |
| JMQ (higher is human) | 0.611 | 0.222 | worse on every dimension |
| Pangram mean human-prob | 0.000 | 0.273 | +27.3 points |

JMQ dimensions, draft -> rewrite: clarity 0.72 -> 0.39, coherence 0.56 -> 0.22,
creativity 0.56 -> 0.22, depth 0.28 -> 0.11, relevance 0.44 -> 0.33.

Pangram distribution: all 36 drafts scored 0.0 (fully AI). The rewrites were
bimodal: 9 of 36 flipped to 1.0 (fully human), 26 stayed at 0.0.

## The finding

Every "looks human" metric moved the right way and the one "is it good writing"
metric moved the wrong way. MMD dropped 9x, token-L2 dropped, Pangram rose 27
points with a quarter of texts fully fooled. JMQ fell on every dimension. The 0.6B
HIP rewrite is a humanizer and a detector-evader, and it works even at 0.6B, but it
is not a prose improver. A detector-only or MMD-only evaluation would call this a
win. The two-metric-family harness catches that the rewrites are more human and
worse writing at the same time.

The 0.6B rewrite's MMD (0.0014) is lower than the published 14B DFT benchmark
(0.018). A tiny degeneration-prone paraphraser beats a 14B production model on the
distribution metric, by making text blander and more repetitive, which is closer to
the human distribution in embedding space. Its JMQ (0.22) is far below any published
row. This is the clearest possible argument for reading the metrics together, never
one alone.

## The open question, restated

HIP's published wins are detector-evasion numbers. The unanswered question is whether
the rewrites are better prose. For the 0.6B the answer is no. The 4B run tests
whether a stronger paraphraser can pull the distribution toward human and clear the
detector without the JMQ collapse. Everything needed to run it now exists and is
proven.

## Integration bugs the real run surfaced

Every stage was unit-tested against mocked seams, so each real integration boundary
hid a mismatch that only a real run could expose. Eight in total:

1. OOM: `list()` over the unbounded FineWeb stream. Bounded with `islice`.
2. Protocol-isinstance class-factory: `isinstance(cls, runtime_checkable_Protocol)`
   is true for the class itself, so the CLI's client class was never constructed.
3. The pre-commit hook never gated ruff (no lint target). Added `just lint`.
4. The corpus manifest carried no `{pair_id, text}` texts file, so the score reader
   KeyErrored on `text` (the Pair stores it as `reference_text`).
5. The embedder needs `trust_remote_code=True` (custom llama_bidirec architecture).
6. The hip-run seam was built against an assumed interface. The real hip-run takes
   `--config` only, reads `adapter_path` and `input_jsonl` and `output_parquet` as
   config keys, writes a parquet, infers the base model from the adapter, and runs
   on CPU (no MPS). A full seam rework, plus a round-filter bug (hip-run stamps
   round 1 on every single-round invocation, so filtering on the cascade round
   skipped every row on round 2).
7. The detector assumed a `pangram` package with `predict()` returning
   `ai_likelihood`. The real SDK is `pangram-sdk`, `predict()` returns
   `fraction_human`, and `model="default"` is required after 2026-09-30.

The lesson: a mock of a seam encodes your assumptions about an interface, never the
interface itself. The three external seams (hip-run, the embedder, the detector)
each need one real-integration smoke test, because unit mocks structurally cannot
catch "the interface I imagined is not the interface that exists."

## Known follow-ups

- Test isolation: the score and self-check CLI tests default to the real
  `data/emb-cache/`, so a real run's dim-4096 vectors collide with the stub tests'
  dim-4 vectors. Tests should use a tmp cache dir.
- SC-004/SC-005 pass and the real quality verdict need the 4B (or larger) adapter.
- `bounds.py` `REAL_INSTRUMENT_BOUNDS` can be filled from this run's self-check
  (MMD ~0.0003, token-L2 ~0.022).
