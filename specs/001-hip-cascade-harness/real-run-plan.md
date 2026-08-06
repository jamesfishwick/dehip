# Real-Instrument Smoke Run Plan (issue #16)

The only ticket that spends real compute and API money. Run the whole pipeline on real instruments at smoke tier (50 pairs) and record it against SC-003 through SC-006. This plan spends nothing on its own. Every paid step is authorized by hand.

## Prerequisites

- `OPENAI_API_KEY`: the judge (gpt-5.4-mini) and prompt reverse-generation in build-corpus.
- `PANGRAM_API_KEY` (or `GPTZERO_API_KEY`): the external detector for SC-005.
- HIP sibling checkout: `just clone-hip` clones YixuanEvenXu/humanization-by-iterative-paraphrasing next door and uv-syncs it.
- HIP adapters: `just fetch-adapter 0.6B` (smoke) and `just fetch-adapter 4B` (the full run). Pass the size positionally, not as `size=...`.
- Hardware: a 36GB Apple Silicon Mac (MPS). The first run downloads roughly 30GB of weights: Qwen3-4B-Instruct-2507 (~8GB), llama-embed-nemotron-8b (~16GB), and the HIP base plus adapter.

The keys never enter this repo or the transcript. Export them in the shell before the run.

## Estimated real cost

| Item | Calls | Approx USD |
|---|---|---|
| build-corpus prompt reverse-gen | 50 | ~$1 |
| self-check full (optional, with JMQ) | ~150 | ~$0.10 |
| score twice (JMQ, 50 pairs x 6 dims x 2) | 600 | ~$0.30 |
| detect (SC-005, 50 draft + 50 rewrite) | 100 | ~$1 (Pangram) |
| Total API | | ~$2 to $3 |

Compute is local (electricity only). Wall-clock is roughly half a day.

## Sequence

1. Setup: `uv sync`, `just clone-hip`, `just fetch-adapter 0.6B` then `just fetch-adapter 4B`.
2. `dehip --seed 42 build-corpus --tier smoke --corpus fineweb` (50 pairs).
3. Self-check first, the trust gate: `dehip self-check --skip-jmq`. MMD near zero, token-L2 near zero on a half-vs-half split. If this fails, stop. It is a harness bug, not a result.
4. `dehip --seed 42 generate` (50 drafts).
5. `dehip rewrite --rounds 2`. Start with the 0.6B adapter (cheapest, fastest, and it validates the hip-run boundary that the tests only mocked), then the 4B. Watch stderr for degeneration flags.
6. `dehip score` twice (draft vs reference, rewrite vs reference) with `--yes`.
7. `dehip report --benchmark` for the SC-003 table.
8. `dehip detect` for the SC-005 delta.
9. Spot-read at least 5 bundles for SC-004.
10. Record wall-clock per step for SC-006 and update the quickstart table.

The full command lines with real paths live in the `just smoke-test` recipe and in quickstart.md.

## Success criteria (record, do not assume)

- SC-003: the report table shows at least two of three metrics closer to human (rewrite vs draft).
- SC-004: degeneration flags at or under 5 percent, and five spot-reads confirm the rewrite still answers the prompt.
- SC-005: the detector rates the rewrites at least 30 points more human.
- SC-006: the full run finishes in under one day without babysitting.

The definition of done is to record the outcomes, not to guarantee they pass. The open research question, whether HIP rewrites are better prose or only detector-evasion, means SC-003 and SC-005 can legitimately come back negative. That is a finding, not a failure.

## Risks

- Memory: the embedder (16GB) plus a 4B model (8GB) on a 36GB machine. Metal weights are unpageable, so generation and embedding run sequentially. The harness already does this, and the embedding cache makes each text a one-time cost.
- HIP integration boundary: the cascade shells out to a real `uv run hip-run`. The tests mocked it, so the first real invocation is where breakage is most likely. The 0.6B smoke pass de-risks this cheaply before the 4B run.
- Model download: about 30GB on the first run.
- Mid-run API failure: the harness is resumable per pair (generate, rewrite, detect), so a failure does not lose prior spend.

## Recommendation

Run the 0.6B adapter smoke pass first. It exercises the full pipeline and the real hip-run boundary for near-zero cost. Commit to the 4B run, the real few dollars and the hours, only once self-check passes and the 0.6B cascade produces sane bundles.
