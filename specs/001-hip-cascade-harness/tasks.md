# Tasks: HIP Cascade and Evaluation Harness

**Input**: Design documents from `/specs/001-hip-cascade-harness/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Included where the design demands them: known-answer tests for metric code (research R5) and the FR-003 self-check are spec/plan requirements, not optional extras. No blanket TDD elsewhere.

**Organization**: Grouped by user story. US1 (harness) and US2 (cascade) are independent after the Foundational phase and can proceed in parallel; US3 composes both.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Parallelizable (different files, no dependency on an incomplete task)
- **[Story]**: US1 = score text sets, US2 = rewrite cascade, US3 = benchmark comparison

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Turn the doc-only repo into a runnable uv Python project matching plan.md's structure

- [ ] T001 Initialize uv project: `pyproject.toml` (Python 3.12, deps per plan.md Technical Context: torch, transformers, datasets, openai, numpy, scipy, pytest, ruff), `src/dehip/__init__.py`, `dehip` console entry point, `uv sync` passes
- [ ] T002 [P] Retire stub dirs: delete `harness/README.md` and `pipeline/README.md` (content already captured in plan.md), keep `judge-prompts/`, `data/`, `results/`; extend `.gitignore` with `data/emb-cache/` and `results/runs/`
- [ ] T003 [P] Configure pytest (`tests/unit/`, `tests/integration/` skeletons with one placeholder test each) and ruff in `pyproject.toml`

**Checkpoint**: `uv run pytest` and `uv run dehip --help` both exit 0

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Schemas, validation, CLI skeleton, and the corpus builder — every user story consumes these

- [ ] T004 Implement entity schemas in `src/dehip/schemas.py`: Pair, TextSet manifest, RewriteBundle, JudgeVerdict, MetricReport dataclasses with `schema_version: 1`, JSONL/JSON read/write helpers, per data-model.md field tables
- [ ] T005 Implement `src/dehip/validate.py`: input-set gates (pair_id pairing, counts, min length, min N floor with override) per FR-009, and degeneration detectors (empty, length ratio [0.5, 2.0], 3-consecutive-same-start-word flag, >5% non-ASCII burst) with thresholds and flag-vs-hard-trip semantics per research R10; unit tests in `tests/unit/test_validate.py`
- [ ] T006 Implement `src/dehip/cli.py` skeleton: subcommand dispatch for `build-corpus`, `generate`, `rewrite`, `score`, `self-check`, `report`; global `--seed` plumbing recorded into all outputs; exit codes 0/2/3/4 per contracts/cli.md; JSON summary to stdout, progress to stderr
- [ ] T007 Implement FineWeb corpus builder in `src/dehip/corpus.py`: stream `HuggingFaceFW/fineweb` sample-10BT, filter English blog/news register 150-1200 words, quality screen, seeded sample to tier size (50/400/2000), reverse-generate prompts via `gpt-5.4-mini` with prompt-variant rotation and resumability, cost-gate preflight with `--yes`, emit Pair JSONL + human_reference TextSet manifest; wire `dehip build-corpus` (FR-010, research R8)
- [ ] T008 [P] Add personal side-corpus ingest to `src/dehip/corpus.py`: read James's published posts from local paths/URLs, reverse-generate a prompt per post via the same R8 path (`prompt_generator` recorded) so pairs are schema-complete, tag `corpus: "personal"` (FR-010)

**Checkpoint**: `dehip build-corpus --tier smoke --corpus fineweb --seed 42` produces a valid 50-pair corpus; validators reject a hand-broken manifest

---

## Phase 3: User Story 1 — Score any text set against a human reference (P1) 🎯 MVP

**Goal**: One invocation computes MMD + JMQ + token L2 for any candidate set vs reference set, with a self-check mode that proves the metrics before they are trusted

**Independent Test**: `dehip self-check` on the smoke corpus passes (half-vs-half: MMD ~0, token L2 ~0, JMQ win-rate 45-55%); scoring plain instruct output vs references produces a complete report with all six JMQ dimensions

- [ ] T009 [P] [US1] Implement token L2 in `src/dehip/metrics/token_l2.py`: Qwen3 tokenizer, 1-gram frequency vectors over union vocabulary, L2 distance, tokenizer id surfaced for report config (research R7); known-answer unit tests in `tests/unit/test_token_l2.py` (identical sets → 0, disjoint vocab → computed constant)
- [ ] T010 [P] [US1] Implement MMD in `src/dehip/metrics/mmd.py`: unbiased MMD^2, Gaussian RBF, median-heuristic bandwidth on pooled sample, bandwidth returned for report config (research R5); known-answer unit tests in `tests/unit/test_mmd.py` (same-distribution Gaussians → ~0, shifted Gaussians → positive, matches closed-form on a tiny fixed case)
- [ ] T011 [P] [US1] Implement embedder + cache in `src/dehip/metrics/embeddings.py`: `nvidia/llama-embed-nemotron-8b` via transformers (MPS/CUDA autodetect, fp16), batched, append-only cache keyed (embedder_id, content_sha256) as parquet index + npy store under `data/emb-cache/` (research R4); unit test with a stub embedder in `tests/unit/test_embeddings.py` proving cache hits skip recompute
- [ ] T012 [P] [US1] Implement JMQ in `src/dehip/metrics/jmq.py`: load six verbatim templates from `judge-prompts/`, seeded per-pair A/B randomization, one `gpt-5.4-mini` call per (pair, dimension), verdict JSONL persisted before aggregation, one retry then excluded-and-counted, concurrency limit, cost preflight (pairs x 6) with `--yes` gate (FR-002, research R6); unit tests in `tests/unit/test_jmq.py` with a mocked client (order ~50/50 over 200 seeded pairs, invalid-verdict exclusion counted)
- [ ] T013 [US1] Implement report assembly in `src/dehip/report.py`: MetricReport JSON + rendered markdown, config block (seed, judge, embedder, tokenizer, bandwidth), auto-caveats (small-N, bandwidth comparability, non-default judge) per data-model.md
- [ ] T014 [US1] Wire `dehip score` in `src/dehip/cli.py`: validation gates before any spend, metric selection flag, `--recompute-jmq-from` path re-aggregating persisted verdicts without API calls (FR-008)
- [ ] T015 [US1] Wire `dehip self-check` in `src/dehip/cli.py`: seeded half-vs-half split of a human set, assert MMD/token-L2 noise bounds and JMQ in [0.9, 1.1], `--skip-jmq` mode, exit 4 out of bounds (FR-003); derive the MMD/token-L2 noise bounds empirically from at least 5 differently-seeded half-splits of the smoke corpus and persist them as documented constants (value + derivation provenance) in `src/dehip/metrics/bounds.py`; integration test in `tests/integration/test_self_check.py` using fixture texts and stub embedder/judge
- [ ] T016 [US1] Positional-bias audit: `dehip score` report includes order-distribution stats from verdicts; integration test in `tests/integration/test_bias_audit.py` asserting recorded assignments reproduce the seeded sequence (Story 1 scenario 4)

**Checkpoint**: Full US1 independent test passes on the smoke corpus with stub instruments; then once with real embedder locally

---

## Phase 4: User Story 2 — Rewrite a draft through the cascade (P2)

**Goal**: Prompt (or existing draft) in → instruct draft + k HIP rewrite rounds out, every intermediate captured, degenerate rounds stopped and flagged

**Independent Test**: One smoke prompt through `generate` + `rewrite --rounds 2` yields a bundle with draft, both rounds, and metadata; a deliberately degenerate fixture triggers stop-and-flag, not silent output; an external AI-text detector rates the k=2 rewrite more human than the draft (SC-005 machinery, spec US2)

- [ ] T017 [P] [US2] Implement draft generation in `src/dehip/generate.py`: `Qwen/Qwen3-4B-Instruct-2507` default via transformers (verify exact HF repo id at implementation; MPS/CUDA), T=0.7 / top-p 0.95 defaults, seeded, per-pair resumability, draft TextSet + nascent RewriteBundle records (research R3); wire `dehip generate`
- [ ] T018 [US2] Implement hip-run wrapper in `src/dehip/cascade.py`: precondition check (sibling checkout resolves, `uv run hip-run --help` succeeds, exit 3 otherwise), YAML config emission per round, JSONL in/out marshalling, `hip_config` inlined into bundle (research R2)
- [ ] T019 [US2] Implement the round loop in `src/dehip/cascade.py`: k explicit single-round invocations (default 2, max 4), degeneration checks from `validate.py` between rounds, hard-trip stops at last good round with bundle flag, per-round TextSet emission (FR-005, FR-006)
- [ ] T020 [US2] Wire `dehip rewrite` in `src/dehip/cli.py`: `--run` continuation mode and `--draft-file` rewrite-only mode producing the same bundle structure (Story 2 scenario 2)
- [ ] T021 [US2] Integration test in `tests/integration/test_cascade.py`: mocked `hip-run` subprocess (fixture stdout), asserts per-round capture, degeneration stop-and-flag on a degenerate fixture round, and bundle schema validity — no model downloads in CI
- [ ] T022 [US2] Implement external detector scoring in `src/dehip/detector.py`: score draft and rewrite TextSets for human-probability via the Pangram SDK (and optionally GPTZero), reusing the HIP repo's `hip-score` config approach; per-set summary + per-text scores persisted under `results/reports/`; wire `dehip detect` per contracts/cli.md; this is SC-005's measurement instrument (≥30-point gain check); unit test with mocked client in `tests/unit/test_detector.py`

**Checkpoint**: US2 independent test passes with mocked hip-run; then one real smoke prompt through the 0.6B adapter locally, with `dehip detect` confirming the rewrite scores more human than the draft

---

## Phase 5: User Story 3 — Compare cascade output to the published benchmark (P3)

**Goal**: Draft scores, rewrite scores, per-metric deltas, and the pinned DFT/superbaseline rows in one artifact, with k-trajectory when multiple round counts are scored

**Independent Test**: With US1+US2 artifacts for 50 smoke prompts, `dehip report --benchmark` emits a comparison containing deltas and benchmark rows; feeding it a personal-corpus report makes it refuse `--benchmark`

- [ ] T023 [P] [US3] Pin the benchmark table in `src/dehip/benchmark.py`: 4B/8B/14B SFT-superbaseline and DFT rows (MMD/JMQ/token-L2) as a code constant citing PLAN.md as source of truth, `external_protocol: true` flag driving the different-judge/corpus caveat in rendering
- [ ] T024 [US3] Implement comparison assembly in `src/dehip/report.py`: per-metric deltas between draft and rewrite reports, k-trajectory table across multiple rewrite reports, hard refusal of `--benchmark` when any input scored a `personal` corpus (FR-007, FR-010)
- [ ] T025 [US3] Wire `dehip report` in `src/dehip/cli.py` per contracts/cli.md; integration test in `tests/integration/test_report.py` on fixture reports (deltas correct, personal-corpus refusal, trajectory ordering)

**Checkpoint**: US3 independent test passes on fixtures; full smoke-tier pipeline (quickstart steps 2-5) runs end to end locally

---

## Phase 6: Polish & Cross-Cutting

- [ ] T026 Run the full quickstart on real instruments (smoke tier, real embedder + judge, 0.6B then 4B adapter), record actual wall-clock against the quickstart table and SC-006, run `dehip detect` for the SC-005 gain check, spot-read at least 5 bundles to confirm rewrites still answer their prompts (SC-004), fix what breaks
- [ ] T027 [P] Replace justfile stubs with real recipes (`clone-hip`, `fetch-adapter`, `smoke-test` invoking the quickstart sequence)
- [ ] T028 [P] Update root README.md status section and PLAN.md Phase 0/1 status; module docstrings absorb any remaining stub-README content; final `uv run ruff check` clean

---

## Dependencies

```text
Phase 1 (Setup) → Phase 2 (Foundational) → ┬→ Phase 3 (US1, MVP)  ─┬→ Phase 5 (US3) → Phase 6
                                            └→ Phase 4 (US2)       ─┘
```

- US1 and US2 are independent of each other; both need only Phases 1-2. They can be built in parallel.
- US3 needs both US1 (reports) and US2 (rewrite sets).
- Within Phase 2: T007 and T008 are independent after T004-T006.

## Parallel Execution Examples

- **Phase 2**: after T004-T006 land, T007 and T008 in parallel
- **US1**: T009, T010, T011, T012 all touch different files — four parallel tracks, then T013-T016 serially
- **Cross-story**: one track runs Phase 3 (US1) while another runs Phase 4 (US2)
- **US3**: T023 parallel with the tail of either story

## Implementation Strategy

**MVP = Phases 1-3** (Setup + Foundational + US1). That delivers the reusable asset — a working, self-checked harness that can score plain instruct output against human references today, before any cascade exists. Ship increments in phase order after that: US2 makes the cascade real, US3 answers the benchmark question, Polish proves SC-006 on real hardware.

Real-model work (embedder download, adapter inference, judge and detector spend) is confined to checkpoints and T026; every other task runs green on stubs and fixtures, so CI never needs a GPU or an API key.
