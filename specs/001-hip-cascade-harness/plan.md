# Implementation Plan: HIP Cascade and Evaluation Harness

**Branch**: `001-hip-cascade-harness` | **Date**: 2026-08-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-hip-cascade-harness/spec.md`

## Summary

Build two coupled deliverables: (1) a rewrite cascade that generates a draft with a Qwen3 instruct model and rewrites it toward the human distribution with the released HIP base-paraphraser LoRA adapters, and (2) an evaluation harness computing the Rosmine protocol's three metrics (MMD over embeddings, JMQ pairwise judge win-rate, token-frequency L2) so any text set can be scored against human references and placed next to the published DFT benchmark numbers. Technical approach: a single uv-managed Python package (`dehip`) that wraps the HIP repo's released `hip-run` CLI for the rewrite stage (never reimplementing it), talks to exact-protocol instruments (GPT-5.4-mini judge, llama-embed-nemotron-8b embedder) pinned by the clarification session, and moves all data between stages as JSONL files so every stage is independently runnable, resumable, and auditable.

## Technical Context

**Language/Version**: Python 3.12, uv-managed (matches the HIP repo's `uv sync` toolchain; forced by the torch/transformers/peft ecosystem)
**Primary Dependencies**: HIP repo cloned as sibling (`../humanization-by-iterative-paraphrasing`, provides `hip-run`/`hip-score` CLIs); torch + transformers (MPS backend for local inference and embedding); `openai` client (GPT-5.4-mini judge, prompt reverse-generation); `datasets` (FineWeb streaming sample); numpy/scipy (MMD, token L2); pytest
**Storage**: Files only. JSONL for text sets, bundles, and verdicts; JSON for metric reports; content-hash-keyed cache for embeddings. All under `data/` and `results/` (gitignored)
**Testing**: pytest; synthetic-distribution unit tests for metrics (known-answer Gaussian toys for MMD); the spec's FR-003 self-check doubles as the integration test
**Target Platform**: macOS (36GB Apple Silicon) for small-N runs; rented Linux GPU for 2000-sample benchmark runs. Same code path, different config
**Project Type**: Single project (CLI package)
**Performance Goals**: SC-006 — 50-prompt end-to-end run (generate, rewrite k≤4, score, report) in under one day unattended on local hardware
**Constraints**: Embedder is an 8B model — embeddings must be cached (one forward pass per unique text, ever); judge calls cost real money — validation (FR-009) must run before any API spend; all randomness seeded and recorded
**Scale/Scope**: Corpus tiers: 50 (smoke), 400 (judged benchmark), 2000 (full distribution metrics). Seven CLI subcommands, ~8 core modules, no service/daemon component

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is the unfilled template — no project-specific gates are defined. Proceeding under general simplicity defaults, all satisfied:

- Single project, no speculative abstraction layers: PASS (one package, file-based I/O, no service)
- Reuse over reimplementation: PASS (HIP inference via its released CLI; metrics are the only new algorithmic code, and they are small pure functions)
- Test-first where it matters: PASS (metric functions get known-answer tests before wiring; self-check mode is a spec requirement, FR-003)
- Post-Phase-1 re-check (2026-08-03): design added no projects, no new services, no violations. PASS

## Project Structure

### Documentation (this feature)

```text
specs/001-hip-cascade-harness/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI + file-format contracts)
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/dehip/
├── __init__.py
├── cli.py               # entry point: dehip {build-corpus,generate,rewrite,score,self-check,detect,report}
├── corpus.py            # FineWeb sampling, quality screen, prompt reverse-generation, side-corpus ingest (FR-010)
├── generate.py          # instruct-model draft generation (transformers, MPS/CUDA)
├── cascade.py           # hip-run wrapper: config emission, subprocess, per-round capture, degeneration checks (FR-005, FR-006)
├── metrics/
│   ├── mmd.py           # unbiased MMD^2, Gaussian RBF, median-heuristic bandwidth
│   ├── token_l2.py      # 1-gram frequency L2
│   ├── jmq.py           # judge orchestration, A/B randomization, verdict recording (FR-002)
│   └── embeddings.py    # llama-embed-nemotron-8b runner + content-hash cache
├── detector.py          # external detector scoring via Pangram SDK (SC-005 instrument)
├── report.py            # metric report + benchmark comparison table (FR-004, FR-007)
└── validate.py          # input-set validation gates (FR-009), degeneration detectors

tests/
├── unit/                # metric known-answer tests, validators, randomization audit
└── integration/         # self-check (FR-003), tiny end-to-end on fixture texts

judge-prompts/           # existing; loaded verbatim by jmq.py (FR-002)
data/                    # corpora, text sets, embedding cache (gitignored)
results/                 # bundles, verdicts, reports (gitignored)
```

**Structure Decision**: Single uv project with a `src/dehip` package. The existing `harness/` and `pipeline/` stub directories (doc placeholders from repo creation) are retired; their READMEs' content moves into module docstrings and this plan. `judge-prompts/`, `data/`, `results/` stay as-is. The HIP repo is a sibling checkout, never vendored — its CLI surface is the integration boundary.

## Complexity Tracking

No constitution violations to justify. One deliberate dependency worth recording: the cascade shells out to `hip-run` in a sibling checkout rather than importing HIP as a library. Rationale in research.md R2 (pinned integration surface, zero fork risk); the cost is a subprocess boundary and YAML config emission, both trivial.
