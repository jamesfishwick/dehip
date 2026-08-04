# Feature Specification: HIP Cascade and Evaluation Harness

**Feature Branch**: `001-hip-cascade-harness`
**Created**: 2026-08-02
**Status**: Draft
**Input**: User description: "Build the Phase 0 zero-training pipeline (instruct model to HIP base-paraphraser cascade using released LoRA adapters) and the Phase 1 evaluation harness (MMD over embeddings, JMQ pairwise judge win-rate, token-frequency L2) so cascade output can be scored against the Rosmine DFT benchmark numbers"

## Clarifications

### Session 2026-08-03

- Q: Where do prompt + human-reference pairs come from? → A: Two corpora. Primary: fresh pairs built from a FineWeb sample, human doc first, prompt reverse-generated with an LLM (mirrors the published protocol and register). Secondary: James's own published posts as a small side corpus for personal-relevance spot checks, never for benchmark comparison.
- Q: Default judge model for JMQ? → A: GPT-5.4-mini, the exact judge the DFT post used, so the benchmark JMQ column stays meaningful. Other judges remain configurable for later consistency checks.
- Q: MMD embedding model? → A: nvidia/llama-embed-nemotron-8b, the exact protocol embedder, so the benchmark MMD column stays meaningful. Embeddings are cached per text set; small-N runs local, full-scale runs on rented compute.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Score any text set against a human reference (Priority: P1)

A researcher has a set of machine-generated texts and a set of human-written reference texts for the same prompts. They run the evaluation harness and receive a report with three scores: a distribution distance over text embeddings (MMD), a pairwise judge win-rate against the human references (JMQ), and a word-frequency distance (token L2). The report lets them say, with numbers, how far any text set is from human writing.

**Why this priority**: The harness is the reusable asset of the whole project. Every later experiment (cascade tuning, retraining, benchmark comparison) is meaningless without trustworthy measurement. It is also independently valuable: it can score any model's output today, including plain instruct-model baselines, with no rewrite pipeline in existence.

**Independent Test**: Run the harness on a human reference set split against itself. By construction the scores must come back as near-zero distribution distance and a judge win-rate near 50%. Then run it on plain instruct-model output vs human references and confirm scores are meaningfully worse. Delivers value without any cascade code existing.

**Acceptance Scenarios**:

1. **Given** a human reference set split into two halves, **When** the harness scores half A against half B, **Then** MMD is near zero, JMQ is near 1.0 (a ~50% win-rate doubled), and token L2 is near zero, within documented noise bounds.
2. **Given** a set of instruct-model outputs and matched human references, **When** the harness runs, **Then** it produces all three metrics plus the six JMQ quality dimensions (overall, clarity, coherence, creativity, depth, relevance) in a single machine-readable report.
3. **Given** the same inputs run twice with the same seed, **When** scores are compared, **Then** deterministic metrics (MMD, token L2) match exactly and judge-based metrics are reproducible from the recorded verdicts.
4. **Given** judge comparisons, **When** each pair is presented, **Then** A/B order is randomized per pair and the assignment is recorded, so positional bias can be audited.

---

### User Story 2 - Rewrite a draft through the cascade (Priority: P2)

A researcher provides a prompt (or an existing draft). The system generates content with an instruct model, then passes it through the HIP base-model paraphraser for a configurable number of rounds, and returns the original draft, every intermediate round, and the final rewrite side by side.

**Why this priority**: This is the mechanism under test, the open alternative to proprietary DFT. It depends on nothing in the harness (a human can eyeball the before/after, and an external detector can sanity-check it), but its results only become science once User Story 1 exists.

**Independent Test**: Feed one prompt through the cascade with 2 rounds and confirm the output preserves the draft's meaning while reading less machine-like, verified by an external AI-text detector score improving from draft to final rewrite.

**Acceptance Scenarios**:

1. **Given** a prompt, **When** the cascade runs with k rewrite rounds, **Then** the output bundle contains the instruct draft, all k intermediate rewrites, and metadata (model identities, adapter identity, round count, sampling settings).
2. **Given** an existing draft supplied directly (skipping generation), **When** the cascade runs, **Then** only the rewrite stage executes and the same bundle structure is produced.
3. **Given** a rewrite round that degenerates (empty output, extreme length change, or repetition), **When** the cascade detects it, **Then** it stops at the last good round and flags the bundle rather than silently returning degenerate text.

---

### User Story 3 - Compare cascade output to the published benchmark (Priority: P3)

A researcher runs a corpus of prompts through the cascade, scores draft and rewrite sets with the harness, and receives a comparison report: cascade-vs-draft deltas on every metric, and both placed alongside the pinned published benchmark numbers (the DFT results and the SFT superbaselines).

**Why this priority**: This is the payoff question (does the open cascade approach the proprietary DFT numbers) but it is pure composition of Stories 1 and 2 plus a report.

**Independent Test**: With Stories 1 and 2 complete, run a small corpus (even 50 prompts) end to end and confirm the report shows per-metric deltas and the benchmark rows side by side.

**Acceptance Scenarios**:

1. **Given** a prompt corpus, **When** the end-to-end run completes, **Then** the report shows draft scores, rewrite scores, per-metric deltas, and the pinned benchmark rows in one artifact.
2. **Given** rewrite sets from different round counts (k=1..4), **When** each is scored, **Then** the report shows the metric trajectory across rounds so the quality-vs-drift sweet spot is visible.

---

### Edge Cases

- Very short texts (under a sentence): embedding and n-gram statistics become unstable; harness must either exclude them with a warning or document the minimum viable length.
- Small sample sizes: MMD and JMQ are noisy at low N; the report must state N and refuse (or loudly caveat) below a documented floor.
- Judge service unavailable or returning malformed verdicts (anything other than A/B): affected pairs are retried then excluded and counted, never silently scored.
- Non-English or garbage tokens appearing in rewrites (a known failure mode of the underlying models at high temperature): detected and flagged in the bundle.
- Semantic drift: rewrites that no longer say what the draft said must be detectable via the recorded per-round outputs and the degeneration checks (full semantic-preservation scoring is Phase 2, out of scope here, but the data to compute it must be captured now).
- Mismatched sets (different counts of model outputs and references, or unpaired prompts): rejected with a clear error before any scoring spend.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The harness MUST compute a distribution distance between two text sets using embeddings (MMD with the method pinned in the project references), a word-frequency distance (token L2 on 1-grams), and a pairwise judge win-rate (JMQ) from the same inputs in one invocation.
- **FR-002**: JMQ MUST use the six judge prompts already transcribed in `judge-prompts/` verbatim, randomize A/B order per pair, record every verdict with its order assignment, and report score as two times the model win-rate.
- **FR-003**: The harness MUST support a self-check mode (human set vs itself) and fail loudly if results fall outside documented noise bounds, so metric bugs cannot masquerade as findings.
- **FR-004**: All scoring runs MUST emit a machine-readable report recording inputs (set sizes, sources), configuration (seed, judge identity, embedding identity), and all metric values, sufficient to reproduce or audit the run later.
- **FR-005**: The cascade MUST accept either a prompt (generate-then-rewrite) or an existing draft (rewrite only), apply the paraphraser for a configurable number of rounds, and persist the draft plus every intermediate round with full run metadata.
- **FR-006**: The cascade MUST detect degenerate rewrites (empty, extreme length ratio, high repetition, non-English token bursts) per round, stop at the last good round, and flag the bundle.
- **FR-007**: The comparison report MUST place draft scores, rewrite scores, and the pinned published benchmark numbers (from PLAN.md) side by side with per-metric deltas.
- **FR-008**: Deterministic metrics MUST be exactly reproducible given the same inputs and seed; judge-based metrics MUST be recomputable from recorded verdicts without re-querying the judge.
- **FR-009**: The harness MUST validate input sets (pairing, counts, minimum lengths, minimum N) before incurring any external scoring cost.
- **FR-010**: The corpus builder MUST construct prompt/reference pairs by starting from human documents and reverse-generating a prompt per document, recording source and register per pair. The personal side corpus (James's published posts) MUST be tagged as such, and comparison reports MUST exclude it from benchmark rows.

### Key Entities

- **Text set**: A named collection of texts with a role (human reference, instruct draft, rewrite round k), each text paired to a prompt.
- **Rewrite bundle**: The output of one cascade run: prompt, draft, every rewrite round, degeneration flags, and run metadata (models, adapter, rounds, sampling settings).
- **Judge verdict**: One pairwise comparison: prompt, the two candidates, dimension, randomized order assignment, and the returned choice.
- **Metric report**: The scored result of comparing two text sets: all metric values, per-dimension JMQ, configuration, set sizes, and timestamps.
- **Benchmark table**: The pinned published numbers (DFT and superbaseline rows) that comparison reports reference; source of truth lives in PLAN.md.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The harness self-check (human vs human) produces distribution distance and word-frequency distance within documented noise bounds of zero and a judge win-rate between 45% and 55%, on every run.
- **SC-002**: Given prepared text sets, a researcher obtains a complete three-metric report from a single invocation with no manual intermediate steps.
- **SC-003**: On a corpus of at least 50 prompts, the cascade's final rewrites score closer to the human reference than the unrewritten drafts on at least two of the three metrics.
- **SC-004**: At the chosen round count, rewrites preserve the draft's meaning: no more than 5% of bundles are flagged for degeneration, and spot-read checks confirm the rewrite still answers the original prompt.
- **SC-005**: An independent external AI-text detector rates the rewritten set at least 30 percentage points more human than the draft set on the same corpus.
- **SC-006**: A full end-to-end run (50 prompts: generate, rewrite, score, report) completes on local hardware in under one day without manual babysitting.

## Assumptions

- The released HIP adapters (pinned in REFERENCES.md) are used as-is; no training or retraining is in scope for this feature.
- The primary evaluation corpus is built fresh from a FineWeb sample: human documents first, prompts reverse-generated with an LLM, mirroring the published protocol and its blog/news register. Sized to budget: benchmark-comparison runs target the published protocol's scale (2000 samples, 400 for judging) but the harness accepts smaller N with loud caveats. A secondary side corpus of James's own published posts supports personal-relevance spot checks only and is never used for benchmark comparison.
- The default JMQ judge is GPT-5.4-mini and the default MMD embedder is nvidia/llama-embed-nemotron-8b: the exact instruments the published protocol used, so the benchmark JMQ and MMD columns stay meaningful. Both remain configurable and the report always records which were used, since absolute values are only comparable within one instrument. Embeddings are cached per text set so the heavyweight embedder cost is paid once; small-N runs happen locally, full-scale (2000-sample) runs on rented compute.
- Benchmark numbers from the DFT post are treated as context for comparison, not as pass/fail gates: they were produced under a different judge and corpus, so only large gaps are meaningful.
- Default rewrite round count is 2 with a maximum of 4 exposed via configuration, per the plan's quality-over-evasion stance; full sweep tuning (stopping rules, semantic-preservation scoring) is Phase 2 and out of scope.
- Semantic preservation in this feature is guarded by degeneration detection and captured per-round data only; formal semantic scoring arrives in Phase 2.
