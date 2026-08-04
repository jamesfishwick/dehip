# Phase 0 Research: HIP Cascade and Evaluation Harness

All unknowns from Technical Context, resolved. Sources: the verified session record in PLAN.md / REFERENCES.md (repo root), the HIP repo README and pyproject (fetched 2026-08-03), and the DFT post PDF.

## R1: Language and toolchain

- **Decision**: Python 3.12, uv-managed project, `pyproject.toml` with pinned ranges mirroring the HIP repo's style.
- **Rationale**: The HIP repo requires Python >=3.11 and installs with `uv sync`; torch/transformers/peft is the only ecosystem where the adapters, embedder, and datasets all load natively. Any other language would wrap this one anyway.
- **Alternatives considered**: None credible for this stack.

## R2: Cascade rewrite stage — wrap `hip-run`, do not reimplement

- **Decision**: The rewrite stage emits a YAML config + JSONL input, invokes `uv run hip-run --config ...` in the sibling HIP checkout as a subprocess, and parses its JSONL output. Per-round outputs are captured by running rounds explicitly (k invocations of one round each) rather than one k-round invocation, so degeneration checks (FR-006) run between rounds and the bundle records every intermediate.
- **Rationale**: The HIP repo ships exactly this CLI surface (`hip-run --config`, JSONL in/out, YAML configs, smoke configs for Qwen3-0.6B) and released adapters that its code is guaranteed to load correctly (chat-template and completion-format details live in their code, not ours). Importing HIP as a library would couple us to internals with no stability promise; reimplementing risks silently diverging from the published method — the exact failure mode this project exists to avoid.
- **Alternatives considered**: (a) Vendor/import HIP internals — rejected, unpinned internals, fork risk. (b) Reimplement iterative inference in dehip — rejected, method-fidelity risk for zero benefit. (c) One k-round `hip-run` call — rejected, loses per-round capture and inter-round degeneration gating unless their config supports emitting intermediates (revisit if it does; config inspection is an implementation task).

## R3: Draft generation stage

- **Decision**: Direct transformers generation in-process (`generate.py`), `Qwen/Qwen3-4B-Instruct-2507` default (exact HF id to be verified at implementation; 8B configurable), MPS locally / CUDA rented, default sampling temperature 0.7, top-p 0.95, all settings recorded in bundle metadata.
- **Rationale**: Stage 1 is plain instruct generation — no HIP involvement — and transformers on MPS handles a 4B comfortably within 36GB. T=0.7 is the DFT post's best-judge-preference SFT setting, a defensible draft-quality default; the cascade's premise is that the paraphraser, not sampler tuning, fixes the distribution.
- **Alternatives considered**: llama-server/llama.cpp (GGUF) — faster on Mac but adds a second runtime, a conversion step, and sampler-implementation drift vs the transformers stack the rest of the pipeline uses; revisit only if MPS throughput threatens SC-006.

## R4: Embedder runtime and cache

- **Decision**: `nvidia/llama-embed-nemotron-8b` via transformers, fp16/bf16, batch embedding with an on-disk cache keyed by (model id, content SHA-256). Cache is append-only parquet + npy under `data/emb-cache/`.
- **Rationale**: Exact-protocol instrument per clarification Q3. 8B fp16 is ~16GB — fits MPS on the 36GB Mac for batched forward passes at small N; the cache makes every text's embedding a one-time cost, so even the 2000-sample corpus embeds once, on whichever hardware ran it, and MMD experiments after that are free.
- **Alternatives considered**: GGUF via llama-server embeddings endpoint — no verified GGUF of this embedder exists; conversion fidelity for a benchmark instrument is not worth the risk. Smaller embedders — rejected by clarification Q3.

## R5: MMD implementation

- **Decision**: Unbiased MMD^2 estimator (Gretton et al.), Gaussian RBF kernel, median-heuristic bandwidth computed on the pooled sample, bandwidth recorded in every report. Pure numpy, known-answer unit tests (identical Gaussians → ~0; shifted Gaussians → known positive).
- **Rationale**: This is the standard estimator the DFT post cites (Gretton). One honest caveat, documented in the report schema: the post does not publish its kernel bandwidth, so even with the identical embedder, absolute MMD comparability to the benchmark rows is best-effort. This is consistent with the spec's assumption that benchmark numbers are "context, not gates" and only large gaps are meaningful. The self-check (FR-003) plus a plain-instruct baseline run calibrate our own scale end-to-end.
- **Alternatives considered**: Linear-time MMD estimator — unnecessary at N≤2000. Learned/other kernels — off-protocol.

## R6: JMQ judge integration

- **Decision**: OpenAI API, model `gpt-5.4-mini` (exact-protocol per clarification Q2), one call per (pair, dimension) using the six verbatim `judge-prompts/` templates. Seeded RNG decides A/B order per pair; every verdict persisted as JSONL (prompt id, dimension, order assignment, raw response, parsed choice) before aggregation. Malformed verdicts: one retry, then excluded-and-counted (edge-case rule from spec). Concurrency-limited client with cost preflight: the validator prints estimated call count (pairs × 6) and requires explicit `--yes` above a configurable spend threshold.
- **Rationale**: Verdict-level persistence makes JMQ recomputable without re-querying (FR-008) and auditable for positional bias (Story 1, scenario 4). Cost control is FR-009's "validate before scoring spend" made concrete.
- **Alternatives considered**: OpenAI Batch API (cheaper, ~24h latency) — worth adding for the 400-pair benchmark tier later; synchronous keeps the smoke tier simple. Multi-judge panels — Phase 2+ consistency check, out of scope.

## R7: Token L2 tokenization

- **Decision**: Tokenize with the Qwen3 tokenizer (the model family under test), 1-gram frequency vectors over the union vocabulary of both sets, L2 distance. Tokenizer identity recorded in the report.
- **Rationale**: The DFT post does not name its tokenizer; its models are Qwen3, making the Qwen3 tokenizer the most faithful guess and internally consistent for our draft-vs-rewrite deltas (which is what SC-003 actually needs). Recorded identity keeps the caveat honest.
- **Alternatives considered**: Whitespace/word tokenization — diverges from any plausible protocol reading; cl100k-style neutral tokenizer — no better claim to comparability.

## R8: Corpus construction (FR-010, clarification Q1)

- **Decision**: Stream `HuggingFaceFW/fineweb` (sample-10BT config) via `datasets`, filter to blog/news-register English docs of 150-1200 words, quality-screen (strip boilerplate-heavy docs), sample to tier size with a recorded seed. Reverse-generate one prompt per document with `gpt-5.4-mini` (prompt-variant rotation for diversity, mirroring the protocol's approach), storing generator identity per pair. Side corpus: ingest James's published posts from their public URLs/repo into the same pair schema, `corpus: "personal"` tag, excluded from benchmark rows by the report layer.
- **Rationale**: Mirrors the published protocol's construction (human doc first, prompt derived) and register, per clarification Q1. Using the judge model for prompt generation avoids introducing a third API dependency; generator id is recorded so this can be revisited.
- **Alternatives considered**: HIP's released eval dataset (`YixuanEvenXu/HIP-training-and-evaluation-data`) — wrong register (RAID/MAGE) for the benchmark story, but useful as a free extra validation set for the harness itself; kept as an optional input, not the primary corpus. Local Qwen3-32B for prompt generation — a ~19GB quantized load on the Mac for no fidelity gain over the API path.

## R9: Storage formats

- **Decision**: JSONL for anything per-text (text sets, bundles, verdicts); JSON for reports; schemas versioned with a `schema_version` field from day one. Reports also render a human-readable markdown table. Everything under `data/` (inputs, cache) and `results/` (outputs), both gitignored.
- **Rationale**: File-based stages are resumable, diffable, and auditable — the properties FR-004 and FR-008 demand. No database earns its keep at N≤2000.
- **Alternatives considered**: SQLite for verdicts — adds query convenience but breaks the "every artifact is a plain file you can cat" audit property; revisit if verdict volume grows.

## R10: Degeneration detection thresholds (FR-006)

- **Decision**: Per-round checks, all recorded per bundle: empty/whitespace output; length ratio vs prior round outside [0.5, 2.0]; 3+ consecutive sentences sharing a start word (the DFT post's own repetitiveness probe, which human text trips at ~17%, so this flags rather than hard-fails); non-ASCII burst above 5% of characters in an English-register run (the post measured 0.1% in human data, 8.1% in DFT output, up to 36% in high-temp SFT). Any hard trip stops iteration at the last good round and flags the bundle.
- **Rationale**: Every threshold is anchored to a measured value from the verified session record rather than invented; flag-vs-fail distinction keeps a known-noisy human signal (repetition) from killing valid runs.
- **Alternatives considered**: Semantic-similarity gating per round — that is Phase 2's formal semantic scoring; here we only capture the data for it.
