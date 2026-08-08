# Data Model: HIP Cascade and Evaluation Harness

File-based model: every entity is a JSONL record (per-text) or JSON document (per-run). All schemas carry `schema_version: 1`. Identity rule: texts are identified by `pair_id` (stable across corpus, draft, and rewrite artifacts) plus content SHA-256 where deduplication matters (embedding cache).

## Pair (corpus record)

One prompt + human reference. Produced by `dehip build-corpus`. JSONL, one per line.

| Field | Type | Notes |
|---|---|---|
| schema_version | int | 1 |
| pair_id | string | stable id, `{corpus}-{seq}` |
| corpus | string | `fineweb` or `personal`; `personal` is excluded from benchmark rows (FR-010) |
| prompt | string | reverse-generated (fineweb) or reconstructed (personal) |
| reference_text | string | the human document |
| source | object | provenance: dataset id + doc id, or URL |
| register | string | e.g. `blog`, `news` |
| prompt_generator | string | model id that generated the prompt |
| word_count | int | validation input (150-1200 for fineweb tier) |

**Validation** (FR-009, FR-010): pair_id unique; corpus tag present; word_count within tier bounds; personal pairs never enter benchmark scoring.

## TextSet (manifest)

A named, role-tagged collection referencing Pair ids — the unit the harness scores. JSON manifest pointing at a JSONL of `{pair_id, text}` records.

| Field | Type | Notes |
|---|---|---|
| set_id | string | e.g. `fineweb-400-human`, `fineweb-400-draft`, `fineweb-400-rewrite-k2` |
| role | enum | `human_reference` / `instruct_draft` / `rewrite` |
| round | int? | rewrite round k, only for role=rewrite |
| corpus | string | must be homogeneous within a set |
| pair_ids | list | membership; scoring requires both sets share identical pair_ids (pairing validation, FR-009) |
| provenance | object | for generated sets: the RewriteBundle run id or generation run id that produced it |

**State transitions**: none — sets are immutable once written; a new run writes a new set.

## RewriteBundle

Output of one cascade run for one pair (FR-005, FR-006). JSONL, one bundle per pair, grouped in a run directory `results/runs/{run_id}/`.

| Field | Type | Notes |
|---|---|---|
| schema_version | int | 1 |
| run_id | string | timestamped run identifier |
| pair_id | string | joins back to Pair |
| prompt | string | as used |
| draft | object | `{text, model_id, sampling: {temperature, top_p, seed}}`; absent when rewrite-only input |
| rounds | list | per round: `{k, text, flags: [..]}` — every intermediate kept |
| final_round | int | last good round (may be < requested k after a hard degeneration trip) |
| degeneration | object | per-check results: empty, length_ratio, repetition (flag-only), non_ascii_burst; `hard_tripped: bool` |
| adapter_id | string | HIP adapter used |
| hip_config | object | the YAML emitted to hip-run, inlined for audit |
| requested_k | int | configured round count (default 2, max 4) |

**Validation**: rounds non-empty unless draft-only; final_round ≤ requested_k; hard_tripped ⇒ bundle flagged in downstream set assembly.

## JudgeVerdict

One (pair, dimension) comparison (FR-002, FR-008). JSONL written before aggregation. The `dehip score` command has no run_id concept and writes verdicts beside its report at `<out-dir>/verdicts.jsonl` (the directory of `--out`). The rewrite/run pipeline (a later ticket) is where a `results/runs/{run_id}/verdicts.jsonl` layout applies.

| Field | Type | Notes |
|---|---|---|
| schema_version | int | 1 |
| pair_id | string | |
| dimension | enum | overall / clarity / coherence / creativity / depth / relevance |
| judge_model | string | default `gpt-5.4-mini` |
| order | enum | `model_first` / `human_first` — the randomized assignment, seeded |
| raw_response | string | verbatim judge output |
| choice | enum | `A` / `B` / `invalid` |
| model_won | bool? | derived from choice + order; null when invalid |
| retry_count | int | 0 or 1; invalid after retry ⇒ excluded-and-counted |

**Validation**: order distribution over a run should be ~50/50 (audit query, Story 1 scenario 4); invalid count reported, never silently dropped.

## EmbeddingCacheEntry

Append-only cache (R4). Parquet index + npy vectors under `data/emb-cache/`.

| Field | Type | Notes |
|---|---|---|
| content_sha256 | string | cache key (composite with embedder_id) |
| embedder_id | string | composite key part — same text under a different embedder is a different entry |
| vector_ref | string | offset into the npy store |

## MetricReport

Scored comparison of two TextSets (FR-001, FR-004, FR-007). JSON + rendered markdown under `results/reports/`.

| Field | Type | Notes |
|---|---|---|
| schema_version | int | 1 |
| report_id | string | |
| compared | object | `{candidate_set, reference_set}` with set sizes (N stated, per edge-case rule) |
| config | object | seed, judge_model, embedder_id, tokenizer_id, mmd_bandwidth (recorded, R5), thresholds |
| mmd | number | unbiased MMD^2 |
| token_l2 | number | 1-gram L2 |
| jmq | object | overall + per-dimension: `{score, wins, losses, invalid, n}` |
| caveats | list | auto-attached: small-N warning below floor, bandwidth comparability note, cross-judge note if non-default judge |
| benchmark_rows | list? | pinned rows from PLAN.md when comparison requested; never includes `personal`-corpus results (FR-010) |
| deltas | object? | candidate vs a second report (draft vs rewrite), per metric |
| timestamps | object | started/finished |

**Validation** (FR-003, FR-009): refuses mismatched pair_ids; self-check mode asserts MMD and token_l2 within documented noise bounds and JMQ win-rate in [0.45, 0.55], failing loudly otherwise.

## BenchmarkTable

Static, pinned in code from PLAN.md (source of truth stays in PLAN.md; the constant cites it). Rows: 4B/8B/14B SFT superbaseline and DFT, with MMD / JMQ / token-L2 values. Attached to reports by reference, flagged `external_protocol: true` so rendering always shows the different-judge/different-corpus caveat.

## Relationships

```text
Pair 1..n ──< TextSet (by pair_id membership)
Pair 1──1 RewriteBundle (per run) ──> produces rewrite TextSets (one per round)
TextSet x TextSet ──> MetricReport (candidate vs reference)
MetricReport ──> JudgeVerdict * (jmq recomputable from verdicts, FR-008)
MetricReport ──? BenchmarkTable (comparison reports only)
```
