# Command reference

Every `dehip` parameter, its valid values, and its default. This is the written
version; `dehip <command> --help` is the always-current source, generated from the
same argparse definitions. Formal contract and exit codes:
`specs/001-hip-cascade-harness/contracts/cli.md`.

## Global

`--seed` goes before the subcommand: `dehip --seed 42 generate ...`.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--seed` | integer | `0` | Recorded in every command's output summary. |

Exit codes (all commands): `0` success, `2` validation failure, `3` external
dependency failure (HIP checkout or API auth), `4` self-check out of bounds, `5`
report-write I/O failure, `6` detect: SC-005 delta requested but not computable.

## build-corpus

Sample a human-reference corpus and write its manifest.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--tier` | `smoke`, `judged`, `full` | `smoke` | 50 / 400 / 2000 pairs. |
| `--corpus` | `fineweb`, `personal` | `fineweb` | Source family. |
| `--source` | path or URLs | none | Required when `--corpus personal`. |
| `--out` | path | derived | Output corpus JSONL. |
| `--yes` | flag | off | Confirm prompt-generation spend above the threshold. |

## generate

Write stage-1 instruct drafts for a corpus.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--corpus` | path | (required) | Corpus JSONL to draft from. |
| `--model` | HF model id | `Qwen/Qwen3-4B-Instruct-2507` | Default is revision-pinned (see docs/security.md); a custom id is unpinned. |
| `--temperature` | float | `0.7` | Sampling temperature. |
| `--top-p` | float | `0.95` | Nucleus sampling. |
| `--out` | path | derived | Output run directory. |

## rewrite

Cascade stage: k rounds of hip-run over the drafts.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--run` | path | none | Run directory to continue from `generate`. |
| `--draft-file` | path | none | Rewrite-only mode: a draft file, skips `generate`. |
| `--adapter` | HF id or local path | `YixuanEvenXu/Qwen3-4B-Base-HIP-adapter` | The HIP LoRA tier. |
| `--rounds` | integer | `2` | Max 4. |
| `--temperature` | float | `1.0` | Matches the HIP inference protocol. |
| `--top-p` | float | `0.95` | Matches the HIP inference protocol. |
| `--hip-repo` | path | `../humanization-by-iterative-paraphrasing` | The HIP sibling checkout. |
| `--out` | path | `results/runs/` | Output run directory for `--draft-file` mode. |

Use either `--run` (continue a generate run) or `--draft-file` (rewrite-only).

## score

Compute the metrics for a candidate set against human references.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--candidate` | path | (required) | Candidate set manifest. |
| `--reference` | path | (required) | Reference set manifest. |
| `--metrics` | comma-separated subset of `mmd`, `token_l2`, `jmq` | `mmd,token_l2,jmq` | Pick any subset to run fewer metrics. |
| `--judge` | model id | `gpt-5.4-mini` | JMQ pairwise judge. |
| `--embedder` | HF model id | `nvidia/llama-embed-nemotron-8b` | MMD embedder (revision-pinned). |
| `--out` | path | derived | Output report path. |
| `--recompute-jmq-from` | path | none | Re-aggregate from a `verdicts.jsonl` with no API calls. |
| `--prompts` | path | none | Prompts JSONL (`{pair_id, prompt}`); required when `jmq` runs. |
| `--yes` | flag | off | Confirm JMQ spend above the threshold. |

`--metrics` including `jmq` needs `--prompts` and an `OPENAI_API_KEY`.

## self-check

Split a human set in half and assert the metrics read near zero (the instrument
calibration gate).

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--reference` | path | (required) | Human set manifest. |
| `--skip-jmq` | flag | off | Run only MMD and token-L2 (no judge, no `OPENAI_API_KEY`). |

## detect

Score one or more sets through an external AI-text detector.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--sets` | one or more paths | (required) | Set manifests, space-separated. |
| `--detector` | `pangram`, `gptzero` | `pangram` | Needs `PANGRAM_API_KEY` or `GPTZERO_API_KEY`. |
| `--out` | path | derived | Output report path. |
| `--yes` | flag | off | Confirm detector spend above the threshold. |

Passing exactly one draft set and one rewrite set computes the SC-005 delta;
other combinations score each set but return exit `6` for the delta.

## report

Assemble the draft-versus-rewrite comparison.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--draft-report` | path | none | Draft MetricReport. |
| `--rewrite-report` | path (repeatable) | none | Rewrite MetricReport; repeat for a k-trajectory table. |
| `--benchmark` | flag | off | Attach the pinned published-benchmark rows. |
| `--out` | path | derived | Output report path. |

`--benchmark` is refused if any input report scored a `personal`-corpus set.

## Environment variables

| Variable | Needed by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `score` (jmq), `self-check` without `--skip-jmq` | JMQ judge. |
| `PANGRAM_API_KEY` | `detect --detector pangram` | Pangram detector. |
| `GPTZERO_API_KEY` | `detect --detector gptzero` | GPTZero detector. |
| `HF_TOKEN` | any real-model stage | Only if HF streaming or downloads rate-limit. |
| `DEHIP_EMB_CACHE_DIR` | any MMD scoring | Override the embedding cache directory (defaults to `data/emb-cache`). |
