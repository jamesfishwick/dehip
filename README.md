# dehip

An open alternative to Rosmine's proprietary Distribution Fine Tuning, built on HIP (Humanization by Iterative Paraphrasing, arXiv 2605.19516). The idea is a two-stage cascade: an instruct model writes the content, then a LoRA-tuned base-model paraphraser rewrites it back toward the human text distribution. Output quality gets scored with the eval harness the DFT post specified but never needed to hide: MMD over embeddings, pairwise judge win-rate (JMQ), and token-frequency L2 distance.

HIP's published results are detector-evasion numbers. Nobody has shown the rewrites are better prose. That question is the point of this project.

See [PLAN.md](PLAN.md) for the full research plan and [REFERENCES.md](REFERENCES.md) for verified sources. Everything cited here was checked against the primary source before it went in.

## Status

The harness is implemented and has been run end to end on real models. All seven `dehip` subcommands are built and covered by unit and integration tests (354 tests). The first real-instrument run (issue #16) drove the whole pipeline against real instruments: FineWeb human references, Qwen3-4B-Instruct drafts, the released HIP paraphraser for the rewrite, `nvidia/llama-embed-nemotron-8b` for MMD, GPT-5.4-mini for JMQ, and Pangram for the external detector.

The result: humanization moves the distribution metrics (MMD, token-L2, the detector) toward human but not the quality judge (JMQ). At 0.6B the quality penalty is a collapse (-0.39 JMQ); at 4B it shrinks to -0.11 and clears the detector bar. The story is a scaling curve toward quality-neutral, not a yes or no. Full scorecard in [docs/smoke-run-findings.md](docs/smoke-run-findings.md) (rendered [PDF](docs/smoke-run-findings.pdf)).

## Commands

### Running the CLI

`dehip` is a console script inside the project's uv-managed virtualenv, not on your global PATH, so a bare `dehip` gives "command not found". Three ways to run it:

- `uv run dehip <command>` — no setup, uses the project env. This is what the `just` recipes and the quickstart use.
- `source .venv/bin/activate` then `dehip <command>` — bare `dehip` works for that shell session.
- `uv tool install --editable .` — puts `dehip` on your global PATH so it works from anywhere (`--editable` picks up code changes without reinstalling).

Single entry point `dehip`, one subcommand per pipeline stage. Every stage reads and writes the formats in [the data model](specs/001-hip-cascade-harness/data-model.md), prints a JSON summary to stdout and human-readable progress to stderr, gates spend before any paid call, and takes `--seed` (recorded in its output). For every flag's valid values and defaults, see the [command reference](docs/commands.md) (or run `dehip <command> --help`); the formal contract and exit codes are in [the CLI contract](specs/001-hip-cascade-harness/contracts/cli.md).

| Command | What it does |
|---|---|
| `dehip build-corpus` | Sample a human-reference corpus and write its manifest. `--tier` smoke/judged/full = 50/400/2000 pairs; `--corpus` fineweb or personal. |
| `dehip generate` | Write instruct-model drafts for a corpus (default model Qwen3-4B-Instruct-2507). |
| `dehip rewrite` | Rewrite drafts through a HIP paraphraser via `hip-run`. `--adapter` selects the tier, `--rounds` the iteration count (max 4). |
| `dehip score` | Compute MMD, token-L2, and JMQ for a candidate set against human references. Preflights JMQ spend behind `--yes`. |
| `dehip self-check` | Split the human set in half and assert the metrics read near zero. The instrument-calibration gate; exits 4 if out of bounds. |
| `dehip detect` | Score sets through an external AI detector (Pangram default). Needs `PANGRAM_API_KEY`. |
| `dehip report` | Assemble the side-by-side draft-vs-rewrite comparison, plus the published-benchmark row with `--benchmark`. |

**Exit codes:** 0 success, 2 validation failure, 3 external dependency failure (HIP checkout or API auth), 4 self-check out of bounds, 5 artifact write failure, 6 detect delta not computed.

### Recipes

`just --list` shows them all. The ones you reach for:

| Recipe | What it does |
|---|---|
| `just smoke-test` | The quickstart end to end: build-corpus, self-check, generate, rewrite, score, report. |
| `just clone-hip` | Clone and install the HIP sibling checkout next to this repo. |
| `just fetch-adapter <size>` | Download a released HIP LoRA adapter (0.6B, 1.7B, 4B, 8B, 14B). |
| `just run-cascade <size>` | Run the full cascade with a chosen adapter, tagged by size. The 8B/14B tiers need a GPU: see [docs/gpu-runbook.md](docs/gpu-runbook.md). |
| `just render-findings` | Render the findings to a branded PDF. |
| `just lint` | `ruff check` (gated at commit time by the pre-commit hook). |

To drive the pipeline end to end, see [the quickstart](specs/001-hip-cascade-harness/quickstart.md); `just smoke-test` runs that sequence, given the prerequisites (uv, an `OPENAI_API_KEY` for the judge, the HIP sibling checkout, and the models). `dehip detect` additionally needs a `PANGRAM_API_KEY`.
