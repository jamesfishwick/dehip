# dehip

An open alternative to Rosmine's proprietary Distribution Fine Tuning, built on HIP (Humanization by Iterative Paraphrasing, arXiv 2605.19516). The idea is a two-stage cascade: an instruct model writes the content, then a LoRA-tuned base-model paraphraser rewrites it back toward the human text distribution. Output quality gets scored with the eval harness the DFT post specified but never needed to hide: MMD over embeddings, pairwise judge win-rate (JMQ), and token-frequency L2 distance.

HIP's published results are detector-evasion numbers. Nobody has shown the rewrites are better prose. That question is the point of this project.

See [PLAN.md](PLAN.md) for the full research plan and [REFERENCES.md](REFERENCES.md) for verified sources. Everything cited here was checked against the primary source before it went in.

## Status

The harness is implemented. All seven `dehip` subcommands are built and covered by unit and integration tests (322 tests): `build-corpus`, `generate`, `rewrite`, `score`, `self-check`, `detect`, and `report`. Each stage reads and writes the file formats in [the data model](specs/001-hip-cascade-harness/data-model.md), gates spend before any judge or detector call, and exits with the codes in [the CLI contract](specs/001-hip-cascade-harness/contracts/cli.md).

What each command does is implemented and tested against injected seams (stub models, judges, and detectors), so the pipeline logic, resumability, cost gates, and exit codes are exercised without downloads or paid calls. Real-instrument validation is a separate step and has not been run: the smoke run on real models, the SC-003 through SC-006 outcomes, and any wall-clock or quality numbers are pending (issue #16). No claim here has been measured on real models.

To drive the pipeline end to end, see [the quickstart](specs/001-hip-cascade-harness/quickstart.md); `just smoke-test` runs that sequence, given the prerequisites (uv, an `OPENAI_API_KEY` for the judge, the HIP sibling checkout, and the models).
