# dehip

An open alternative to Rosmine's proprietary Distribution Fine Tuning, built on HIP (Humanization by Iterative Paraphrasing, arXiv 2605.19516). The idea is a two-stage cascade: an instruct model writes the content, then a LoRA-tuned base-model paraphraser rewrites it back toward the human text distribution. Output quality gets scored with the eval harness the DFT post specified but never needed to hide: MMD over embeddings, pairwise judge win-rate (JMQ), and token-frequency L2 distance.

HIP's published results are detector-evasion numbers. Nobody has shown the rewrites are better prose. That question is the point of this project.

See [PLAN.md](PLAN.md) for the full research plan and [REFERENCES.md](REFERENCES.md) for verified sources. Everything cited here was checked against the primary source before it went in.

## Status

Planning. No implementation yet. Phase 0 (zero-training pipeline with released HIP adapters) is next.
