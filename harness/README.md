# harness

Phase 1: the Rosmine eval metrics. Nothing implemented yet.

- `token_l2` -- n-gram frequency vectors over model vs reference outputs, Euclidean distance, 1-grams primary
- `mmd` -- embed with nvidia/llama-embed-nemotron-8b, Gaussian RBF kernel, unbiased MMD^2 estimator
- `jmq` -- pairwise judge win-rate vs human references, randomized A/B order, templates in ../judge-prompts/

Sanity check before trusting any of it: human reference vs itself should give MMD near 0 and JMQ near 0.5 by construction.
