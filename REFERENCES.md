# References

Every entry below was verified against the primary source on 2026-08-01. Verdicts note where a secondary source (the `harshaneel/humanize` README) characterized a paper inaccurately.

## Key sources

### Base Models Look Human To AI Detectors (HIP)

- arXiv 2605.19516. Yixuan Even Xu, Ziqian Zhong, Aditi Raghunathan, Fei Fang, J. Zico Kolter (CMU).
- https://arxiv.org/abs/2605.19516
- Verdict: verified, the key paper. Base-model output reads as human to GPTZero and Pangram (Llama3-8B base: 96.7% / 98.8% human; instruct: 30.3% / 17.1%). Detectors track instruction-tuning artifacts. Introduces HIP, a LoRA-trained base-model paraphraser applied iteratively.
- Code: https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing (confirmed live)
- Adapters (all confirmed on HuggingFace under `YixuanEvenXu/`):
  - Qwen3-0.6B-Base-HIP-adapter, Qwen3-0.6B-HIP-adapter
  - Qwen3-1.7B-Base-HIP-adapter, Qwen3-1.7B-HIP-adapter
  - Qwen3-4B-Base-HIP-adapter, Qwen3-4B-HIP-adapter
  - Qwen3-8B-Base-HIP-adapter, Qwen3-8B-HIP-adapter
  - Qwen3-14B-Base-HIP-adapter, Qwen3-14B-HIP-adapter
  - Llama-3-8B-HIP-adapter, Llama-3-8B-Instruct-HIP-adapter
  - Llama-3-70B-HIP-adapter, Llama-3-70B-Instruct-HIP-adapter

### Fixing LLM writing with Distribution Fine Tuning (Rosmine)

- Blog post, May 2026. Archived: https://web.archive.org/web/20260519133855/https://rosmine.ai/2026/05/18/fixing-llm-writing-with-distribution-fine-tuning/ (canonical: https://rosmine.ai/?p=753)
- Local copy: `~/Downloads/Fixing LLM writing with Distribution Fine Tuning.pdf`
- Verdict: algorithm proprietary and undisclosed. Fully specifies the eval harness (MMD, JMQ, token L2), data construction, baselines, and judge prompts (Appendix 9). Source of the benchmark numbers in PLAN.md.

### harshaneel/humanize

- https://github.com/harshaneel/humanize
- Verdict: the lead source. Two inference-time prompt skills, no training content. Its "ceiling" and "complementary techniques" sections pointed to the papers below. Citation quality is mixed, see flags.

## Supporting papers (verified accurate)

### Adversarial Paraphrasing

- arXiv 2506.07001. Cheng, Sadasivan, Saberi, Saha, Feizi (Maryland).
- Verdict: real, the 87.88% figure checks out (average T@1%F reduction under OpenAI-RoBERTa-Large guidance). Caveat: the humanize README calls it "detector-scored best-of-N" but the mechanism is detector-guided paraphrasing. Training-free.

### PADBen

- arXiv 2511.00416. Zha, Min, Sushmita.
- Verdict: real, accurately cited. Detectors fail catastrophically on iteratively paraphrased text (the "intermediate laundering region"). 11 detectors evaluated.

### DAMAGE

- arXiv 2501.03437.
- Verdict: real, accurately cited. Evaluated 19 AI humanizer tools; many detectors fail on humanized text, but a robust detector trained with data-centric augmentation survives. Supports the claim that single cross-model rewriting alone does not defeat trained detectors.

### HyPerAlign

- arXiv 2505.00038. Garbacea, Tan.
- Verdict: real. Caveat: humanize README calls it "writer-profile distillation" but it is hypothesis-based personalized alignment (infers a user's style hypotheses from few samples), not distillation or fine-tuning. Win-rates >90% vs preference fine-tuning on authorship tasks.

### Authorship impersonation

- arXiv 2603.29454.
- Verdict: real, defensibly cited. LLM impersonation of specific authors fails to bypass authorship verification. Attributed to higher lexical diversity and entropy in LLM text.

## Flagged citations (do not rely on the humanize README's description)

### PIFE, arXiv 2510.02319

- Actual paper: "Modeling the Attack: Detecting AI-Generated Text by Quantifying Adversarial Perturbations." PIFE = Perturbation-Invariant Feature Engineering, a detection-hardening method (82.6% accuracy under semantic attack vs 48.8% for adversarial training).
- The humanize README miscast it as an "iterative paraphrase pass" evasion technique. It is the opposite: a defense.

### arXiv 2601.07974

- Actual paper: "Explaining Generalization of AI-Generated Text Detectors Through Linguistic Analysis." 80 linguistic features, correlation analysis across 6 prompting strategies, 7 LLMs, 4 domains.
- The humanize README described it as "EACL 2026 SHAP analysis" showing features are "dataset-specific." No SHAP, no EACL claim in the abstract. Direction of the finding is loosely compatible, description is not.

## Metric background (from the DFT post's own references)

- MMD: Gretton et al., kernel two-sample test. Embeddings: nvidia/llama-embed-nemotron-8b (arXiv 2511.07025).
- MAUVE (Pillutla et al., arXiv 2102.01454): rejected by the DFT post because it saturates (.997+ even for a 4B baseline).
- FID (Heusel et al., arXiv 1706.08500): tested but rejected, assumes Gaussian distributions.
- FineWeb (Penedo et al., arXiv 2406.17557): the DFT training corpus.
- LLM judge bias: Laurito et al. (AI-AI bias, PNAS 2025), Panickssery et al. (self-preference, NeurIPS 2024). Why JMQ flatters models and must be read alongside MMD and token L2.
