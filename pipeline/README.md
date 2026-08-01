# pipeline

Phases 0 and 2: the instruct -> HIP cascade. Nothing implemented yet.

Stage 1: instruct model (Qwen3-4B or 8B instruct) generates content from a prompt.
Stage 2: HIP base-model paraphraser (released LoRA adapters, see REFERENCES.md) rewrites for k rounds at temperature 1.0, top-p 0.95.

Phase 2 adds the stopping rule: iterate until MMD or Pangram plateaus, or the HIP repo's semantic judge drops below a floor. Target k is 2-4, not the paper's evasion-optimized 10.
