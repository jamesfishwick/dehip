# HIP vs `pangram-4`: a negative result

## Question

Does HIP (Humanization by Iterative Paraphrasing) rewrite AI text so it reads as human to Pangram's current `pangram-4` detector? The bar is SC-005: the rewrite set's mean human-probability minus the draft set's mean, at least 0.30.

## Setup

30 AI drafts seeded from FineWeb, all scored 1.0 AI by `pangram-4`. Each draft runs through the HIP cascade at k=4, then `pangram-4` scores the survivors. Degenerate collapses count as failures at human-prob 0.0 over the full 30-draft universe, not dropped from the denominator. Uncertainty is an imprecise-probability envelope with a three-way verdict: `robust_pass`, `robust_fail`, or `indeterminate`.

## Results

| Paraphraser | degenerated | delta (effective) | envelope | verdict |
| --- | --- | --- | --- | --- |
| `Qwen3-4B-Base` (temp 1.0) | 11/30 | 0.076 | [0.009, 0.134] | `robust_fail` |
| `Qwen3-4B-Base` (temp 0.7) | 9/30 | 0.067 | [0.000, 0.125] | `robust_fail` |
| Llama-3-8B (temp 1.0) | 5/30 | 0.092 | [0.023, 0.148] | `robust_fail` |

## Finding

All three fail decisively. The entire uncertainty envelope sits below 0.30 in every run.

Scale is a real lever but a weak one. Going from 4B to 8B cut degeneration from 11 to 5 and lifted the delta from 0.076 to 0.092, about +0.016 per doubling. Projecting to 70B (roughly three more doublings) lands near 0.14 to 0.20, still well short of 0.30. Reaching the bar from 0.092 would take on the order of thirteen doublings, an unreachable model size.

## Why the HIP paper disagrees

The paper evaluated against Pangram v3 (`configs/eval_pangram.yaml` sets `detector_version: pangram_v3`). `pangram-4` is newer. HIP beat the detector of its day. The detector has since caught up, and a bigger paraphraser does not buy back the gap.

## Caveats

- n = 30, one detector, one draft source (Qwen instruct).
- Temperature 0.7 did not help. It cut degeneration but also cut humanization, leaving the delta flat.
- We did not run the 70B. The 8B trend predicts it also fails, so running it would confirm the finding.
