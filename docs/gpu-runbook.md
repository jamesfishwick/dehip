# GPU Runbook: the 8B and 14B cascade

The 0.6B and 4B runs (see smoke-run-findings.md) ran on a Mac through hip-run on
CPU. The 8B and 14B adapters want a GPU. hip-run selects CUDA automatically when a
GPU is present and never uses MPS, so the harness code does not change between the
laptop and the GPU box. Only the hardware and the adapter size differ.

This runbook is the whole procedure for running the larger tiers on a CUDA
machine. Nothing here has been run yet; the 8B/14B result is the open item the
findings call out.

## Why not on the Mac

hip-run runs on CUDA or CPU. On this Mac that means CPU, and CPU paraphrase
generation for an 8B base over 50 drafts and two rounds runs for hours before the
first score. The 14B base in fp16 is about 28GB, which is tight against 36GB of
unified memory once the embedder and the OS take their share. A GPU removes both
problems.

## Prerequisites

- A CUDA GPU sized for the tier (see the table below).
- The HIP sibling checkout next to this repo, installed: `just clone-hip`.
- `OPENAI_API_KEY` exported for the JMQ judge (GPT-5.4-mini).
- A Hugging Face token (`HF_TOKEN`) if the FineWeb stream or the adapter download
  rate-limits.
- The adapter fetched locally: `just fetch-adapter 8B` (or `14B`).

## Sizing

Rough VRAM for the base model in fp16 plus the LoRA and generation activations.
Add headroom; these are floors, not targets.

| Adapter | Base fp16 | Practical VRAM | Example card |
|---|---|---|---|
| 0.6B / 1.7B / 4B | 1-8 GB | any modern GPU, or CPU | anything, incl. the Mac |
| 8B | ~16 GB | 20-24 GB | RTX 3090 / 4090, A10, L4 |
| 14B | ~28 GB | 32-48 GB | A100 40GB, L40S 48GB |

If VRAM is short, the base can load in 8-bit or 4-bit (QLoRA-style) through the
HIP checkout's own config, at some quality cost. The paper used QLoRA for the 70B
tier, so the pattern is supported upstream; that path is out of scope for this
runbook.

## Run it

```
just clone-hip            # once, if the sibling checkout is absent
export OPENAI_API_KEY=... # the JMQ judge
just fetch-adapter 8B     # downloads to adapters/Qwen3-8B-Base-HIP-adapter
just run-cascade 8B       # or: just run-8b
```

`run-cascade` is seeded (`--seed 42`), so a fresh box rebuilds the same 50-pair
FineWeb corpus and the same drafts the Mac produced, then rewrites them with the
8B adapter. Swap `8B` for `14B` to run the larger tier. The two are independent;
run whichever the hardware fits.

## Output

Each run writes three reports under `results/reports/`, tagged by size:

- `<size>-draft.json` -- the drafts scored against the human references.
- `<size>-rewrite-k2.json` -- the two-round rewrites scored the same way.
- `<size>-comparison.json` -- the side-by-side plus the published-benchmark row.

Read them the way smoke-run-findings.md reads the 0.6B and 4B: MMD and token-L2
should fall toward human, JMQ is the honest quality check, and Pangram is the
external detector. The question the larger tiers answer is whether the JMQ penalty
that was -0.39 at 0.6B and -0.11 at 4B crosses into quality-neutral or positive.
If it does, HIP stops being only a humanizer and starts being a rewriter. If it
does not, the scaling curve is flatter than the trend suggested, which is itself
the result.

## After the run

Add the numbers to smoke-run-findings.md (a new size column in the comparison
tables), re-render the PDF with `just render-findings`, and if the noise bounds
were re-derived over multiple seeds, tighten `REAL_INSTRUMENT_BOUNDS` in
src/dehip/metrics/bounds.py with the multi-seed range in place of the current
single-run margins.
