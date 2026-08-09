# Security and trust model

dehip runs third-party model code and calls paid APIs. This is the short account
of what it trusts, what it does to bound that trust, and where the residual risk
sits. It exists because the analysis pass flagged `trust_remote_code` as an
undocumented assumption; now it is documented.

## What executes third-party code

Two places load model code from Hugging Face:

- **The embedder** (`metrics/embeddings.py`) loads `nvidia/llama-embed-nemotron-8b`
  with `trust_remote_code=True`. That model is a custom `llama_bidirec`
  architecture that ships its own Python, and it will not load without the flag.
  The code runs in-process at load time.
- **The HIP base model** is loaded by `hip-run`, not by dehip. The cascade
  (`cascade.py`) sets `trust_remote_code` in the hip-run config; hip-run infers
  the base from the adapter's PeftConfig and does the load in its own subprocess.

The instruct draft model (`generate.py`, Qwen) is a standard architecture and
loads without `trust_remote_code`.

## How the trust is bounded

- **Pinned revisions.** The embedder and the default draft model load at a fixed
  commit SHA (`DEFAULT_EMBEDDER_REVISION`, `DEFAULT_MODEL_REVISION`), so a
  compromised or updated upstream cannot ship new code into a run bound to that
  SHA. The pins are the exact revisions that produced the issue-#16 smoke run, so
  they also make the metrics reproducible. Bump them deliberately, never silently.
- **No shell.** The only subprocess is `subprocess.run(["uv", "run", "hip-run",
  "--config", ...])` with list args, no `shell=True`, and a timeout. The untrusted
  text being paraphrased travels in a JSONL file, never on the command line, so it
  cannot be interpreted as a command.
- **No pickle in the cache.** The embedding cache uses `np.load`/`np.save` with
  `allow_pickle=False`, so a tampered cache file cannot execute code. The cache
  also refuses to mix vectors of different dimensions or embedder ids.
- **Safe YAML.** The hip-run config is written with `yaml.safe_dump`.

## Secrets

API keys (`OPENAI_API_KEY`, `PANGRAM_API_KEY`, `GPTZERO_API_KEY`) are read only
from the environment, never hardcoded, and their presence is enforced before any
client is constructed, so a missing key fails fast rather than causing partial
paid spend. Keys are never logged; the code references only the env-var names.
`.env` is gitignored.

## Residual risk

- **The HIP base model revision is not pinned by dehip.** hip-run resolves it from
  the adapter, so the pin belongs at the fetch step: `just fetch-adapter <size>`
  downloads a specific adapter, and you can pass a `--revision` to `hf download`
  if you want to pin the adapter (and therefore the base) to a known commit. Until
  then, that model tracks upstream.
- **`trust_remote_code` still trusts the pinned code.** Pinning stops new code from
  arriving; it does not audit the code at the pinned commit. The sources here are
  NVIDIA and the HIP authors, which is the trust decision being made explicitly.

## Bumping a pin

Change the SHA constant, run the affected stage on real models, and re-derive the
noise bounds if the embedder moved (the bounds in `metrics/bounds.py` are tied to
the embedder revision). Record the new SHA the same way the current ones were: the
commit that produced the run whose numbers you trust.
