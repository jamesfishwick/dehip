"""Empirically derive the stub-instrument self-check noise bounds.

Runs the REAL self-check code path (dehip.self_check.run_self_check, which calls
dehip.report.score) over a 50-pair stub smoke corpus across several differently
seeded half-splits, using the SAME stub embedder / stub tokenizer the integration
tests use. Prints the observed MMD^2 and token-L2 distances per seed and the max
across seeds, which is what the documented bounds are set just above.

This is a derivation/audit tool, not part of the shipped package or the test
suite. Re-run it to reproduce the numbers baked into dehip/metrics/bounds.py.

    uv run python scripts/derive_bounds.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from dehip.metrics.embeddings import EmbeddingCache
from dehip.schemas import TextSet, write_json
from dehip.self_check import StubInstrumentBounds, run_self_check

# Seeds used for the derivation. Documented in bounds.py so a reader re-derives
# from the same set.
DERIVATION_SEEDS = tuple(range(24))
CORPUS_ID = "stub-smoke-50"
CORPUS_SIZE = 50


class StubEmbedder:
    """Hash-seeded 4-D standard-normal embedder (identical to the test stub)."""

    embedder_id = "stub-embedder"

    def __call__(self, texts):
        rows = []
        for text in texts:
            seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\x00"), "big")
            rng = np.random.default_rng(seed)
            rows.append(rng.standard_normal(4).astype(np.float32))
        return np.stack(rows, axis=0).astype(np.float32)


class StubTokenizer:
    tokenizer_id = "stub-tokenizer"

    def tokenize(self, text: str):
        return text.split()


def _build_stub_smoke_corpus(root: Path) -> str:
    """Write a 50-pair human_reference manifest + texts JSONL of varied prose.

    The texts imitate the smoke corpus's shape: multi-sentence human-like prose of
    varying length and vocabulary. They are fixed strings (deterministic), so the
    derived bounds reproduce exactly.
    """
    # Same generator as tests/integration/test_self_check.py::_write_reference, so
    # the documented bounds are derived over exactly the corpus shape the SC-001
    # verification test exercises (12 subjects, 6 verbs, (i%9)+4 extra words).
    subjects = [
        "the river", "a small town", "the old library", "her grandmother",
        "the market", "the mountain trail", "the winter storm", "the harbor",
        "an empty theater", "the garden", "the night sky", "the train station",
    ]
    verbs = [
        "changed slowly over the years", "held a strange quiet",
        "drew people from far away", "kept its secrets well",
        "smelled of rain and dust", "seemed larger at dusk",
    ]
    pairs = []
    for i in range(CORPUS_SIZE):
        subj = subjects[i % len(subjects)]
        verb = verbs[i % len(verbs)]
        extra = " ".join(f"word{(i * 7 + j) % 40}" for j in range((i % 9) + 4))
        text = (
            f"{subj.capitalize()} {verb}. It was a place many remembered long "
            f"after they left. {extra}. Nothing about it was ordinary."
        )
        pairs.append((f"{CORPUS_ID}-{i}", text))

    manifest = TextSet(
        set_id=CORPUS_ID,
        role="human_reference",
        corpus="fineweb",
        pair_ids=[pid for pid, _ in pairs],
        provenance={"texts_path": f"{CORPUS_ID}.jsonl", "count": len(pairs)},
    )
    manifest_path = root / f"{CORPUS_ID}.manifest.json"
    write_json(manifest, manifest_path)
    with (root / f"{CORPUS_ID}.jsonl").open("w", encoding="utf-8") as fh:
        for pid, text in pairs:
            fh.write(json.dumps({"pair_id": pid, "text": text}) + "\n")
    return str(manifest_path)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = _build_stub_smoke_corpus(root)

        # Wide-open bounds so the derivation never raises; we read the values.
        open_bounds = StubInstrumentBounds(
            mmd_max=float("inf"),
            token_l2_max=float("inf"),
            jmq_win_rate_min=0.0,
            jmq_win_rate_max=1.0,
        )

        mmds = []
        token_l2s = []
        for seed in DERIVATION_SEEDS:
            cache = EmbeddingCache(StubEmbedder(), cache_dir=root / f"emb-{seed}")
            result = run_self_check(
                manifest,
                seed=seed,
                skip_jmq=True,
                embed_cache=cache,
                tokenizer=StubTokenizer(),
                bounds=open_bounds,
                embedder_id="stub-embedder",
            )
            mmds.append(result.mmd)
            token_l2s.append(result.token_l2)
            print(
                f"seed={seed:>2}  half_size={result.half_size:>2}  "
                f"dropped={result.dropped_pair_id}  "
                f"mmd={result.mmd:+.8f}  token_l2={result.token_l2:.8f}"
            )

        print("\n--- summary over stub smoke corpus ---")
        print(f"corpus_id       = {CORPUS_ID} ({CORPUS_SIZE} pairs)")
        print(f"seeds           = {list(DERIVATION_SEEDS)}")
        print(f"mmd  min/max    = {min(mmds):+.8f} / {max(mmds):+.8f}")
        print(f"mmd  abs-max    = {max(abs(m) for m in mmds):.8f}")
        print(f"tokL2 min/max   = {min(token_l2s):.8f} / {max(token_l2s):.8f}")


if __name__ == "__main__":
    main()
