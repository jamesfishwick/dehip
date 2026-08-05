"""Unit tests for the embedding cache using a STUB embedder.

No test loads or downloads the real `nvidia/llama-embed-nemotron-8b` model. The
stub embedder produces deterministic vectors from the text and counts how many
texts it was asked to embed, which lets us assert cache hits skip recompute.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from dehip.metrics.embeddings import EmbeddingCache, content_sha256


class StubEmbedder:
    """Deterministic, call-counting stand-in for the real embedder."""

    def __init__(self, embedder_id: str = "stub-embedder-v1", dim: int = 8) -> None:
        self.embedder_id = embedder_id
        self.dim = dim
        self.calls = 0  # number of texts actually embedded

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        self.calls += len(texts)
        rows = [self._vector(t) for t in texts]
        return np.stack(rows, axis=0).astype(np.float32)

    def _vector(self, text: str) -> np.ndarray:
        # Seed an RNG off (embedder_id, text) so the same input always maps to the
        # same vector, and different embedders map the same text differently.
        keyed = f"{self.embedder_id}\x00{text}".encode()
        seed = int.from_bytes(hashlib.sha256(keyed).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim)


def test_second_request_hits_cache_and_skips_recompute(tmp_path):
    stub = StubEmbedder()
    cache = EmbeddingCache(stub, cache_dir=tmp_path / "emb-cache")

    first = cache.embed_one("the same text")
    assert stub.calls == 1

    second = cache.embed_one("the same text")
    # Cache hit: the stub is NOT invoked again.
    assert stub.calls == 1
    np.testing.assert_array_equal(first, second)


def test_only_misses_are_embedded_within_a_batch(tmp_path):
    stub = StubEmbedder()
    cache = EmbeddingCache(stub, cache_dir=tmp_path / "emb-cache")

    cache.embed(["a", "b"])
    assert stub.calls == 2

    # "a" and "b" are cached; only "c" is a miss.
    cache.embed(["a", "b", "c"])
    assert stub.calls == 3


def test_repeated_text_in_one_batch_embeds_once(tmp_path):
    stub = StubEmbedder()
    cache = EmbeddingCache(stub, cache_dir=tmp_path / "emb-cache")

    out = cache.embed(["dup", "dup", "dup"])
    assert stub.calls == 1
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])


def test_cache_persists_across_fresh_instantiation(tmp_path):
    cache_dir = tmp_path / "emb-cache"

    stub1 = StubEmbedder()
    cache1 = EmbeddingCache(stub1, cache_dir=cache_dir)
    original = cache1.embed_one("persist me")
    assert stub1.calls == 1

    # A brand-new cache object over the same dir, with a fresh stub whose call
    # counter starts at zero. The read must come from disk, not the stub.
    stub2 = StubEmbedder()
    cache2 = EmbeddingCache(stub2, cache_dir=cache_dir)
    reloaded = cache2.embed_one("persist me")
    assert stub2.calls == 0
    np.testing.assert_array_equal(original, reloaded)


def test_batch_path_matches_single_text_path(tmp_path):
    texts = ["alpha", "beta", "gamma", "delta"]

    stub_single = StubEmbedder()
    single_cache = EmbeddingCache(stub_single, cache_dir=tmp_path / "single")
    singles = np.stack([single_cache.embed_one(t) for t in texts], axis=0)

    stub_batch = StubEmbedder()
    batch_cache = EmbeddingCache(stub_batch, cache_dir=tmp_path / "batch")
    batched = batch_cache.embed(texts)

    np.testing.assert_array_equal(singles, batched)


def test_same_text_different_embedder_is_distinct_entry(tmp_path):
    cache_dir = tmp_path / "emb-cache"

    stub_a = StubEmbedder(embedder_id="embedder-A")
    cache_a = EmbeddingCache(stub_a, cache_dir=cache_dir)
    vec_a = cache_a.embed_one("shared text")

    # Different embedder_id over the SAME dir and same text: a distinct entry, so
    # the second embedder is invoked rather than serving embedder A's vector.
    stub_b = StubEmbedder(embedder_id="embedder-B")
    cache_b = EmbeddingCache(stub_b, cache_dir=cache_dir)
    vec_b = cache_b.embed_one("shared text")
    assert stub_b.calls == 1

    # Same content hash, but keyed separately by embedder_id.
    assert content_sha256("shared text") == content_sha256("shared text")
    assert not np.array_equal(vec_a, vec_b)

    # Embedder A's entry still resolves from cache without recompute.
    stub_a2 = StubEmbedder(embedder_id="embedder-A")
    cache_a2 = EmbeddingCache(stub_a2, cache_dir=cache_dir)
    np.testing.assert_array_equal(vec_a, cache_a2.embed_one("shared text"))
    assert stub_a2.calls == 0
