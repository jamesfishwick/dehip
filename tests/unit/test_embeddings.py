"""Unit tests for the embedding cache using a STUB embedder.

No test loads or downloads the real `nvidia/llama-embed-nemotron-8b` model. The
stub embedder produces deterministic vectors from the text and counts how many
texts it was asked to embed, which lets us assert cache hits skip recompute.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from dehip.metrics.embeddings import (
    CacheIntegrityError,
    EmbeddingCache,
    content_sha256,
)


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
        return rng.standard_normal(self.dim).astype(np.float32)


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
    # counter starts at zero. The requested text must come from disk, not the
    # stub. The stub is invoked exactly once, for the one-time live-dim probe
    # that guards against a drifted model serving stale-width hits (an accepted
    # correctness cost on the warm path); the probe text is NOT "persist me".
    stub2 = StubEmbedder()
    cache2 = EmbeddingCache(stub2, cache_dir=cache_dir)
    reloaded = cache2.embed_one("persist me")
    assert stub2.calls == 1  # the dim probe only; "persist me" was a cache hit
    np.testing.assert_array_equal(original, reloaded)

    # A second serve on the same instance does NOT re-probe: the live dim is
    # verified once per instance, so this hit adds zero further calls.
    again = cache2.embed_one("persist me")
    assert stub2.calls == 1
    np.testing.assert_array_equal(original, again)


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

    # Embedder A's entry still resolves from cache without recomputing the
    # requested text. The one call is the one-time live-dim probe on a warm
    # reopen (see test_cache_persists_across_fresh_instantiation), not a recompute
    # of "shared text", which is served from disk.
    stub_a2 = StubEmbedder(embedder_id="embedder-A")
    cache_a2 = EmbeddingCache(stub_a2, cache_dir=cache_dir)
    np.testing.assert_array_equal(vec_a, cache_a2.embed_one("shared text"))
    assert stub_a2.calls == 1  # dim probe only


# --- Integrity / hardening tests (Korgull findings) -----------------------


def test_forged_index_mapping_raises_on_read(tmp_path):
    """Critical 1: a key that resolves to a mismatched store row must raise.

    The index maps a key to a store ref; a corrupt/forged index (or a store
    swapped under a stale index) can point a key at a row whose recorded
    provenance is a different sha. The read path must re-check the row's
    provenance against the requested key and raise, not hand back the wrong
    vector. Here we drive that guard directly: the on-disk row for ref 0 records
    "beta"'s sha, yet a lookup for "alpha" resolves to ref 0.
    """
    cache_dir = tmp_path / "emb-cache"
    stub = StubEmbedder()
    cache = EmbeddingCache(stub, cache_dir=cache_dir)
    cache.embed(["alpha", "beta"])  # two rows

    sha_alpha = content_sha256("alpha")
    sha_beta = content_sha256("beta")

    reopened = EmbeddingCache(StubEmbedder(), cache_dir=cache_dir)
    # Simulate a stale/forged index whose key points at a row owned by another
    # sha: row (ref) for beta is where alpha's key now resolves.
    ref_beta = reopened._index[(stub.embedder_id, sha_beta)]
    reopened._index[(stub.embedder_id, sha_alpha)] = ref_beta

    with pytest.raises(CacheIntegrityError):
        reopened.get("alpha")


def test_changed_output_dim_same_embedder_id_raises(tmp_path):
    """Critical 2: model/dim drift under a stable embedder_id must raise."""
    cache_dir = tmp_path / "emb-cache"
    cache = EmbeddingCache(StubEmbedder(dim=8), cache_dir=cache_dir)
    cache.embed_one("drifting text")

    # Same embedder_id, different output dim -> stale/mixed spaces.
    drifted = StubEmbedder(embedder_id="stub-embedder-v1", dim=16)
    reopened = EmbeddingCache(drifted, cache_dir=cache_dir)
    with pytest.raises(CacheIntegrityError):
        reopened.embed_one("new text")


def test_all_hits_drift_raises_before_serving_stale_vectors(tmp_path):
    """Re-review IMPORTANT: dim drift must be caught even on an ALL-HITS path.

    ``_check_live_dim`` used to run only after misses were computed, so a warm
    cache whose requested texts are all hits, under a model that changed output
    dim but kept the embedder_id, would serve OLD-width vectors with ZERO embed
    calls and no raise. This is the same stale-vector failure as the miss-path
    test, on the most common path (re-scoring an unchanged corpus after a model
    swap). Build a dim-8 cache, reopen with a dim-16 stub under the same id, and
    request ONLY already-cached text: it must RAISE, not return a dim-8 vector.
    """
    cache_dir = tmp_path / "emb-cache"
    warm = EmbeddingCache(StubEmbedder(dim=8), cache_dir=cache_dir)
    warm.embed_one("cached text")  # populate a dim-8 entry

    drifted = StubEmbedder(embedder_id="stub-embedder-v1", dim=16)
    reopened = EmbeddingCache(drifted, cache_dir=cache_dir)

    with pytest.raises(CacheIntegrityError):
        # ONLY an already-cached text: an all-hits request, no miss to trigger
        # the old miss-path check.
        reopened.embed_one("cached text")

    # get() is the other serve path and must guard identically.
    reopened_get = EmbeddingCache(
        StubEmbedder(embedder_id="stub-embedder-v1", dim=16), cache_dir=cache_dir
    )
    with pytest.raises(CacheIntegrityError):
        reopened_get.get("cached text")


def test_desynced_store_fewer_rows_than_index_raises(tmp_path):
    """Critical 4: index refs beyond the store's row count must raise on load."""
    cache_dir = tmp_path / "emb-cache"
    stub = StubEmbedder()
    cache = EmbeddingCache(stub, cache_dir=cache_dir)
    cache.embed(["one", "two", "three"])  # three store rows

    # Truncate the store to a single row while the index still names three.
    stored = np.load(cache_dir / "vectors.npy", allow_pickle=False)
    np.save(cache_dir / "vectors.npy", stored[:1], allow_pickle=False)

    with pytest.raises(CacheIntegrityError):
        EmbeddingCache(StubEmbedder(), cache_dir=cache_dir)


def test_appending_changed_dim_raises_clear_message(tmp_path):
    """Important 5: a changed-dim append under one embedder_id raises clearly."""
    cache_dir = tmp_path / "emb-cache"
    cache = EmbeddingCache(StubEmbedder(dim=8), cache_dir=cache_dir)
    cache.embed_one("first")

    # Reopen and append a vector of a different width under the same id.
    wider = StubEmbedder(embedder_id="stub-embedder-v1", dim=16)
    reopened = EmbeddingCache(wider, cache_dir=cache_dir)
    with pytest.raises(CacheIntegrityError, match="dim"):
        reopened.embed_one("second")


def test_mixed_dim_embedders_one_cache_dir_raises_clear_message(tmp_path):
    """Important: two embedder_ids of different dims in one cache_dir must raise.

    The cache key space admits multiple embedder_ids per cache_dir, but the
    vector store is a single dense array (``np.stack`` in ``_flush``), so it can
    only hold one output dimension. Two DIFFERENT embedder_ids with different
    output dims sharing one dir used to crash on the second's first flush with a
    raw ``ValueError: all input arrays must have the same shape`` from
    ``np.stack``. It must instead raise a ``CacheIntegrityError`` naming both
    dims and pointing at the fix (separate cache_dir per dim).
    """
    cache_dir = tmp_path / "emb-cache"

    emb_a = StubEmbedder(embedder_id="embedder-A", dim=4)
    EmbeddingCache(emb_a, cache_dir=cache_dir).embed_one("a")  # one dim-4 row

    # A distinct embedder_id emitting a different dim, over the same dir. The key
    # is new (so same-id drift checks pass), but the store cannot hold both
    # widths. Expect a clear CacheIntegrityError, never a raw np.stack ValueError.
    emb_b = StubEmbedder(embedder_id="embedder-B", dim=8)
    cache_b = EmbeddingCache(emb_b, cache_dir=cache_dir)
    with pytest.raises(CacheIntegrityError) as exc:
        cache_b.embed_one("b")

    message = str(exc.value)
    assert "4" in message and "8" in message  # both dims named
    assert str(cache_dir) in message  # names the offending cache_dir


def test_corrupt_index_parquet_raises_naming_path(tmp_path):
    """Important 7: a corrupt index.parquet raises a domain error, not a miss."""
    cache_dir = tmp_path / "emb-cache"
    cache = EmbeddingCache(StubEmbedder(), cache_dir=cache_dir)
    cache.embed_one("real content")

    # Clobber the parquet with garbage bytes.
    (cache_dir / "index.parquet").write_bytes(b"not a parquet file at all")

    with pytest.raises(CacheIntegrityError) as exc:
        EmbeddingCache(StubEmbedder(), cache_dir=cache_dir)
    assert str(cache_dir) in str(exc.value)


def test_embed_fn_without_embedder_id_raises(tmp_path):
    """Nit 8: embed_fn must carry an explicit embedder_id."""

    class NoIdEmbedder:
        def __call__(self, texts):
            return np.zeros((len(list(texts)), 8), dtype=np.float32)

    with pytest.raises(ValueError, match="embedder_id"):
        EmbeddingCache(NoIdEmbedder(), cache_dir=tmp_path / "emb-cache")


# --- cache-dir env override (test isolation, issue #16 follow-up) --------------


def test_cache_dir_resolves_from_env(tmp_path, monkeypatch):
    from dehip.metrics.embeddings import CACHE_DIR_ENV

    custom = tmp_path / "from-env"
    monkeypatch.setenv(CACHE_DIR_ENV, str(custom))
    cache = EmbeddingCache(StubEmbedder())  # no explicit cache_dir -> reads env
    assert cache._cache_dir == custom


def test_explicit_cache_dir_beats_env(tmp_path, monkeypatch):
    from dehip.metrics.embeddings import CACHE_DIR_ENV

    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "env"))
    cache = EmbeddingCache(StubEmbedder(), cache_dir=tmp_path / "explicit")
    assert cache._cache_dir == tmp_path / "explicit"
