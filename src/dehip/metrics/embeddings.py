"""Protocol embedder and content-hash embedding cache (R4).

The embedder is the exact-protocol instrument: `nvidia/llama-embed-nemotron-8b`
loaded via transformers with MPS/CUDA autodetect and fp16. Loading the real model
is expensive, so the runtime is placed behind an injectable seam (the ``Embedder``
protocol and the ``embed_fn`` argument to ``EmbeddingCache``). Tests supply a stub
that counts its own calls and never touches the network.

The cache (see data-model.md, EmbeddingCacheEntry) is append-only and keyed by
``(embedder_id, content SHA-256)``. It stores a parquet index plus an npy vector
store under ``data/emb-cache/``. The same text under a different ``embedder_id`` is
a distinct entry. Existing entries are never rewritten and there is no eviction.

Integrity model
---------------
The cache is defensive against silent corruption and drift. Every stored vector
carries its own ``content_sha256`` and ``embedder_id`` in the index, aligned
row-for-row with the vector store, and each read re-checks that the row it is
about to return actually belongs to the requested ``(embedder_id, sha)``. A
per-``embedder_id`` output dimension is recorded so that a model change hiding
behind an unchanged ``embedder_id`` is caught rather than served as stale
vectors. Writes are atomic (temp file + ``os.replace``, store before index) so a
crash mid-flush can never leave a visible half-written cache. Any detected
corruption or desync raises ``CacheIntegrityError`` naming the ``cache_dir`` and
the recovery step (clear that directory), rather than degrading into an opaque
``IndexError`` or silently re-embedding over the damage.

The vector store is one dense array, so each ``cache_dir`` holds a single output
dimension across every ``embedder_id`` sharing it. Two embedders with different
output dims cannot share a dir; an append of a mismatched width raises
``CacheIntegrityError`` (naming both dims) instead of a raw ``np.stack``
``ValueError``. Use a separate ``cache_dir`` per output dimension.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_CACHE_DIR = Path("data/emb-cache")
# Env override for the cache dir, resolved at EmbeddingCache construction time.
# Lets tests isolate the cache to a tmp dir (see tests/conftest.py) and lets a
# real run relocate it off the shared default.
CACHE_DIR_ENV = "DEHIP_EMB_CACHE_DIR"
DEFAULT_EMBEDDER_ID = "nvidia/llama-embed-nemotron-8b"

# Vectors are stored deliberately as float32: it is the precision the Embedder
# protocol promises, it halves the on-disk footprint versus float64, and it is
# lossless for the fp16/fp32 model outputs the production embedder produces. The
# cache asserts this on append rather than silently downcasting a wider dtype,
# so a protocol violation surfaces loudly instead of corrupting recall math.
_STORE_DTYPE = np.float32


class CacheIntegrityError(RuntimeError):
    """Raised when the on-disk embedding cache is corrupt, desynced, or drifted.

    The message always names the offending ``cache_dir`` and the recovery step
    (clear that directory to rebuild the cache from scratch), because every one
    of these conditions means the persisted state can no longer be trusted.
    """


@runtime_checkable
class Embedder(Protocol):
    """A batched text -> vector function.

    Implementations return a 2-D float32 array of shape ``(len(texts), dim)``.
    The rows must line up with the input order.
    """

    embedder_id: str

    def __call__(self, texts: Sequence[str]) -> np.ndarray: ...


def content_sha256(text: str) -> str:
    """SHA-256 of the UTF-8 bytes of ``text``, as a hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TransformersEmbedder:
    """Loads `nvidia/llama-embed-nemotron-8b` and embeds batches on MPS/CUDA/CPU.

    This is the production seam. It is intentionally lazy: the model and tokenizer
    load on first call so that constructing the object (and importing this module)
    never pulls the 8B weights. Tests inject a stub instead of this class.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDER_ID,
        *,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.embedder_id = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        self._device = device
        # The R4-pinned embedder (nvidia/llama-embed-nemotron-8b) is a custom
        # `llama_bidirec` architecture that ships its own modeling code, so it
        # loads only with trust_remote_code=True. Without it, from_pretrained
        # prompts for interactive approval and EOFs in a non-interactive shell.
        # This is the exact-protocol instrument from a trusted source (NVIDIA).
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.embedder_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.embedder_id, torch_dtype=dtype, trust_remote_code=True
        )
        self._model.to(device)
        self._model.eval()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_loaded()
        import torch

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                output = self._model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.float().cpu().numpy())

        return np.concatenate(vectors, axis=0).astype(np.float32)


def _mean_pool(last_hidden_state, attention_mask):
    """Attention-masked mean pooling over token embeddings."""
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class EmbeddingCache:
    """Append-only, content-hash embedding cache keyed by (embedder_id, sha256).

    Vectors are appended to an npy store; a parquet index records
    ``(content_sha256, embedder_id, vector_ref)`` where ``vector_ref`` is the row
    offset into the store. Existing entries are never rewritten. The index and
    store live under ``cache_dir`` (default ``data/emb-cache/``) so the cache
    survives across process restarts and fresh ``EmbeddingCache`` instances.

    The index doubles as a per-row provenance record: ``content_sha256`` and
    ``embedder_id`` are stored aligned with each store row, so every read can
    confirm the row it returns belongs to the requested key, and the loader can
    detect a store/index desync before it becomes an opaque ``IndexError``.

    One output dimension per cache_dir
    ----------------------------------
    The vector store is a single dense array (``np.stack`` in ``_flush``), so a
    given ``cache_dir`` holds exactly one output dimension across every
    ``embedder_id`` that shares it. Multiple ``embedder_id``s may coexist in one
    dir only if they emit the SAME dim; two embedders with different output dims
    cannot. An append whose width differs from the store's existing rows raises
    ``CacheIntegrityError`` (see ``_check_store_dim``) naming both dims and the
    fix (a separate ``cache_dir`` per output dimension, or clear this one) rather
    than dying with a raw ``ValueError`` from ``np.stack``.
    """

    _INDEX_SCHEMA = pa.schema(
        [
            ("content_sha256", pa.string()),
            ("embedder_id", pa.string()),
            ("vector_ref", pa.int64()),
        ]
    )

    def __init__(
        self,
        embed_fn: Embedder,
        *,
        cache_dir: Path | str | None = None,
    ) -> None:
        embedder_id = getattr(embed_fn, "embedder_id", None)
        if not embedder_id:
            raise ValueError(
                "embed_fn must carry a non-empty 'embedder_id' attribute; the "
                "cache key and per-embedder dim fingerprint depend on it. Set "
                "embed_fn.embedder_id explicitly (e.g. the model name)."
            )
        self._embed_fn = embed_fn
        self._embedder_id = embedder_id
        # When no cache_dir is passed, resolve the env override at construction
        # time (not import time) so a per-test fixture can point every
        # CLI-constructed cache at a tmp dir. The real store lives under
        # data/emb-cache by default; DEHIP_EMB_CACHE_DIR relocates it (also lets
        # a real run and the test suite avoid colliding on the shared default).
        if cache_dir is None:
            cache_dir = os.environ.get(CACHE_DIR_ENV, str(DEFAULT_CACHE_DIR))
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "index.parquet"
        self._store_path = self._cache_dir / "vectors.npy"

        # In-memory mirror of the on-disk state. ``_index`` maps
        # (embedder_id, sha) -> vector_ref; ``_rows`` records the provenance
        # (embedder_id, sha) for each store row so a read can re-verify it, and
        # ``_dims`` records the output dim seen per embedder_id for drift checks.
        self._index: dict[tuple[str, str], int] = {}
        self._vectors: list[np.ndarray] = []
        self._rows: list[tuple[str, str]] = []
        self._dims: dict[str, int] = {}
        # Whether we have verified the live embedder's output dim against the
        # cached dim for this session. See ``_verify_live_dim_once``.
        self._live_dim_verified = False
        self._load()

    @property
    def embedder_id(self) -> str:
        """The wrapped embedder's id (its cache-key namespace).

        Exposed so a caller (e.g. the self-check) can record the embedder that
        actually ran instead of a config default, keeping provenance honest.
        """
        return self._embedder_id

    def _integrity_error(self, detail: str) -> CacheIntegrityError:
        return CacheIntegrityError(
            f"{detail} (cache_dir={self._cache_dir}). Recovery: clear this "
            f"directory to rebuild the cache from scratch."
        )

    def _load(self) -> None:
        if not self._index_path.exists() or not self._store_path.exists():
            return

        try:
            table = pq.read_table(self._index_path)
        except Exception as exc:  # noqa: BLE001 - normalize to a domain error
            # A parse failure means a genuinely corrupt index. Do NOT treat this
            # as a cache miss: silently re-embedding would write fresh entries on
            # top of the corruption and mask it. Fail loudly instead.
            raise self._integrity_error(
                f"index.parquet is unreadable ({type(exc).__name__}: {exc})"
            ) from exc

        try:
            stored = np.load(self._store_path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - normalize to a domain error
            raise self._integrity_error(
                f"vectors.npy is unreadable ({type(exc).__name__}: {exc})"
            ) from exc

        shas = table.column("content_sha256").to_pylist()
        emb_ids = table.column("embedder_id").to_pylist()
        refs = table.column("vector_ref").to_pylist()

        n_rows = len(stored)
        if len(shas) != n_rows:
            raise self._integrity_error(
                f"index/store desync: index has {len(shas)} entries but the "
                f"vector store has {n_rows} rows"
            )

        # Rebuild the row-aligned provenance and the key -> ref map, validating
        # that every ref is in range and points at the row the index claims.
        rows: list[tuple[str, str] | None] = [None] * n_rows
        for sha, emb_id, ref in zip(shas, emb_ids, refs, strict=True):
            if not isinstance(ref, int) or ref < 0 or ref >= n_rows:
                raise self._integrity_error(
                    f"index/store desync: vector_ref {ref!r} is out of range for "
                    f"a store of {n_rows} rows"
                )
            if rows[ref] is not None:
                raise self._integrity_error(
                    f"index corruption: vector_ref {ref} is claimed by two "
                    f"entries"
                )
            rows[ref] = (emb_id, sha)
            self._index[(emb_id, sha)] = ref

        if any(r is None for r in rows):
            raise self._integrity_error(
                "index/store desync: some store rows have no index entry"
            )

        self._vectors = [row.copy() for row in stored]
        self._rows = [r for r in rows if r is not None]

        # Record and cross-check the per-embedder output dim. A store row's width
        # is the embedder's output dim; two rows for one embedder_id must match.
        for (emb_id, _sha), vec in zip(self._rows, self._vectors, strict=True):
            dim = int(vec.shape[0])
            existing = self._dims.get(emb_id)
            if existing is None:
                self._dims[emb_id] = dim
            elif existing != dim:
                raise self._integrity_error(
                    f"cached vectors for embedder_id={emb_id!r} disagree on dim "
                    f"({existing} vs {dim})"
                )

    def _check_live_dim(self, dim: int) -> None:
        """Assert the live embedder's output dim matches the cached one.

        Catches a model/config change hiding behind an unchanged ``embedder_id``:
        the cache holds vectors of the old width, the live model now emits a
        different width, and serving the cached rows would mix incompatible
        spaces. Raises loudly so the caller bumps the embedder_id or clears the
        cache instead of silently getting stale vectors.
        """
        cached = self._dims.get(self._embedder_id)
        if cached is not None and cached != dim:
            raise self._integrity_error(
                f"output-dim drift for embedder_id={self._embedder_id!r}: cached "
                f"vectors are dim {cached} but the live embedder emitted dim "
                f"{dim}. Bump the embedder_id or clear the cache"
            )

    def _check_store_dim(self, dim: int) -> None:
        """Reject a vector whose width differs from any row already in the store.

        The vector store is a single dense array (``np.stack`` in ``_flush``), so
        it can only hold one output dimension for the whole ``cache_dir``, across
        all ``embedder_id``s that share it. That is narrower than the key space,
        which admits multiple ``embedder_id``s: two embedders with DIFFERENT
        output dims cannot coexist in one store. Without this guard the second
        embedder's first append would reach ``np.stack`` on a ragged list and die
        with a raw ``ValueError: all input arrays must have the same shape``.

        Catch it here, before any in-memory mutation, and raise a
        ``CacheIntegrityError`` naming both dims and the fix. ``_check_live_dim``
        only covers same-``embedder_id`` drift; this covers the cross-embedder
        case an unchanged ``embedder_id`` never triggers.
        """
        for other_id, other_dim in self._dims.items():
            if other_dim != dim:
                raise self._integrity_error(
                    f"cannot store dim-{dim} vectors for "
                    f"embedder_id={self._embedder_id!r} in a cache_dir that "
                    f"already holds dim-{other_dim} vectors for "
                    f"embedder_id={other_id!r}; the vector store holds one "
                    f"output dimension per cache_dir. Use a separate cache_dir "
                    f"per embedder output dimension, or clear this one"
                )

    def _verify_live_dim_once(self) -> None:
        """Probe the live embedder's output dim and compare it to the cache.

        Runs at most once per instance, on the first serve of any kind (``get``,
        ``embed``, ``embed_one``). It closes the ALL-HITS drift hole: if the model
        behind an unchanged ``embedder_id`` changes its output width, a warm cache
        whose requested texts are all hits would otherwise serve OLD-width vectors
        with zero embed calls and no check. So when the cache already records a dim
        for this embedder_id, we embed one probe text purely to learn the live
        width and hand it to ``_check_live_dim``, which raises on mismatch. The one
        extra embed call on the warm path is an accepted correctness cost.

        Limitation: this catches only *dimension* drift. A same-dimension model
        swap (e.g. a fine-tune that keeps the output width) is undetectable without
        a model-provided identity fingerprint the Embedder protocol does not carry;
        that is a known gap, not something this probe attempts to solve.
        """
        if self._live_dim_verified:
            return
        # Only worth probing when there is a cached dim to compare against. With no
        # cached dim, the first real embed establishes it via the miss path.
        cached = self._dims.get(self._embedder_id)
        if cached is None:
            self._live_dim_verified = True
            return
        probe = self._embed_fn(["\x00dehip-dim-probe\x00"])
        if probe.ndim != 2 or probe.shape[0] < 1:
            raise ValueError(
                f"embed_fn must return a 2-D array with at least one row for the "
                f"dim probe; got shape {probe.shape!r}"
            )
        self._check_live_dim(int(probe.shape[1]))
        self._live_dim_verified = True

    def _flush(self) -> None:
        """Atomically persist the store and index (store first, then index).

        Each artifact is written to a uniquely-named temp file in ``cache_dir``
        and ``os.replace``'d into place, which is atomic on the same filesystem.
        The store is replaced before the index, so the index a reader sees never
        references a store row that is not yet on disk (no half-written orphan
        ref). A crash *between* the two replaces leaves the store advanced but the
        index stale, an inconsistent on-disk pair; the guarantee is not that this
        window is impossible but that the loader detects the row-count desync and
        raises ``CacheIntegrityError`` on the next reopen rather than serving a
        mismatched cache.

        ``np.stack`` here requires every row to share a width, which is why the
        store holds one output dimension per ``cache_dir``. That invariant is
        enforced upstream on append by ``_check_store_dim``, so by the time a
        flush runs, ``self._vectors`` is already known to be uniform-width and
        the stack cannot raise a ragged-shape ``ValueError``.
        """
        store = (
            np.stack(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, 0), dtype=_STORE_DTYPE)
        )
        self._atomic_write_npy(self._store_path, store)

        keys = list(self._index.items())
        table = pa.table(
            {
                "content_sha256": pa.array([sha for (_, sha), _ in keys], pa.string()),
                "embedder_id": pa.array([emb for (emb, _), _ in keys], pa.string()),
                "vector_ref": pa.array([ref for _, ref in keys], pa.int64()),
            },
            schema=self._INDEX_SCHEMA,
        )
        self._atomic_write_parquet(self._index_path, table)

    def _atomic_write_npy(self, path: Path, array: np.ndarray) -> None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(array)}.tmp")
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, array, allow_pickle=False)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _atomic_write_parquet(self, path: Path, table: pa.Table) -> None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(table)}.tmp")
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _vector_for(self, ref: int, sha: str) -> np.ndarray:
        """Return the store row at ``ref`` after verifying it owns ``sha``.

        The index maps a key to a ref, but a corrupt or forged index could point
        the key at the wrong row. Re-check the row's recorded provenance against
        the requested (embedder_id, sha) before trusting it.
        """
        row_emb_id, row_sha = self._rows[ref]
        if row_emb_id != self._embedder_id or row_sha != sha:
            raise self._integrity_error(
                f"index maps (embedder_id={self._embedder_id!r}, sha={sha}) to "
                f"store row {ref}, but that row holds "
                f"(embedder_id={row_emb_id!r}, sha={row_sha})"
            )
        return self._vectors[ref].copy()

    def get(self, text: str) -> np.ndarray | None:
        """Return the cached vector for ``text`` under this embedder, or None."""
        self._verify_live_dim_once()
        sha = content_sha256(text)
        ref = self._index.get((self._embedder_id, sha))
        if ref is None:
            return None
        return self._vector_for(ref, sha)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for ``texts``, computing only cache misses.

        Cached texts are served from the store and never re-sent to ``embed_fn``.
        Misses are batched into a single ``embed_fn`` call, appended to the cache,
        and flushed to disk. The returned array preserves input order; identical
        inputs map to the same cached vector.
        """
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=_STORE_DTYPE)

        # Verify the live dim before serving ANY hit, so an all-hits warm cache
        # under a drifted model raises instead of returning stale-width vectors.
        self._verify_live_dim_once()

        shas = [content_sha256(t) for t in texts]

        # Collect the distinct misses (dedup within the batch too, so a repeated
        # text in one call is embedded once).
        miss_texts: list[str] = []
        miss_shas: list[str] = []
        seen_misses: set[str] = set()
        for text, sha in zip(texts, shas, strict=True):
            key = (self._embedder_id, sha)
            if key in self._index or sha in seen_misses:
                continue
            seen_misses.add(sha)
            miss_texts.append(text)
            miss_shas.append(sha)

        if miss_texts:
            computed = self._embed_fn(miss_texts)
            if computed.shape[0] != len(miss_texts):
                raise ValueError(
                    f"embed_fn returned {computed.shape[0]} vectors "
                    f"for {len(miss_texts)} texts"
                )
            if computed.ndim != 2:
                raise ValueError(
                    f"embed_fn must return a 2-D array; got ndim={computed.ndim}"
                )
            if computed.dtype != _STORE_DTYPE:
                # Protocol says float32. Refuse a wider/narrower dtype instead of
                # silently casting and losing (or misrepresenting) precision.
                raise ValueError(
                    f"embed_fn must return {_STORE_DTYPE.__name__} vectors per the "
                    f"Embedder protocol; got {computed.dtype}"
                )

            dim = int(computed.shape[1])
            # Validate dim against the store's existing dim for this embedder_id
            # AND against any OTHER embedder_id already in this cache_dir, BEFORE
            # mutating any in-memory state, so a mismatch never leaves the cache
            # half-appended.
            self._check_live_dim(dim)
            self._check_store_dim(dim)

            for sha, vector in zip(miss_shas, computed, strict=True):
                ref = len(self._vectors)
                self._vectors.append(np.asarray(vector, dtype=_STORE_DTYPE))
                self._rows.append((self._embedder_id, sha))
                self._index[(self._embedder_id, sha)] = ref
            self._dims[self._embedder_id] = dim
            self._flush()

        return np.stack(
            [
                self._vector_for(self._index[(self._embedder_id, sha)], sha)
                for sha in shas
            ],
            axis=0,
        )

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper: embed a single text, returning a 1-D vector."""
        return self.embed([text])[0]
