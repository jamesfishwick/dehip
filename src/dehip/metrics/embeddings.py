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
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_CACHE_DIR = Path("data/emb-cache")
DEFAULT_EMBEDDER_ID = "nvidia/llama-embed-nemotron-8b"


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
        self._tokenizer = AutoTokenizer.from_pretrained(self.embedder_id)
        self._model = AutoModel.from_pretrained(self.embedder_id, torch_dtype=dtype)
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
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ) -> None:
        self._embed_fn = embed_fn
        self._embedder_id = getattr(embed_fn, "embedder_id", DEFAULT_EMBEDDER_ID)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "index.parquet"
        self._store_path = self._cache_dir / "vectors.npy"

        # In-memory mirror of the on-disk state: (embedder_id, sha) -> vector_ref.
        self._index: dict[tuple[str, str], int] = {}
        self._vectors: list[np.ndarray] = []
        self._load()

    def _load(self) -> None:
        if not self._index_path.exists() or not self._store_path.exists():
            return
        table = pq.read_table(self._index_path)
        stored = np.load(self._store_path, allow_pickle=False)
        self._vectors = [row.copy() for row in stored]
        for sha, emb_id, ref in zip(
            table.column("content_sha256").to_pylist(),
            table.column("embedder_id").to_pylist(),
            table.column("vector_ref").to_pylist(),
            strict=True,
        ):
            self._index[(emb_id, sha)] = ref

    def _flush(self) -> None:
        store = (
            np.stack(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, 0), dtype=np.float32)
        )
        np.save(self._store_path, store, allow_pickle=False)

        keys = list(self._index.items())
        table = pa.table(
            {
                "content_sha256": pa.array([sha for (_, sha), _ in keys], pa.string()),
                "embedder_id": pa.array([emb for (emb, _), _ in keys], pa.string()),
                "vector_ref": pa.array([ref for _, ref in keys], pa.int64()),
            },
            schema=self._INDEX_SCHEMA,
        )
        pq.write_table(table, self._index_path)

    def get(self, text: str) -> np.ndarray | None:
        """Return the cached vector for ``text`` under this embedder, or None."""
        ref = self._index.get((self._embedder_id, content_sha256(text)))
        if ref is None:
            return None
        return self._vectors[ref].copy()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for ``texts``, computing only cache misses.

        Cached texts are served from the store and never re-sent to ``embed_fn``.
        Misses are batched into a single ``embed_fn`` call, appended to the cache,
        and flushed to disk. The returned array preserves input order; identical
        inputs map to the same cached vector.
        """
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

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
            for sha, vector in zip(miss_shas, computed, strict=True):
                ref = len(self._vectors)
                self._vectors.append(np.asarray(vector, dtype=np.float32))
                self._index[(self._embedder_id, sha)] = ref
            self._flush()

        return np.stack(
            [self._vectors[self._index[(self._embedder_id, sha)]] for sha in shas],
            axis=0,
        )

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper: embed a single text, returning a 1-D vector."""
        return self.embed([text])[0]
