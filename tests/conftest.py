"""Shared pytest fixtures for the dehip suite."""

from __future__ import annotations

import pytest

from dehip.metrics.embeddings import CACHE_DIR_ENV


@pytest.fixture(autouse=True)
def _isolate_emb_cache(tmp_path, monkeypatch):
    """Point every test's embedding cache at a tmp dir.

    The score and self-check CLI construct an EmbeddingCache with no explicit
    cache_dir, which otherwise defaults to the real ``data/emb-cache``. That made
    a real run's dim-4096 vectors collide with the stub tests' dim-4 vectors (and
    the reverse), so the suite could not run after a real scoring pass without
    clearing the cache by hand. Isolating the cache per test keeps the suite
    hermetic and lets a real run coexist with the tests. Tests that pass an
    explicit ``cache_dir`` are unaffected (the explicit value wins).
    """
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "emb-cache"))
