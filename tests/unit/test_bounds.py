"""Unit tests for the instrument-aware noise bounds.

The load-bearing invariant here: bounds are only comparable to the instrument
that produced them. ``documented()`` must return the real-instrument bounds only
for the real embedder, and the stub bounds for the stub embedder or an
unspecified instrument -- otherwise a stub self-check would be judged against a
real-instrument noise floor an order of magnitude tighter than its own.
"""

from __future__ import annotations

import pytest

from dehip.metrics.bounds import (
    REAL_EMBEDDER_ID,
    REAL_INSTRUMENT_BOUNDS,
    STUB_INSTRUMENT_BOUNDS,
    StubInstrumentBounds,
)


def test_real_embedder_id_matches_embeddings_default():
    """Drift guard: the local mirror equals the real embedder constant.

    bounds.py keeps REAL_EMBEDDER_ID as a local literal to avoid importing the
    torch-heavy embeddings module. If the canonical constant ever moves, this
    fails loudly instead of silently selecting the wrong bounds.
    """
    from dehip.metrics.embeddings import DEFAULT_EMBEDDER_ID

    assert REAL_EMBEDDER_ID == DEFAULT_EMBEDDER_ID


def test_documented_none_returns_stub():
    """An unspecified instrument gets the safe stub default, never the real set."""
    assert StubInstrumentBounds.documented(None) is STUB_INSTRUMENT_BOUNDS


def test_documented_stub_embedder_returns_stub():
    """A non-real embedder id resolves to the stub bounds."""
    assert (
        StubInstrumentBounds.documented("stub-embedder-8d") is STUB_INSTRUMENT_BOUNDS
    )


def test_documented_real_embedder_returns_real():
    """The real embedder resolves to the filled real-instrument bounds."""
    assert REAL_INSTRUMENT_BOUNDS is not None  # filled from the #16 smoke run
    assert StubInstrumentBounds.documented(REAL_EMBEDDER_ID) is REAL_INSTRUMENT_BOUNDS


def test_documented_real_embedder_falls_back_to_stub_when_real_unset(monkeypatch):
    """If the real slot is empty, even the real embedder gets the stub default.

    Guards the ``REAL_INSTRUMENT_BOUNDS is not None`` branch: an unfilled slot
    must not crash or return None; it falls back to the stub bounds.
    """
    monkeypatch.setattr("dehip.metrics.bounds.REAL_INSTRUMENT_BOUNDS", None)
    assert (
        StubInstrumentBounds.documented(REAL_EMBEDDER_ID) is STUB_INSTRUMENT_BOUNDS
    )


def test_real_bounds_are_tighter_than_stub():
    """The real embedder's noise floor is well below the stub's, as observed."""
    assert REAL_INSTRUMENT_BOUNDS is not None
    assert REAL_INSTRUMENT_BOUNDS.mmd_max < STUB_INSTRUMENT_BOUNDS.mmd_max
    assert REAL_INSTRUMENT_BOUNDS.token_l2_max < STUB_INSTRUMENT_BOUNDS.token_l2_max


@pytest.mark.parametrize("value", [0.000307, 0.05, -0.05])
def test_observed_self_check_mmd_is_within_real_bounds(value):
    """The smoke run's observed MMD (0.000307) sits comfortably inside the set."""
    assert REAL_INSTRUMENT_BOUNDS is not None
    assert REAL_INSTRUMENT_BOUNDS.mmd_min <= value <= REAL_INSTRUMENT_BOUNDS.mmd_max
