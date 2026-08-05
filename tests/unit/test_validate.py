"""Unit tests for input-set gates and degeneration detectors (FR-006/FR-009)."""

import pytest

from dehip.schemas import TextSet
from dehip.validate import (
    LENGTH_RATIO_MAX,
    LENGTH_RATIO_MIN,
    DegenerationReport,
    InputSetValidationError,
    detect_degeneration,
    validate_input_sets,
)


def _text_set(role: str, pair_ids: list[str]) -> TextSet:
    kwargs = {"round": 2} if role == "rewrite" else {}
    return TextSet(
        set_id=f"fineweb-{role}",
        role=role,
        corpus="fineweb",
        pair_ids=pair_ids,
        **kwargs,
    )


# --- Input-set gates (FR-009) -----------------------------------------------


def test_pairing_gate_accepts_identical_pair_ids():
    cand = _text_set("rewrite", ["fineweb-1", "fineweb-2"])
    ref = _text_set("human_reference", ["fineweb-1", "fineweb-2"])
    # Should not raise.
    validate_input_sets(cand, ref)


def test_pairing_gate_rejects_hand_broken_manifest():
    # Same count, but one pair_id differs -- a hand-broken manifest.
    cand = _text_set("rewrite", ["fineweb-1", "fineweb-2"])
    ref = _text_set("human_reference", ["fineweb-1", "fineweb-99"])
    with pytest.raises(InputSetValidationError, match="pair_id mismatch"):
        validate_input_sets(cand, ref)


def test_pairing_gate_reports_asymmetric_difference():
    cand = _text_set("rewrite", ["a", "b", "c"])
    ref = _text_set("human_reference", ["a", "b", "z"])
    with pytest.raises(InputSetValidationError) as exc:
        validate_input_sets(cand, ref)
    assert "'c'" in str(exc.value)  # only in candidate
    assert "'z'" in str(exc.value)  # only in reference


def test_count_gate_accepts_matching_counts():
    validate_input_sets(["a", "b"], ["a", "b"])


def test_count_gate_rejects_count_mismatch():
    cand = _text_set("rewrite", ["a", "b", "c"])
    ref = _text_set("human_reference", ["a", "b"])
    with pytest.raises(InputSetValidationError, match="count mismatch"):
        validate_input_sets(cand, ref)


def test_min_length_gate_accepts_sufficient_texts():
    texts = {"a": "a real document", "b": "another real one"}
    validate_input_sets(["a", "b"], ["a", "b"], texts=texts, min_text_length=5)


def test_min_length_gate_rejects_short_text():
    texts = {"a": "a real document", "b": "  "}
    with pytest.raises(InputSetValidationError, match="shorter than"):
        validate_input_sets(["a", "b"], ["a", "b"], texts=texts, min_text_length=5)


def test_min_length_gate_rejects_missing_text():
    texts = {"a": "a real document"}
    with pytest.raises(InputSetValidationError, match="missing"):
        validate_input_sets(["a", "b"], ["a", "b"], texts=texts)


def test_min_n_gate_accepts_at_floor():
    validate_input_sets(["a", "b"], ["a", "b"], min_n=2)


def test_min_n_gate_rejects_below_floor():
    with pytest.raises(InputSetValidationError, match="too few pairs"):
        validate_input_sets(["a"], ["a"], min_n=2)


def test_min_n_gate_override_flag_allows_below_floor():
    # Explicit override lets a single-pair set through.
    validate_input_sets(["a"], ["a"], min_n=2, allow_below_min_n=True)


# --- Degeneration detectors (FR-006, R10 thresholds) ------------------------


def test_healthy_round_trips_nothing():
    text = (
        "The morning fog lifted slowly. Birds began their chorus. "
        "A cyclist rolled past the bakery. Warm bread scented the street."
    )
    report = detect_degeneration(text, prior_length=len(text))
    assert isinstance(report, DegenerationReport)
    assert not report.hard_tripped
    assert report.flags == []


def test_empty_output_hard_trips():
    report = detect_degeneration("   \n\t  ")
    assert report.empty is True
    assert report.hard_tripped is True
    assert "empty" in report.flags


def test_length_ratio_within_bounds_does_not_trip():
    # Ratio 1.0 -- squarely inside [0.5, 2.0].
    text = "word " * 20
    report = detect_degeneration(text, prior_length=len(text))
    assert report.length_ratio == pytest.approx(1.0)
    assert not report.length_ratio_tripped
    assert not report.hard_tripped


def test_length_ratio_explosion_hard_trips():
    prior = 100
    text = "x" * 300  # ratio 3.0 > 2.0
    report = detect_degeneration(text, prior_length=prior)
    assert report.length_ratio > LENGTH_RATIO_MAX
    assert report.length_ratio_tripped is True
    assert report.hard_tripped is True
    assert "length_ratio" in report.flags


def test_length_ratio_collapse_hard_trips():
    prior = 100
    text = "x" * 40  # ratio 0.4 < 0.5
    report = detect_degeneration(text, prior_length=prior)
    assert report.length_ratio < LENGTH_RATIO_MIN
    assert report.hard_tripped is True


def test_repetition_flags_without_hard_tripping():
    # Four consecutive sentences all starting with "The" -- flag, not fatal.
    text = (
        "The sky was clear. The road was long. "
        "The car was fast. The trip was good."
    )
    report = detect_degeneration(text, prior_length=len(text))
    assert report.repetition_flagged is True
    assert "repetition" in report.flags
    assert report.hard_tripped is False  # flag-only, never fatal


def test_non_ascii_burst_hard_trips():
    # Well above 5% non-ASCII characters.
    text = "normal " + ("éüñç" * 20)
    report = detect_degeneration(text)
    assert report.non_ascii_fraction > 0.05
    assert report.non_ascii_burst_tripped is True
    assert report.hard_tripped is True
    assert "non_ascii_burst" in report.flags


def test_low_non_ascii_does_not_trip():
    # A single accented char in a long ASCII run stays under 5%.
    text = "This is a perfectly normal English sentence with one café."
    report = detect_degeneration(text)
    assert report.non_ascii_fraction < 0.05
    assert not report.non_ascii_burst_tripped
    assert not report.hard_tripped


def test_first_round_skips_length_ratio():
    report = detect_degeneration("A normal first round.", prior_length=None)
    assert report.length_ratio is None
    assert not report.hard_tripped
