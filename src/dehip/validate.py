"""Validation gates and degeneration detectors for the HIP cascade harness.

Two independent concerns live here:

1. **Input-set gates (FR-009).** Cheap, pre-scoring structural checks on a
   candidate/reference pair of text sets: identical ``pair_id`` pairing,
   count match, a minimum per-text length, and a minimum N floor. Every gate
   runs before any external scoring cost is incurred, so a hand-broken
   manifest is rejected loudly rather than silently scored.

2. **Degeneration detectors (FR-006, thresholds per research R10).** Per-round
   checks on a rewrite's text: empty/whitespace output, length ratio vs the
   prior round outside [0.5, 2.0], three or more consecutive sentences sharing
   a start word, and a non-ASCII character burst above 5%. The repetition
   check FLAGS only (human text trips it ~17% of the time), while the other
   checks are HARD trips that signal stop-and-flag. The distinction is
   preserved in the returned record: ``hard_tripped`` is true only for a hard
   check, a flag is recorded but never fatal.

Every threshold is anchored to a measured value in research.md R10 rather
than invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_MIN_N",
    "DEFAULT_MIN_TEXT_LENGTH",
    "LENGTH_RATIO_MAX",
    "LENGTH_RATIO_MIN",
    "NON_ASCII_BURST_FRACTION",
    "REPETITION_RUN_LENGTH",
    "DegenerationReport",
    "InputSetValidationError",
    "detect_degeneration",
    "validate_input_sets",
]

# --- Thresholds (research.md R10 + data-model.md) ---------------------------

# Input-set gate defaults (FR-009). data-model states the fineweb tier holds
# 150-1200 word documents; a low character floor rejects obviously empty or
# truncated texts without second-guessing the corpus's own word_count bounds.
DEFAULT_MIN_TEXT_LENGTH = 1
DEFAULT_MIN_N = 2

# Degeneration thresholds (research.md R10). Length ratio vs the prior round
# must stay within [0.5, 2.0]; a non-ASCII burst above 5% of characters hard
# trips (the DFT post measured 0.1% in human data, 8.1% in DFT output); three
# or more consecutive sentences sharing a start word flags only (human text
# trips this ~17% of the time).
LENGTH_RATIO_MIN = 0.5
LENGTH_RATIO_MAX = 2.0
NON_ASCII_BURST_FRACTION = 0.05
REPETITION_RUN_LENGTH = 3


class InputSetValidationError(ValueError):
    """Raised when a candidate/reference text-set pair fails an input gate."""


# --- Input-set gates (FR-009) -----------------------------------------------


def _pair_ids(text_set: Any, name: str) -> list[str]:
    """Extract the ``pair_ids`` list from a TextSet-like object or a sequence."""
    pair_ids = getattr(text_set, "pair_ids", text_set)
    if not isinstance(pair_ids, (list, tuple)):
        raise InputSetValidationError(
            f"{name} must expose a list of pair_ids, got {type(pair_ids).__name__}"
        )
    return list(pair_ids)


def validate_input_sets(
    candidate: Any,
    reference: Any,
    *,
    texts: dict[str, str] | None = None,
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
    min_n: int = DEFAULT_MIN_N,
    allow_below_min_n: bool = False,
) -> None:
    """Validate a candidate/reference text-set pair before any scoring cost.

    Runs four gates (FR-009), raising :class:`InputSetValidationError` with a
    clear message on the first failure:

    - **Pairing.** ``candidate`` and ``reference`` must reference identical
      ``pair_id`` sets (same ids, no extras on either side).
    - **Count match.** The two sets must hold the same number of pair_ids.
    - **Minimum length.** When ``texts`` is supplied, every referenced text
      must be at least ``min_text_length`` characters after stripping.
    - **Minimum N.** The set size must be at least ``min_n`` unless
      ``allow_below_min_n`` explicitly overrides the floor.

    Args:
        candidate: TextSet-like object (has ``pair_ids``) or a list of ids.
        reference: TextSet-like object or a list of ids.
        texts: Optional ``{pair_id: text}`` map for the length gate. When
            omitted, the length gate is skipped (ids-only validation).
        min_text_length: Minimum stripped character length per text.
        min_n: Minimum number of pairs required to score.
        allow_below_min_n: Explicit override letting a set fall below ``min_n``.
    """
    cand_ids = _pair_ids(candidate, "candidate")
    ref_ids = _pair_ids(reference, "reference")

    # Count match. Checked before the set comparison so a length mismatch is
    # reported as a count problem rather than a confusing membership diff.
    if len(cand_ids) != len(ref_ids):
        raise InputSetValidationError(
            f"count mismatch: candidate has {len(cand_ids)} pair_ids, "
            f"reference has {len(ref_ids)}"
        )

    # Pairing: identical pair_id sets. Report the asymmetric difference so the
    # caller can find the hand-broken id.
    cand_set, ref_set = set(cand_ids), set(ref_ids)
    if cand_set != ref_set:
        only_cand = sorted(cand_set - ref_set)
        only_ref = sorted(ref_set - cand_set)
        raise InputSetValidationError(
            "pair_id mismatch between candidate and reference sets; "
            f"only in candidate: {only_cand}; only in reference: {only_ref}"
        )

    # Minimum N floor, with an explicit override flag.
    if not allow_below_min_n and len(cand_ids) < min_n:
        raise InputSetValidationError(
            f"too few pairs to score: {len(cand_ids)} < minimum N {min_n} "
            "(pass allow_below_min_n=True to override)"
        )

    # Minimum text length (only when texts are supplied).
    if texts is not None:
        for pair_id in cand_ids:
            for role, mapping in (("candidate", texts), ("reference", texts)):
                text = mapping.get(pair_id)
                if text is None:
                    raise InputSetValidationError(
                        f"{role} text for pair_id {pair_id!r} is missing from "
                        "the supplied texts map"
                    )
                if len(text.strip()) < min_text_length:
                    raise InputSetValidationError(
                        f"{role} text for pair_id {pair_id!r} is shorter than "
                        f"the minimum length {min_text_length}"
                    )


# --- Degeneration detectors (FR-006, thresholds per research R10) -----------


@dataclass
class DegenerationReport:
    """Per-round degeneration check results (data-model.md degeneration object).

    ``hard_tripped`` is true only for a hard check (empty, length ratio,
    non-ASCII burst); the repetition check is flag-only and never sets it. A
    hard trip signals stop-iteration-at-the-last-good-round; a flag is recorded
    but not fatal.
    """

    empty: bool = False
    length_ratio: float | None = None
    length_ratio_tripped: bool = False
    repetition_flagged: bool = False
    non_ascii_fraction: float = 0.0
    non_ascii_burst_tripped: bool = False
    hard_tripped: bool = False
    flags: list[str] = field(default_factory=list)


# A sentence boundary is a run of . ! or ? followed by whitespace. Good enough
# for the repetitiveness probe, which only needs sentence-start words.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")


def _sentence_start_words(text: str) -> list[str]:
    """First alphabetic word (lowercased) of each non-empty sentence."""
    starts: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        match = _WORD.search(sentence)
        if match:
            starts.append(match.group(0).lower())
    return starts


def _max_consecutive_run(items: list[str]) -> int:
    """Length of the longest run of an identical consecutive value."""
    best = run = 0
    prev: str | None = None
    for item in items:
        run = run + 1 if item == prev else 1
        best = max(best, run)
        prev = item
    return best


def detect_degeneration(
    text: str,
    *,
    prior_length: int | None = None,
) -> DegenerationReport:
    """Run per-round degeneration checks on one rewrite's ``text`` (R10).

    Hard checks (set ``hard_tripped``): empty/whitespace output; length ratio
    vs ``prior_length`` outside [0.5, 2.0]; non-ASCII characters above 5% of
    the total. Flag-only check (recorded in ``flags``, never fatal): three or
    more consecutive sentences sharing a start word.

    Args:
        text: The round's rewrite output.
        prior_length: Character length of the prior round's output, for the
            length-ratio check. Omit for the first round (ratio is skipped).

    Returns:
        A :class:`DegenerationReport` with each check's result.
    """
    report = DegenerationReport()

    # Empty / whitespace (hard).
    if not text.strip():
        report.empty = True
        report.hard_tripped = True
        report.flags.append("empty")
        # Nothing else is meaningful on an empty string; return early.
        return report

    # Length ratio vs the prior round (hard). Skipped on the first round.
    if prior_length is not None and prior_length > 0:
        ratio = len(text) / prior_length
        report.length_ratio = ratio
        if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
            report.length_ratio_tripped = True
            report.hard_tripped = True
            report.flags.append("length_ratio")

    # Non-ASCII burst (hard). Fraction of characters outside 7-bit ASCII.
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    fraction = non_ascii / len(text)
    report.non_ascii_fraction = fraction
    if fraction > NON_ASCII_BURST_FRACTION:
        report.non_ascii_burst_tripped = True
        report.hard_tripped = True
        report.flags.append("non_ascii_burst")

    # Consecutive-sentence repetition (flag only, never hard). Human text trips
    # this ~17% of the time, so it is recorded but does not stop iteration.
    if _max_consecutive_run(_sentence_start_words(text)) >= REPETITION_RUN_LENGTH:
        report.repetition_flagged = True
        report.flags.append("repetition")

    return report
