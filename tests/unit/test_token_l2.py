"""Known-answer tests for the token-frequency L2 metric (research.md R7).

These use a STUB tokenizer (whitespace split) so the suite never downloads the
real Qwen3 tokenizer. The metric's arithmetic is tokenizer-agnostic, so a
whitespace split exercises every path the real tokenizer would: union-vocab
layout, per-set frequency normalization, and the Euclidean distance itself.
"""

import math

import numpy as np
import pytest

from dehip.metrics.token_l2 import (
    DEFAULT_TOKENIZER_ID,
    Qwen3Tokenizer,
    TokenL2Result,
    token_l2,
)


def _whitespace(text: str) -> list[str]:
    """Stub tokenizer: split on whitespace."""
    return text.split()


class _WhitespaceTokenizer:
    """Stub object-shaped tokenizer exposing tokenize + a recorded id."""

    tokenizer_id = "stub-whitespace"

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def test_identical_sets_give_zero() -> None:
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    result = token_l2(texts, list(texts), tokenizer=_whitespace)
    assert result.distance == 0.0


def test_identical_distribution_different_totals_gives_zero() -> None:
    # Same token distribution, different total length: frequencies match, so the
    # distance is zero. Proves the metric normalizes per set rather than
    # comparing raw counts.
    result = token_l2(["a b"], ["a b a b a b"], tokenizer=_whitespace)
    assert result.distance == pytest.approx(0.0)


def test_fully_disjoint_vocabularies_give_sqrt2() -> None:
    # Set A is all "a", set B is all "b". Union vocab {a, b}. Frequency vectors
    # are (1, 0) and (0, 1); their L2 distance is sqrt(1^2 + 1^2) = sqrt(2).
    result = token_l2(["a a a"], ["b b"], tokenizer=_whitespace)
    assert result.distance == pytest.approx(math.sqrt(2.0))
    assert result.vocab_size == 2


def test_two_token_case_matches_closed_form() -> None:
    # Set A: "a a a b"  -> freq {a: 3/4, b: 1/4}
    # Set B: "a b b b"  -> freq {a: 1/4, b: 3/4}
    # Union vocab {a, b}. Differences per token: 3/4 - 1/4 = 1/2 for both.
    # L2 = sqrt((1/2)^2 + (1/2)^2) = sqrt(1/2).
    result = token_l2(["a a a b"], ["a b b b"], tokenizer=_whitespace)
    expected = math.sqrt(0.5)
    assert result.distance == pytest.approx(expected)

    # Independent recomputation of the frequency vectors.
    a = np.array([3 / 4, 1 / 4])
    b = np.array([1 / 4, 3 / 4])
    assert result.distance == pytest.approx(float(np.linalg.norm(a - b)))


def test_tokenizer_id_present_and_recorded() -> None:
    # An object tokenizer that exposes tokenizer_id has that id surfaced.
    result = token_l2(["a b"], ["a c"], tokenizer=_WhitespaceTokenizer())
    assert isinstance(result, TokenL2Result)
    assert result.tokenizer_id == "stub-whitespace"


def test_bare_callable_tokenizer_id_falls_back_to_name() -> None:
    # A plain callable with no tokenizer_id attribute still yields a non-empty id
    # (its function name), so the value is never recorded without provenance.
    result = token_l2(["a"], ["b"], tokenizer=_whitespace)
    assert result.tokenizer_id == "_whitespace"


def test_default_tokenizer_id_is_qwen3() -> None:
    # The default seam is the real Qwen3 tokenizer; its recorded id is the Qwen3
    # model id. Constructing it must not trigger a download (lazy load).
    tok = Qwen3Tokenizer()
    assert tok.tokenizer_id == DEFAULT_TOKENIZER_ID


def test_deterministic_same_inputs_identical_result() -> None:
    a = ["alpha beta beta", "gamma alpha"]
    b = ["beta gamma gamma", "alpha"]
    first = token_l2(a, b, tokenizer=_whitespace)
    second = token_l2(a, b, tokenizer=_whitespace)
    assert first == second


def test_empty_set_against_populated_set() -> None:
    # An empty set is an all-zero distribution; the other set's frequencies sum
    # to 1 across its vocab, so the distance is the L2 norm of that distribution.
    result = token_l2([], ["a a b b"], tokenizer=_whitespace)
    # freq {a: 1/2, b: 1/2}; distance from zero vector = sqrt(1/4 + 1/4).
    assert result.distance == pytest.approx(math.sqrt(0.5))
    assert result.vocab_size == 2


def test_invalid_tokenizer_type_raises() -> None:
    with pytest.raises(TypeError):
        token_l2(["a"], ["b"], tokenizer=object())
