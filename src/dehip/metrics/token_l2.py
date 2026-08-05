"""Token-frequency L2 distance between two text sets.

Tokenizes each set with the Qwen3 tokenizer (the model family under test),
builds a 1-gram token-frequency vector for each set over the *union* vocabulary
of both, and returns the L2 (Euclidean) distance between those two frequency
distributions. See research.md R7 and spec FR-001 (the token-L2-on-1-grams
clause).

The tokenizer identity travels with the value. The DFT post never names its
tokenizer, so absolute comparability to the benchmark rows is best-effort; the
only way our number stays honest is to record which tokenizer produced it. The
returned ``tokenizer_id`` is that record.

The tokenizer is an injectable seam. The default constructs the real Qwen3
tokenizer lazily (it is only loaded on first use, so importing this module never
pulls it), but any tokenize-callable or an object exposing a ``tokenize`` method
can be supplied instead, which is what the tests do to avoid a download.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_TOKENIZER_ID",
    "TokenL2Result",
    "Qwen3Tokenizer",
    "token_l2",
]

DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"

# A tokenizer seam is either a plain callable str -> list[token] or an object
# exposing a ``tokenize(str) -> list[token]`` method (the transformers shape).
Tokenizer = Callable[[str], Sequence[str]]


@dataclass(frozen=True)
class TokenL2Result:
    """Result of a token-frequency L2 computation.

    Attributes:
        distance: The L2 (Euclidean) distance between the two 1-gram token
            frequency distributions, over the union vocabulary of both sets.
            Zero when the two sets have identical token distributions.
        tokenizer_id: Identity of the tokenizer that produced the counts, so the
            value can be reproduced and its comparability caveat kept honest.
        vocab_size: Number of distinct tokens in the union vocabulary (the
            dimensionality of the frequency vectors that were compared).
    """

    distance: float
    tokenizer_id: str
    vocab_size: int


class Qwen3Tokenizer:
    """The default production tokenizer: the Qwen3 tokenizer via transformers.

    Lazy on purpose: the weights/vocab load on first ``tokenize`` call so that
    constructing this object (and importing this module) never triggers a
    download. Tests inject a stub callable instead of this class.
    """

    def __init__(self, model_name: str = DEFAULT_TOKENIZER_ID) -> None:
        self.tokenizer_id = model_name
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

    def tokenize(self, text: str) -> list[str]:
        self._ensure_loaded()
        assert self._tokenizer is not None  # narrowed by _ensure_loaded
        return self._tokenizer.tokenize(text)


def _resolve_tokenizer(tokenizer: Tokenizer | object | None) -> tuple[Tokenizer, str]:
    """Return a (tokenize-callable, tokenizer_id) pair from the seam argument.

    Accepts three shapes: None (build the default Qwen3 tokenizer), a plain
    callable, or an object exposing a ``tokenize`` method. The id is read from a
    ``tokenizer_id`` attribute when present, otherwise a best-effort label.
    """
    if tokenizer is None:
        tokenizer = Qwen3Tokenizer()

    tokenizer_id = getattr(tokenizer, "tokenizer_id", None)

    tokenize_method = getattr(tokenizer, "tokenize", None)
    if callable(tokenize_method):
        fn: Tokenizer = tokenize_method
    elif callable(tokenizer):
        fn = tokenizer  # type: ignore[assignment]
    else:
        raise TypeError(
            "tokenizer must be None, a callable str -> tokens, or an object with "
            f"a tokenize(str) method, got {type(tokenizer).__name__}"
        )

    if tokenizer_id is None:
        tokenizer_id = getattr(fn, "__name__", type(tokenizer).__name__)

    return fn, str(tokenizer_id)


def _frequency_counts(texts: Sequence[str], tokenize: Tokenizer) -> Counter[str]:
    """1-gram token counts pooled across every text in a set."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts


def _as_distribution(counts: Counter[str], vocab: Sequence[str]) -> np.ndarray:
    """Normalized frequency vector for ``counts`` laid out over ``vocab``.

    Frequencies (counts divided by the set's total token count), not raw counts,
    so two sets of different total length are compared on the same scale. An
    empty set maps to an all-zero vector.
    """
    total = sum(counts.values())
    vec = np.array([counts.get(tok, 0) for tok in vocab], dtype=np.float64)
    if total > 0:
        vec /= total
    return vec


def token_l2(
    text_set_a: Sequence[str],
    text_set_b: Sequence[str],
    *,
    tokenizer: Tokenizer | object | None = None,
) -> TokenL2Result:
    """L2 distance between the 1-gram token-frequency distributions of two sets.

    Both sets are tokenized, pooled into per-set 1-gram counts, normalized to
    frequency distributions, and laid out over the *union* vocabulary of both
    sets. The result is the Euclidean distance between those two vectors.

    Deterministic: identical inputs and tokenizer yield an identical result.

    Args:
        text_set_a: First set of texts.
        text_set_b: Second set of texts.
        tokenizer: The tokenizer seam. None builds the default Qwen3 tokenizer.
            May also be a plain callable ``str -> tokens`` or an object exposing
            a ``tokenize(str)`` method (and optionally a ``tokenizer_id``).

    Returns:
        A TokenL2Result with the distance, the tokenizer id, and the union
        vocabulary size.
    """
    tokenize, tokenizer_id = _resolve_tokenizer(tokenizer)

    counts_a = _frequency_counts(text_set_a, tokenize)
    counts_b = _frequency_counts(text_set_b, tokenize)

    # Union vocabulary, sorted for a deterministic vector layout.
    vocab = sorted(set(counts_a) | set(counts_b))

    dist_a = _as_distribution(counts_a, vocab)
    dist_b = _as_distribution(counts_b, vocab)

    distance = float(np.linalg.norm(dist_a - dist_b))
    return TokenL2Result(
        distance=distance,
        tokenizer_id=tokenizer_id,
        vocab_size=len(vocab),
    )
