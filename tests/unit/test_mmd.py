"""Known-answer tests for the unbiased MMD^2 estimator (research.md R5)."""

import math

import numpy as np
import pytest

from dehip.metrics.mmd import (
    MMDResult,
    median_heuristic_bandwidth,
    mmd2_unbiased,
)


def _naive_mmd2_unbiased(
    x: np.ndarray, y: np.ndarray, bandwidth: float
) -> float:
    """Reference unbiased MMD^2 via explicit loops (diagonal excluded).

    Independent of the vectorized implementation, so agreement between the two
    is real evidence the diagonal terms are being excluded correctly.
    """

    def k(a, b):
        d2 = float(np.sum((a - b) ** 2))
        return math.exp(-d2 / (2.0 * bandwidth * bandwidth))

    n = len(x)
    m = len(y)

    xx = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:  # exclude the diagonal -> unbiased
                xx += k(x[i], x[j])
    xx /= n * (n - 1)

    yy = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                yy += k(y[i], y[j])
    yy /= m * (m - 1)

    xy = 0.0
    for i in range(n):
        for j in range(m):
            xy += k(x[i], y[j])
    xy /= n * m

    return xx + yy - 2.0 * xy


def _naive_mmd2_biased(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Biased V-statistic that KEEPS the diagonal, for the contrast test."""

    def k(a, b):
        d2 = float(np.sum((a - b) ** 2))
        return math.exp(-d2 / (2.0 * bandwidth * bandwidth))

    n = len(x)
    m = len(y)
    xx = sum(k(x[i], x[j]) for i in range(n) for j in range(n)) / (n * n)
    yy = sum(k(y[i], y[j]) for i in range(m) for j in range(m)) / (m * m)
    xy = sum(k(x[i], y[j]) for i in range(n) for j in range(m)) / (n * m)
    return xx + yy - 2.0 * xy


# --- Known-answer: tiny hand-computed 2-to-3 point case ----------------------


def test_tiny_hand_computed_case():
    """A 2-vs-3 point case whose MMD^2 is derived by hand below.

    Points are 1D (as (n, 1) arrays). Fix bandwidth sigma=1 so the kernel is
    k(a, b) = exp(-(a-b)^2 / 2).

    x = [0, 2]                 (n = 2)
    y = [1, 1, 3]              (m = 3)

    Kernel value for squared distance s: exp(-s / 2).

    XX term (i != j): only the pair (0, 2), squared dist 4, both orders ->
        2 * exp(-4/2) = 2 * exp(-2)
        normalizer n(n-1) = 2  =>  term_xx = exp(-2)

    YY term (i != j): pairs among [1, 1, 3]
        (1,1) sq 0 -> exp(0)=1, appears both orders -> 2 * 1
        (1,3) sq 4 -> exp(-2), two such unordered pairs (y0-y2, y1-y2),
            both orders -> 4 * exp(-2)
        sum = 2 + 4*exp(-2); normalizer m(m-1) = 6
        term_yy = (2 + 4*exp(-2)) / 6

    XY term (all n*m pairs):
        x0=0 vs y: (0,1) sq1 -> e^-0.5 ; (0,1) sq1 -> e^-0.5 ; (0,3) sq9 -> e^-4.5
        x1=2 vs y: (2,1) sq1 -> e^-0.5 ; (2,1) sq1 -> e^-0.5 ; (2,3) sq1 -> e^-0.5
        sum = 5*e^-0.5 + e^-4.5 ; normalizer n*m = 6
        term_xy = (5*exp(-0.5) + exp(-4.5)) / 6

    mmd2 = term_xx + term_yy - 2*term_xy
    """
    x = np.array([[0.0], [2.0]])
    y = np.array([[1.0], [1.0], [3.0]])

    term_xx = math.exp(-2.0)
    term_yy = (2.0 + 4.0 * math.exp(-2.0)) / 6.0
    term_xy = (5.0 * math.exp(-0.5) + math.exp(-4.5)) / 6.0
    expected = term_xx + term_yy - 2.0 * term_xy

    result = mmd2_unbiased(x, y, bandwidth=1.0)
    assert isinstance(result, MMDResult)
    assert result.bandwidth == 1.0
    assert result.mmd2 == pytest.approx(expected, abs=1e-12)


# --- Known-answer: same distribution -> near zero ----------------------------


def test_same_gaussian_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(400, 8))
    y = rng.normal(0.0, 1.0, size=(400, 8))

    result = mmd2_unbiased(x, y)
    # Unbiased estimator centers on zero under H0; can be slightly negative.
    assert abs(result.mmd2) < 0.02
    assert result.bandwidth > 0.0


# --- Known-answer: shifted distribution -> known positive --------------------


def test_shifted_gaussian_is_positive():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(400, 8))
    y = rng.normal(3.0, 1.0, size=(400, 8))  # large mean shift

    result = mmd2_unbiased(x, y)
    # A 3-sigma shift in every dim is far apart; MMD^2 should be solidly
    # positive and well clear of the near-zero H0 band.
    assert result.mmd2 > 0.3
    assert result.bandwidth > 0.0


def test_shift_exceeds_same_distribution():
    """The shifted pair must score higher than the same-distribution pair."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, size=(300, 6))
    b = rng.normal(0.0, 1.0, size=(300, 6))
    c = rng.normal(2.0, 1.0, size=(300, 6))

    same = mmd2_unbiased(a, b).mmd2
    shifted = mmd2_unbiased(a, c).mmd2
    assert shifted > same
    assert shifted > 0.1


# --- Unbiasedness: matches naive reference, differs from biased --------------


def test_matches_naive_unbiased_reference():
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(15, 4))
    y = rng.normal(0.5, 1.2, size=(11, 4))
    sigma = 1.3

    result = mmd2_unbiased(x, y, bandwidth=sigma)
    expected = _naive_mmd2_unbiased(x, y, sigma)
    assert result.mmd2 == pytest.approx(expected, abs=1e-10)


def test_diagonal_is_actually_excluded():
    """Confirm the estimator excludes the diagonal (differs from biased)."""
    rng = np.random.default_rng(99)
    x = rng.normal(0.0, 1.0, size=(12, 3))
    y = rng.normal(0.3, 1.0, size=(9, 3))
    sigma = 1.1

    ours = mmd2_unbiased(x, y, bandwidth=sigma).mmd2
    unbiased_ref = _naive_mmd2_unbiased(x, y, sigma)
    biased_ref = _naive_mmd2_biased(x, y, sigma)

    assert ours == pytest.approx(unbiased_ref, abs=1e-10)
    # The two references genuinely differ, so matching one is meaningful.
    assert abs(unbiased_ref - biased_ref) > 1e-6


# --- Bandwidth as an explicit, honored value ---------------------------------


def test_median_heuristic_matches_returned_bandwidth():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=(50, 5))
    y = rng.normal(1.0, 1.0, size=(50, 5))

    expected_bw = median_heuristic_bandwidth(x, y)
    result = mmd2_unbiased(x, y)  # bandwidth=None -> median heuristic
    assert result.bandwidth == pytest.approx(expected_bw, abs=1e-12)


def test_explicit_bandwidth_is_honored():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, size=(30, 4))
    y = rng.normal(0.0, 1.0, size=(30, 4))
    result = mmd2_unbiased(x, y, bandwidth=2.5)
    assert result.bandwidth == 2.5


# --- Error handling ----------------------------------------------------------


def test_rejects_single_point_set():
    x = np.array([[0.0, 0.0]])
    y = np.array([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="at least two points"):
        mmd2_unbiased(x, y)


def test_rejects_dim_mismatch():
    x = np.zeros((3, 4))
    y = np.zeros((3, 5))
    with pytest.raises(ValueError, match="embedding dim"):
        mmd2_unbiased(x, y)


def test_rejects_nonpositive_bandwidth():
    x = np.zeros((3, 2))
    y = np.ones((3, 2))
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        mmd2_unbiased(x, y, bandwidth=0.0)
