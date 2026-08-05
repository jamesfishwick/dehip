"""Unbiased MMD^2 between two embedding sets with a Gaussian RBF kernel.

Implements the unbiased Maximum Mean Discrepancy estimator of Gretton et al.
(JMLR 2012), the estimator the DFT benchmark cites. The kernel is a Gaussian
RBF and the bandwidth follows the median heuristic computed on the pooled
sample. See research.md R5.

The bandwidth is an explicit return value, not a hidden default: absolute MMD
values are only comparable across runs that share a bandwidth, and the DFT post
does not publish theirs, so every report must record the value used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MMDResult", "median_heuristic_bandwidth", "rbf_gram", "mmd2_unbiased"]


@dataclass(frozen=True)
class MMDResult:
    """Result of an unbiased MMD^2 computation.

    Attributes:
        mmd2: The unbiased MMD^2 estimate. Can be slightly negative for
            samples drawn from the same distribution (the unbiased estimator is
            not constrained to be non-negative).
        bandwidth: The Gaussian RBF bandwidth (sigma) actually used.
    """

    mmd2: float
    bandwidth: float


def _as_2d(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array (n x d), got shape {x.shape}")
    if x.shape[0] < 1:
        raise ValueError(f"{name} must have at least one row")
    return x


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between every row of ``a`` and ``b``."""
    a_sq = np.einsum("ij,ij->i", a, a)[:, None]
    b_sq = np.einsum("ij,ij->i", b, b)[None, :]
    sq = a_sq + b_sq - 2.0 * (a @ b.T)
    # Floating-point error can push diagonal / near-duplicate pairs slightly
    # below zero; clamp so the kernel never sees a negative distance.
    return np.maximum(sq, 0.0)


def median_heuristic_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    """Median-heuristic Gaussian bandwidth over the pooled sample.

    The bandwidth (sigma) is the square root of the median of the squared
    pairwise Euclidean distances across all points in ``x`` and ``y`` pooled
    together, excluding each point's zero self-distance.

    Raises:
        ValueError: if the pooled sample has fewer than two distinct points, so
            no positive bandwidth exists.
    """
    x = _as_2d("x", x)
    y = _as_2d("y", y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must share embedding dim, got {x.shape[1]} and {y.shape[1]}"
        )

    pooled = np.vstack([x, y])
    sq = _pairwise_sq_dists(pooled, pooled)
    # Exclude the zero self-distances on the diagonal from the median.
    iu = np.triu_indices(pooled.shape[0], k=1)
    off_diagonal = sq[iu]
    if off_diagonal.size == 0:
        raise ValueError("pooled sample must have at least two points")

    median_sq = float(np.median(off_diagonal))
    if median_sq <= 0.0:
        raise ValueError(
            "median pairwise distance is zero; pooled sample is degenerate "
            "(all points coincident), so no positive bandwidth exists"
        )
    return float(np.sqrt(median_sq))


def rbf_gram(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian RBF kernel matrix: exp(-||a_i - b_j||^2 / (2 * sigma^2))."""
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    sq = _pairwise_sq_dists(a, b)
    return np.exp(-sq / (2.0 * bandwidth * bandwidth))


def mmd2_unbiased(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float | None = None,
) -> MMDResult:
    """Unbiased MMD^2 estimate between two embedding sets.

    Uses the unbiased estimator of Gretton et al. (2012, eq. for MMD_u^2): the
    within-set kernel sums exclude the diagonal self-similarity terms, which is
    what makes the estimator unbiased. Requires at least two points per set.

    Args:
        x: First set of embeddings, shape (n, d), n >= 2.
        y: Second set of embeddings, shape (m, d), m >= 2.
        bandwidth: Gaussian RBF bandwidth (sigma). If None, the median
            heuristic over the pooled sample is used.

    Returns:
        An MMDResult carrying the MMD^2 estimate and the bandwidth used.

    Raises:
        ValueError: on shape mismatch, fewer than two points in either set, or
            a non-positive bandwidth.
    """
    x = _as_2d("x", x)
    y = _as_2d("y", y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must share embedding dim, got {x.shape[1]} and {y.shape[1]}"
        )
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        raise ValueError(
            "unbiased MMD^2 needs at least two points per set "
            f"(got n={n}, m={m}); the diagonal-excluded normalizers vanish otherwise"
        )

    if bandwidth is None:
        bandwidth = median_heuristic_bandwidth(x, y)
    elif bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")

    k_xx = rbf_gram(x, x, bandwidth)
    k_yy = rbf_gram(y, y, bandwidth)
    k_xy = rbf_gram(x, y, bandwidth)

    # Unbiased within-set terms exclude the diagonal (self-similarity) entries.
    sum_xx = float(k_xx.sum() - np.trace(k_xx))
    sum_yy = float(k_yy.sum() - np.trace(k_yy))
    term_xx = sum_xx / (n * (n - 1))
    term_yy = sum_yy / (m * (m - 1))
    # The cross term has no diagonal to exclude; every x_i vs y_j pair counts.
    term_xy = float(k_xy.sum()) / (n * m)

    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return MMDResult(mmd2=mmd2, bandwidth=float(bandwidth))
