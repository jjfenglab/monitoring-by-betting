# DEPRECATED
# The first baseline we tried was Sliding window-based CP detection methods.
# This is not under consideration any more.
"""Kernel and MMD primitives for two-sample testing."""

from __future__ import annotations

import numpy as np


def pairwise_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared Euclidean pairwise distances between rows of X and rows of Y."""
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)


def rbf_kernel(D: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel matrix from a squared-distance matrix D."""
    return np.exp(-D / (2 * sigma**2))


def median_heuristic_bandwidth(X: np.ndarray) -> float:
    """RBF bandwidth sigma chosen as the median pairwise Euclidean distance.

    Standard default for MMD; robust to feature scale.
    """
    diffs = X[:, None, :] - X[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=-1))
    iu = np.triu_indices_from(dists, k=1)
    return float(np.median(dists[iu]))


def mmd_squared(X1: np.ndarray, X2: np.ndarray, sigma: float) -> float:
    """Biased estimator of squared MMD with an RBF kernel.

    The biased estimator is always non-negative and is fine for permutation
    testing because the null distribution is calibrated against the same
    estimator. Switch to the unbiased version if reporting MMD^2 as an effect
    size across studies.
    """
    K_xx = rbf_kernel(pairwise_dist(X1, X1), sigma)
    K_yy = rbf_kernel(pairwise_dist(X2, X2), sigma)
    K_xy = rbf_kernel(pairwise_dist(X1, X2), sigma)
    return float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
