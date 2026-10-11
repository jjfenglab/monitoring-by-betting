# DEPRECATED
# The first baseline we tried was Sliding window-based CP detection methods.
# This is not under consideration any more.
"""Tier-1 MMD-based streaming detector and its per-step threshold calibrator.

Baseline only: `SlidingMMDDetector` controls *per-step* false-alarm rate via a
pre-calibrated MMD^2 threshold. It is NOT anytime-valid — peeking after every
observation inflates Type-I error. Used as the comparison baseline against the
e-process / e-detector family (which IS anytime-valid via Ville).
"""

from __future__ import annotations

import numpy as np

from .kernels import median_heuristic_bandwidth, mmd_squared


def calibrate_threshold(
    reference: np.ndarray,
    window_size: int,
    sigma: float,
    alpha: float,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    """Calibrate a per-step alarm threshold from the reference distribution.

    Repeatedly draws a random sub-window of size `window_size` from the
    reference, compares it (via MMD^2) to the rest of the reference, and takes
    the (1 - alpha) quantile as the threshold.

    NOTE: This controls *per-step* false-alarm rate, not stream-level FPR. Use
    the e-process / e-detector family for sequentially valid (anytime) testing.
    """
    n_ref = len(reference)
    assert window_size < n_ref, (
        f"window_size ({window_size}) must be smaller than reference size ({n_ref})"
    )
    stats = np.empty(n_draws)
    for k in range(n_draws):
        idx = rng.choice(n_ref, size=window_size, replace=False)
        mask = np.ones(n_ref, dtype=bool)
        mask[idx] = False
        stats[k] = mmd_squared(reference[idx], reference[mask], sigma)
    return float(np.quantile(stats, 1 - alpha))


class SlidingMMDDetector:
    """Streaming MMD detector with a frozen reference and a sliding window.

    State:
        reference  : (N_REFERENCE, d) frozen embeddings from the calibration period.
        sigma      : RBF bandwidth, set once from the reference via median heuristic.
        threshold  : MMD^2 alarm threshold, calibrated from reference subsampling.
        window     : (<=window_size, d) ndarray of most recent stream observations.
        t          : integer count of stream observations ingested so far.
    """

    def __init__(
        self,
        reference: np.ndarray,
        window_size: int,
        alpha: float = 0.05,
        rng: np.random.Generator | None = None,
        n_calib_draws: int = 500,
        sigma: float | None = None,
        threshold: float | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng(0)
        self.reference = reference
        self.window_size = window_size
        self.sigma = (
            sigma if sigma is not None else median_heuristic_bandwidth(reference)
        )
        self.threshold = (
            threshold
            if threshold is not None
            else calibrate_threshold(
                reference, window_size, self.sigma, alpha, n_calib_draws, rng
            )
        )
        self.window = np.empty((0, reference.shape[1]))
        self.t = 0

    def update(self, x: np.ndarray) -> tuple[float, bool]:
        """Ingest one observation x (shape (d,)) and return (mmd_squared, alarm).

        Recomputing MMD^2 from scratch each step is O(R*W + W^2). At typical
        prototype scale (R=100, W=30) this is trivial. Online/incremental MMD
        exists but isn't worth the complexity here.
        """
        self.window = np.vstack([self.window, x[None, :]])
        if self.window.shape[0] > self.window_size:
            self.window = self.window[-self.window_size :]
        self.t += 1
        if self.window.shape[0] < self.window_size:
            return (0.0, False)
        stat = mmd_squared(self.reference, self.window, self.sigma)
        return (stat, stat > self.threshold)
