"""Online embedding drift statistics after Evidently (baselines): KS drifted share and cosine distance.

Both compare a fixed reference (a stream's controls) with the last W stream points, following
two of the embedding drift methods compared in Evidently's "embedding drift detection" post
(https://www.evidentlyai.com/blog/embedding-drift-detection). Evidently's implementation is in
https://github.com/evidentlyai/evidently/blob/main/src/evidently/legacy/metrics/data_drift/embedding_drift_methods.py

- KS drifted share ("share of drifted components"; Evidently's `ratio` with
  component_stattest="ks"): each embedding dimension is tested on its own, and the statistic is
  the share of dimensions that drift. A dimension drifts when its two-sample KS p-value is
  <= alpha, with no multiple-testing correction, as in Evidently.
- Cosine distance (Evidently's `distance` with dist="cosine"): 1 - cosine similarity between the
  controls' mean and the window's mean.

Each detector emits only its statistic, and alarms are decided downstream.

Differences from Evidently:
  - Sliding window instead of two fixed batches. The window starts empty and fills with stream
    points, and the statistic is NaN for t < n_burn_in, as for the kernel baselines.
  - The per-dimension test is KS, Evidently's suggestion for small samples, rather than its
    default normed Wasserstein >= 0.1. At n = 30 per side the normed Wasserstein estimate is
    biased upward (median ~0.3 with no drift), so every dimension would always "drift".
  - No fixed drift-score thresholds. Thresholds are calibrated on null streams
    (null_calibration.py).
  - No PCA.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance
from scipy.stats import ks_2samp


class OnlineKSShareDetector:
    """Share of embedding dimensions whose KS test of controls vs the last W points rejects.

        share_t = mean_j 1[ KS p-value(controls[:, j], window[:, j]) <= alpha ]

    With no drift each dimension rejects with probability ~alpha, so the share sits near alpha.
    """

    def __init__(self, controls: np.ndarray, window_size: int, n_burn_in: int, alpha: float):
        assert 2 <= window_size <= n_burn_in, (
            f"need 2 <= window_size <= n_burn_in, got {window_size=} {n_burn_in=}")
        assert 0 < alpha < 1, f"need 0 < alpha < 1, got {alpha=}"
        self.controls = np.asarray(controls, dtype=float)
        self.window_size = window_size
        self.n_burn_in = n_burn_in
        self.alpha = alpha
        self.window = np.empty((0, self.controls.shape[1]))
        self.history_statistic: list[float] = []

    @property
    def t(self) -> int:
        """Number of stream observations consumed."""
        return len(self.history_statistic)

    def update(self, x_t: np.ndarray) -> float:
        """Push x_t into the window and return share_t (NaN during burn-in)."""
        self.window = np.vstack([self.window, np.asarray(x_t, dtype=float)[None, :]])
        self.window = self.window[-self.window_size:]
        statistic = float("nan") if self.t < self.n_burn_in else self._compute_drifted_share()
        self.history_statistic.append(statistic)
        return statistic

    def _compute_drifted_share(self) -> float:
        """Share of dimensions with KS p-value <= alpha over the full window."""
        assert len(self.window) == self.window_size, "window not full"
        p_values = ks_2samp(self.controls, self.window, axis=0).pvalue
        return float(np.mean(p_values <= self.alpha))


class OnlineCosineDetector:
    """Cosine distance between the controls' mean embedding and the last W points' mean.

        d_t = 1 - <mean(controls), mean(window)> / (||mean(controls)|| ||mean(window)||)
    """

    def __init__(self, controls: np.ndarray, window_size: int, n_burn_in: int):
        assert 2 <= window_size <= n_burn_in, (
            f"need 2 <= window_size <= n_burn_in, got {window_size=} {n_burn_in=}")
        self.controls_mean = np.asarray(controls, dtype=float).mean(0)
        assert np.linalg.norm(self.controls_mean) > 0, "controls' mean embedding is zero"
        self.window_size = window_size
        self.n_burn_in = n_burn_in
        self.window = np.empty((0, len(self.controls_mean)))
        self.history_statistic: list[float] = []

    @property
    def t(self) -> int:
        """Number of stream observations consumed."""
        return len(self.history_statistic)

    def update(self, x_t: np.ndarray) -> float:
        """Push x_t into the window and return d_t (NaN during burn-in)."""
        self.window = np.vstack([self.window, np.asarray(x_t, dtype=float)[None, :]])
        self.window = self.window[-self.window_size:]
        statistic = float("nan") if self.t < self.n_burn_in else self._compute_cosine_distance()
        self.history_statistic.append(statistic)
        return statistic

    def _compute_cosine_distance(self) -> float:
        """Cosine distance between the controls' mean and the full window's mean."""
        assert len(self.window) == self.window_size, "window not full"
        return float(distance.cosine(self.controls_mean, self.window.mean(0)))
