"""Anytime-valid sequential e-processes.

`EProcess` (ABC) + `SymmetricEProcess`. Each concrete subclass owns its per-step
`_compute_increment(p_t, p_null)`. Used as components inside `EDetector`
subclasses (see `e_detector.py` / `e_detector_sr.py`).

Pre-computed probabilities passed in from the monitor:
  p_t    = SCL real classifier P(y=1 | x_t)
  p_null = SCL.predict_proba_null() — the calibration pool's mean prediction,
           read at x_t's own per-row cell budget so both sides of the bet reach
           the classifier through the same annotation missingness
"""

from __future__ import annotations

import abc


class EProcess(abc.ABC):
    """Single anytime-valid e-process. Cumulative product of L_t increments.

    Subclasses implement `_compute_increment(x_t, *, concept_model, controls_t)`
    to define the per-step bet shape. The base class handles the multiplicative
    update `E *= L`, step counter, and threshold comparison via Ville's inequality.

    Concrete-subclass warm-up logic (e.g. classifier waiting for enough samples)
    lives inside `_compute_increment` and returns 1.0 while warming up.
    """

    def __init__(self, lam: float = 1.0, alpha_sig: float = 0.05, t_start: int = 0):
        assert 0 < alpha_sig < 1, f"alpha_sig must be in (0, 1), got {alpha_sig}"
        assert 0 < lam <= 1, f"lam must be in (0, 1], got {lam}"
        self.alpha_sig = alpha_sig
        self.lam = lam
        self.t_start = t_start
        self.threshold = 1.0 / alpha_sig
        self.hist_E = []
        self.hist_L = []
    
    @property
    def t(self):
        return len(self.hist_E)
    
    @property
    def E(self):
        if self.hist_E:
            return self.hist_E[-1]
        else:
            return 1

    @abc.abstractmethod
    def _compute_increment(
        self,
        p_t: float,
        p_null: float | None = None,
    ) -> float:
        """Per-step multiplicative factor L_t > 0. Return 1.0 during own warm-up."""

    def update(
        self,
        p_t: float,
        p_null: float | None = None,
    ) -> tuple[float, float]:
        L = self._compute_increment(p_t, p_null=p_null)
        assert L > 0, f"L must be positive, got {L}"
        new_E = self.E * L
        self.hist_E.append(new_E)
        self.hist_L.append(L)


class SymmetricEProcess(EProcess):
    """
    Bet: f_t = p_t - p_null;  L_t = 1 + lam * f_t.

    p_t    = P(y=1 | x_t) from classifier.
    p_null = expected value under the null

    Under H_0, expects E[f_t | F_{t-1}] = 0 (asymptotically)
    """

    def _compute_increment(
        self,
        p_t: float,
        p_null: float,
    ) -> float:
        return 1.0 + self.lam * (float(p_t) - float(p_null))