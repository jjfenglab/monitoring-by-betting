"""E-detector ABCs: mixture aggregators of `EProcess` components.

An e-detector maintains a list of e-process components (`components`) and an
aggregate statistic M_t = sum_j Lambda^(j)_t. Two-level ABC:

* `EDetector`  : shared state + concrete accessors + abstract `update`.
* `SharedLEDetector(EDetector)` : every component shares the per-step L_t. Each
  post-burn-in step births one e-process and M_t is the sum of their E values,
  which equals the Shiryaev-Roberts recursion M_t = (1 + M_{t-1}) * L_t.
  Concrete subclasses set `eprocess_cls`.

Concrete detectors live in `e_detector_sr.py`.
"""

from __future__ import annotations

import abc
import logging
from .e_process import EProcess


class EDetector(abc.ABC):
    """
    Mixture aggregator of EProcess components

    Two unrelated levels on M_t, easily confused:

    * `threshold` = 1/alpha, the nominal Ville rejection level. Read only by
      `is_rejected()` and the `update` return flag, both of which reach nothing but
      log lines -- every reported operating characteristic comes from the post-hoc
      threshold sweep in `metrics.py` instead.
    * `max_m_statistic`, an optional cost cap driving `force_stop()`: the driver
      halts the stream once M_t reaches it, saving LLM calls on the real arm. It
      bounds how far that post-hoc sweep stays valid, so keep it at or above the
      sweep's ceiling (`aggregate_cell.py` drops thresholds above it).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        lam: float = 0.5,
        n_burn_in: int = 3,
        max_m_statistic: float | None = None,
    ):
        assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
        assert 0 < lam <= 1, f"lam must be in (0, 1], got {lam}"
        assert n_burn_in >= 1, f"n_burn_in must be >= 1, got {n_burn_in}"
        assert max_m_statistic is None or max_m_statistic > 1, (
            f"max_m_statistic must exceed the null level M=1, got {max_m_statistic}"
        )
        self.alpha = alpha
        self.lam = lam
        self.n_burn_in = n_burn_in
        self.threshold = 1.0 / alpha
        self.max_m_statistic = max_m_statistic
        self.components: list[EProcess] = []
        self.history_MStatistic: list[float] = []
        self.history_increment: list[float] = []
    
    @property
    def t(self):
        return len(self.history_MStatistic)

    @property
    def MStatistic(self):
        return self.history_MStatistic[-1]

    @abc.abstractmethod
    def update(
        self,
        p_t: float | None = None,
        p_null: float | None = None,
    ) -> tuple[float, bool]:
        """Step forward by one observation. Returns (M_t, reject).

        p_t    = P(y=1 | x_t) from SCL's real classifier.
        p_null = SCL.predict_proba_null(), the calibration pool's mean prediction.
        """

    def is_rejected(self) -> bool:
        """Return True iff M_t >= self.threshold."""
        return self.MStatistic >= self.threshold

    def force_stop(self) -> bool:
        """Return True iff the run may halt: M_t >= max_m_statistic.
        """
        return (
            self.max_m_statistic is not None
            and self.t > 0
            and self.MStatistic >= self.max_m_statistic
        )


class SharedLEDetector(EDetector):
    """
    SR e-detector where all components share the same per-step L_t.
    """

    eprocess_cls: type[EProcess]

    def update(
        self,
        p_t: float | None = None,
        p_null: float | None = None,
    ) -> tuple[float, bool]:
        if self.t < self.n_burn_in:
            self.history_MStatistic.append(float("nan"))
            self.history_increment.append(float("nan"))
        else:
            # Birth a new e-process
            new_eproc = self.eprocess_cls(lam=self.lam, alpha_sig=self.alpha, t_start=self.t)
            self.components.append(new_eproc)
            
            # update each e-process
            for e_proc in self.components:
                e_proc.update(p_t, p_null=p_null)
            
            # update the e-detector
            curr_MStatistic = sum(c.E for c in self.components)

            self.history_MStatistic.append(curr_MStatistic)
            self.history_increment.append(e_proc.hist_L[-1])
            logging.info(f"EDETECTOR STATISTIC {self.t}: {curr_MStatistic}")
            return curr_MStatistic, curr_MStatistic >= self.threshold
