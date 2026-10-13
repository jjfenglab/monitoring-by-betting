"""Concrete Shiryaev-Roberts e-detectors.

* `SymmetricSREDetector`       — SR shortcut + symmetric two-sample bet.
"""

from __future__ import annotations

from .e_detector import SharedLEDetector
from .e_process import (
    SymmetricEProcess,
)


class SymmetricSREDetector(SharedLEDetector):
    """SR e-detector driven by the Shekhar-Ramdas symmetric two-sample bet.

    Calibration-free: under H_0 x_t and the controls are i.i.d. given the
    predictable classifier, so E[f_t | F_{t-1}] = 0 for ANY classifier. No
    `beta_t` calibration condition. Component-independent L_t means the SR
    shortcut `M_t = (1 + M_{t-1}) * L_t` is exact, not an argmax-tau
    approximation.

    LGM passes p_t = SCL.predict_proba(x_t) and p_null = SCL.predict_proba_null()
    each step. How the calibration pool is read is an SCL concern, not a detector one.
    """

    eprocess_cls = SymmetricEProcess