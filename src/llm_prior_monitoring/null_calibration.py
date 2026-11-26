"""Null-stream threshold calibration for statistics without a nominal ARL scale.

A detector that only emits a statistic (the sliding-window baselines) gets one threshold per nominal
ARL0 from null runs. ARL(h) is the mean post-burn-in null run length, with a no-alarm run
counted at its horizon len(traj) - n_burn_in. That is the convention of
metrics.null_run_lengths_for_threshold, and it matches the truncated EDD on shift streams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import null_run_length_matrix


def compute_record_thresholds(null_results: list[dict], n_burn_in: int) -> np.ndarray:
    """Sorted union of the running-max record values of the post-burn-in null trajectories.

    A run's first crossing of h only moves when h passes one of its running-max records. So
    ARL(h) is a step function, constant on (h_{k-1}, h_k] between consecutive records, and
    evaluating it at the records gives the whole curve exactly.
    """
    records = []
    for result in null_results:
        post_burn_in = np.asarray(result["res"]["history"]["traj"], dtype=float)[n_burn_in:]
        assert len(post_burn_in) and not np.isnan(post_burn_in).any(), (
            "post-burn-in null trajectory is empty or has NaN")
        running_max = np.maximum.accumulate(post_burn_in)
        is_record = np.r_[True, running_max[1:] > running_max[:-1]]
        records.append(running_max[is_record])
    return np.unique(np.concatenate(records))


def compute_null_arl_curve(null_results: list[dict], n_burn_in: int) -> pd.DataFrame:
    """ARL(h) and censoring diagnostics at every record threshold.

    Columns:
      threshold, arl: the censored mean run length after burn-in.
      censor_rate: the share of runs that never alarm (counted at their horizon).
      n_alarmed: the count of runs that alarm before their horizon.
    """
    thresholds = compute_record_thresholds(null_results, n_burn_in)
    lengths, censored = null_run_length_matrix(null_results, thresholds, n_burn_in)
    return pd.DataFrame({
        "threshold": thresholds,
        "arl": lengths.mean(0),
        "censor_rate": censored.mean(0),
        "n_alarmed": (1 - censored).sum(0).astype(int),
    })


def calibrate_threshold(arl_curve: pd.DataFrame, target_arl: float) -> dict:
    """Threshold for a nominal ARL0, interpolated between the two curve points bracketing it.

    The interpolated threshold lies in (h_{j-1}, h_j], where ARL is the constant arl_j >=
    target, so the calib_* fields are the in-sample values at that threshold. threshold_exact
    is h_j, the smallest record threshold reaching the target. Targets above the curve's
    maximum (every run censored) are unreachable and get NaN thresholds.

    Ties in arl happen when a run's last record falls on its final step, where alarm and no
    alarm give the same run length. They keep their smallest threshold, so interpolation sees
    a strictly increasing curve.
    """
    curve = arl_curve.drop_duplicates("arl", keep="first").reset_index(drop=True)
    arl = curve["arl"].to_numpy()
    thresholds = curve["threshold"].to_numpy()
    assert np.all(np.diff(arl) > 0), "ARL curve is not strictly increasing after de-duplication"

    row = {"nominal_arl": float(target_arl), "reachable": bool(target_arl <= arl[-1])}
    keys = ["threshold", "threshold_exact", "lower_threshold", "lower_arl", "upper_threshold",
            "upper_arl", "calib_arl", "calib_censor_rate"]
    if not row["reachable"]:
        return row | dict.fromkeys(keys, float("nan"))

    j = int(np.searchsorted(arl, target_arl, side="left"))
    if j == 0:
        threshold, lower_threshold, lower_arl = thresholds[0], float("nan"), float("nan")
    else:
        lower_threshold, lower_arl = thresholds[j - 1], arl[j - 1]
        threshold = float(np.interp(target_arl, arl[j - 1:j + 1], thresholds[j - 1:j + 1]))
    return row | {
        "threshold": float(threshold),
        "threshold_exact": float(thresholds[j]),
        "lower_threshold": float(lower_threshold),
        "lower_arl": float(lower_arl),
        "upper_threshold": float(thresholds[j]),
        "upper_arl": float(arl[j]),
        "calib_arl": float(arl[j]),
        "calib_censor_rate": float(curve["censor_rate"].iloc[j]),
    }
