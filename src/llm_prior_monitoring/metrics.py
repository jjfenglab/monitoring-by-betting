"""Operating-characteristic metrics for the e-detector family.

All metrics operate on the per-seed trajectories loaded and regrouped by
`aggregate_cell.py` — they are post-hoc analyzers, not detectors. Most consume
threshold sweeps to produce ARL0 / EDD curves.

Grouped:

* Threshold-crossing primitives (`first_crossing`, `cap_thresholds_at_stop`).
* Null-stream run lengths (`null_run_lengths_for_threshold`).
* Shift-stream detection accounting (`detection_accounting`).
* Per-seed curve matrices for the curve bootstrap (`null_run_length_matrix`,
  `truncated_delay_matrix`).
* Summary rows (`compute_summary`).
"""

from __future__ import annotations


import numpy as np

from .utils import bootstrap_ci, bootstrap_curve_bands


def first_crossing(traj: np.ndarray, threshold: float) -> int | None:
    """Index of the first step where `traj` >= threshold, or None if never."""
    idx = np.where(traj >= threshold)[0]
    return int(idx[0]) if len(idx) else None


def cap_thresholds_at_stop(results: list[dict], thresholds: np.ndarray) -> np.ndarray:
    """Drop sweep thresholds above the binding --max-m-statistic across `results`.

    A run halted at M_t >= max_m_statistic gives exact run lengths for every threshold
    h <= max_m_statistic: the trajectory ends at or above h, so h's first crossing is
    inside the retained prefix and matches a full-length run exactly. Above that cap a
    crossing may have been cut off, and the callers below would silently score the seed
    as censored at len(traj) -- a horizon truncated by the early stop, not by the run
    reaching the end of its stream, which distorts ARL0 / no-alarm rates. Those
    thresholds are dropped rather than reported. Conversely, for every RETAINED
    threshold an early-stopped run always crosses, so it never takes a censored branch.

    Runs predating the flag carry no `max_m_statistic` and impose no cap.
    """
    caps = [
        result["res"]["args"].get("max_m_statistic")
        for result in results
    ]
    cap = min([c for c in caps if c is not None], default=float("inf"))
    kept = thresholds[thresholds <= cap]
    assert len(kept), (
        f"--max-m-statistic {cap} is below every swept threshold (min "
        f"{thresholds[0]:g}): the runs stopped before any reportable operating point"
    )
    return kept



def null_run_lengths_for_threshold(
    null_results: list[dict],
    threshold: float,
    n_burn_in: int,
) -> tuple[list[float], list[float]]:
    """Per-seed (run_length, is_censored) at one threshold, in post-burn-in process time.

    Burn-in steps are excluded from both branches: the e-process only starts betting at
    step n_burn_in, so its clock -- the one Ville's bound and hence `nominal_arl` refer to
    -- starts there too. Detecting on the first post-burn-in sample gives run_length = 1.

    The censored horizon is len(traj), the trajectory the run actually produced. That is
    exact per seed: it already reflects the run's --max-steps cap and any early stop, unlike
    `stream_length`, which is n_pre + n_post from the stream metadata and ignores both.
    """
    run_lengths = []
    censored = []
    for result in null_results:
        traj = result["res"]["history"]["traj"]
        crossing = first_crossing(traj, threshold)
        if crossing is None:
            run_lengths.append(float(len(traj) - n_burn_in))
            censored.append(1.0)
        else:
            assert crossing >= n_burn_in, (
                f"crossing at raw index {crossing} precedes burn-in {n_burn_in}: "
                "burn-in steps are NaN and cannot cross"
            )
            run_lengths.append(float(crossing + 1 - n_burn_in))
            censored.append(0.0)
    return run_lengths, censored


def detection_accounting(
    shift_results: list[dict],
    threshold: float,
) -> dict[str, float]:
    """Per-threshold pre/post alarm rates + conditional EDD on shift streams.

    EDD is a count of observations from the change to the alarm (crossing+1 -
    t_change): detecting on the first post-change sample gives EDD=1. This shares
    ARL0's crossing+1 convention, so the `EDD = ARL0 - N_PRE` reference line sits
    exactly on a zero-power detector.

    `conditional_edd` averages post-change alarms only; `post_or_censored_delay` (the
    truncated EDD that `compute_summary` reports as `edd`) also counts no-alarm runs at
    their horizon len(traj) - t_change. Both drop pre-change alarms.
    """
    pre = 0
    post = 0
    none = 0
    delays: list[int] = []
    post_or_censored_delays: list[int] = []
    for result in shift_results:
        traj = result["res"]["history"]["traj"]
        crossing = first_crossing(traj, threshold)
        t_change = result["t_change"]
        if crossing is None:
            none += 1
            # len(traj), not stream_length: the latter is n_pre + n_post from the stream
            # metadata and ignores the run's --max-steps cap, crediting post-change steps the
            # run never observed.
            post_or_censored_delays.append(len(traj) - t_change)
        elif crossing < t_change:
            pre += 1
        else:
            post += 1
            # EDD as a count (obs from change to alarm): +1 so it matches ARL0's
            # crossing+1 convention and the censored branch's len(traj)-t_change.
            # No n_burn_in term, unlike null_run_lengths_for_threshold: EDD is a
            # difference of two raw indices, so the offset cancels -- and t_change =
            # N_PRE > n_burn_in, so burn-in is over before the change.
            delay = crossing + 1 - t_change
            delays.append(delay)
            post_or_censored_delays.append(delay)
    n = len(shift_results)
    return {
        "pre_alarm_rate": pre / n,
        "post_alarm_rate": post / n,
        "no_alarm_rate": none / n,
        "conditional_edd": float(np.mean(delays)) if delays else float("nan"),
        "post_or_censored_delay": (
            float(np.mean(post_or_censored_delays))
            if post_or_censored_delays
            else float("nan")
        ),
    }


def null_run_length_matrix(
    null_results: list[dict],
    thresholds: np.ndarray,
    n_burn_in: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed (run_length, is_censored) curves, each [n_seed, n_threshold].

    One row per seed keeps each seed's whole curve together for the curve bootstrap.
    """
    per_threshold = [
        null_run_lengths_for_threshold(null_results, float(h), n_burn_in)
        for h in thresholds
    ]
    lengths = np.array([lengths for lengths, _ in per_threshold], dtype=float).T
    censored = np.array([censored for _, censored in per_threshold], dtype=float).T
    return lengths, censored


def truncated_delay_matrix(
    shift_results: list[dict],
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed (truncated delay, is_no_alarm) curves, each [n_seed, n_threshold].

    Same conventions as `detection_accounting`'s post_or_censored_delay: a post-change
    alarm gives crossing + 1 - t_change, no alarm gives len(traj) - t_change, and a
    pre-change alarm is NaN (dropped by the nan-mean). Column nan-means equal
    post_or_censored_delay.
    """
    delays = np.full((len(shift_results), len(thresholds)), np.nan)
    no_alarm = np.zeros((len(shift_results), len(thresholds)))
    for i, result in enumerate(shift_results):
        traj = result["res"]["history"]["traj"]
        t_change = result["t_change"]
        for j, h in enumerate(thresholds):
            crossing = first_crossing(traj, float(h))
            if crossing is None:
                delays[i, j] = len(traj) - t_change
                no_alarm[i, j] = 1.0
            elif crossing >= t_change:
                delays[i, j] = crossing + 1 - t_change
    return delays, no_alarm



def compute_summary(
    *,
    label: str,
    null_results: list[dict],
    shift_results: list[dict],
    thresholds: np.ndarray,
    n_bootstrap: int,
    n_burn_in: int,
    band_confidence: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Per-threshold summary with empirical ARL and detection metrics.

    Returns one dict per threshold. Each row contains:
    - nominal_arl: the threshold itself (= 1/α under e-process theory)
    - empirical_arl: observed mean POST-BURN-IN run length on null streams, the clock
      Ville's bound refers to and therefore the one comparable to nominal_arl; NaN when
      null_results is empty (an arm run shift-only)
    - empirical_arl_lo/hi: bootstrap CI for empirical_arl
    - edd: truncated EDD -- mean delay over shift runs that did not alarm pre-change,
      with no-alarm runs counted at their horizon len(traj) - t_change, so the
      denominator does not shrink with power (conditional_edd kept for reference)
    - detection accounting metrics at that threshold
    - empirical_arl_{pw,sim}_{lo,hi}, edd_{pw,sim}_{lo,hi}: pointwise and sup-t
      simultaneous seed-bootstrap bands over the threshold curve (`bootstrap_curve_bands`)
    """
    null_lengths_matrix, null_censored_matrix = null_run_length_matrix(
        null_results, thresholds, n_burn_in)
    delay_matrix, no_alarm_matrix = truncated_delay_matrix(shift_results, thresholds)
    band_kwargs = dict(n_bootstrap=n_bootstrap, confidence=band_confidence, seed=seed)
    all_valid = np.ones(len(thresholds), dtype=bool)
    arl_bands = bootstrap_curve_bands(null_lengths_matrix, all_valid, **band_kwargs)
    edd_bands = bootstrap_curve_bands(delay_matrix, all_valid, **band_kwargs)

    rows: list[dict] = []

    for j, threshold in enumerate(thresholds):
        null_lengths, null_censors = null_run_lengths_for_threshold(
            null_results, float(threshold), n_burn_in
        )
        # An arm run without null streams (SEED_CAPS null: 0) has no ARL0 to measure.
        empirical_arl = float(np.mean(null_lengths)) if null_lengths else float("nan")
        censor_rate = float(np.mean(null_censors)) if null_censors else float("nan")
        arl_lo, arl_hi = bootstrap_ci(null_lengths, n_bootstrap=n_bootstrap)

        acct = detection_accounting(shift_results, float(threshold))

        row = {
            "setting": label,
            "threshold": float(threshold),
            "nominal_arl": float(threshold),
            "empirical_arl": empirical_arl,
            "empirical_arl_lo": arl_lo,
            "empirical_arl_hi": arl_hi,
            "censor_rate": censor_rate,
            "n_burn_in": n_burn_in,
            "edd": acct["post_or_censored_delay"],
            **acct,
            **{f"empirical_arl_{key}": arl_bands[key][j]
               for key in ("pw_lo", "pw_hi", "sim_lo", "sim_hi")},
            **{f"edd_{key}": edd_bands[key][j]
               for key in ("pw_lo", "pw_hi", "sim_lo", "sim_hi")},
            "arl_sup_t_crit": arl_bands["crit"],
            "edd_sup_t_crit": edd_bands["crit"],
        }
        rows.append(row)
    return rows
