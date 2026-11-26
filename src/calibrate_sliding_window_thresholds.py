"""Calibrate sliding-window-baseline thresholds on null streams, one per nominal ARL0.

Writes three files:
  - null_arl_curve.csv: the full step curve ARL(h) with censoring diagnostics, for re-picking
    thresholds or trying other estimators later without rerunning anything.
  - thresholds.csv: one row per nominal ARL0 on the same grid our method sweeps.
  - null_lambda_counts.csv: the frequency of LSDD's per-stream lambda over the null runs.
    Header-only for every method but LSDD.
"""

from __future__ import annotations

import argparse
import glob
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from llm_prior_monitoring.null_calibration import calibrate_threshold, compute_null_arl_curve


def load_null_results(res_glob: str, n_runs: int, n_burn_in: int) -> list[dict]:
    """Wrap each null result.pkl matching res_glob as {"res": ...}, the shape metrics.py reads."""
    pkl_paths = sorted(Path(p) for p in glob.glob(res_glob))
    assert len(pkl_paths) == n_runs, f"{res_glob} matched {len(pkl_paths)} runs, expected {n_runs}"
    results = []
    for path in pkl_paths:
        with path.open("rb") as f:
            results.append({"res": pickle.load(f)})
    configs = {(r["res"]["sliding_window_config"]["method"],
                r["res"]["sliding_window_config"]["window_size"]) for r in results}
    assert len(configs) == 1, f"null runs mix sliding-window configurations: {configs}"
    burn_ins = {r["res"]["sliding_window_config"]["n_burn_in"] for r in results}
    assert burn_ins == {n_burn_in}, f"runs used n_burn_in {burn_ins}, expected {n_burn_in}"
    return results


def count_lambdas(results: list[dict]) -> pd.DataFrame:
    """Frequency of the selected LSDD lambda over runs (empty for every other method)."""
    columns = ["lambda", "n_streams", "fraction"]
    if "lsdd" not in results[0]["res"]:
        return pd.DataFrame(columns=columns)
    counts = Counter(r["res"]["lsdd"]["lambda"] for r in results)
    return pd.DataFrame(
        [(lam, n, n / len(results)) for lam, n in sorted(counts.items(), reverse=True)],
        columns=columns)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--null-res-glob", required=True, help="glob for the null result.pkls")
    p.add_argument("--n-null-runs", type=int, required=True,
                   help="number of null runs the glob must match")
    p.add_argument("--n-burn-in", type=int, required=True)
    p.add_argument("--threshold-min", type=float, required=True, help="smallest nominal ARL0")
    p.add_argument("--threshold-max", type=float, required=True, help="largest nominal ARL0")
    p.add_argument("--n-thresholds", type=int, required=True)
    p.add_argument("--out-curve-csv", type=Path, required=True)
    p.add_argument("--out-thresholds-csv", type=Path, required=True)
    p.add_argument("--out-lambda-counts-csv", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("args: %s", vars(args))

    null_results = load_null_results(args.null_res_glob, args.n_null_runs, args.n_burn_in)
    logging.info("loaded %d null runs (%s, W=%d)", len(null_results),
                 null_results[0]["res"]["sliding_window_config"]["method"],
                 null_results[0]["res"]["sliding_window_config"]["window_size"])

    lambda_counts = count_lambdas(null_results)
    if len(lambda_counts):
        logging.info("selected lambda over null runs:\n%s", lambda_counts.to_string(index=False))

    curve = compute_null_arl_curve(null_results, args.n_burn_in)
    logging.info("null ARL curve: %d record thresholds, ARL range [%.2f, %.2f]",
                 len(curve), curve["arl"].min(), curve["arl"].max())

    nominal_arls = np.linspace(args.threshold_min, args.threshold_max, args.n_thresholds)
    thresholds = pd.DataFrame([calibrate_threshold(curve, a) for a in nominal_arls])
    n_unreachable = int((~thresholds["reachable"]).sum())
    logging.info("%d/%d nominal ARL0 targets unreachable (above max ARL %.2f)",
                 n_unreachable, len(thresholds), curve["arl"].max())
    logging.info("calibration:\n%s", thresholds[
        ["nominal_arl", "threshold", "calib_arl", "calib_censor_rate"]
    ].to_string(index=False))

    for path, df in [(args.out_curve_csv, curve), (args.out_thresholds_csv, thresholds),
                     (args.out_lambda_counts_csv, lambda_counts)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logging.info("wrote %s (%d rows)", path, len(df))


if __name__ == "__main__":
    main()
