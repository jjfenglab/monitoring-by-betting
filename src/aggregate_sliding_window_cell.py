"""Aggregate one sliding-window-baseline cell: score shift runs at null-calibrated thresholds.

Sibling of aggregate_ctm_cell.py, always on the skip-null path: the null streams were used
to calibrate the thresholds, so no empirical ARL0 is reported (empirical_arl is NaN).
compute_summary runs at the calibrated thresholds h, and each row's nominal_arl is then
set to the ARL0 target its threshold was calibrated for. `threshold` keeps h on the
statistic's own scale.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

from llm_prior_monitoring.metrics import compute_summary
from llm_prior_monitoring.utils import write_rows

CALIBRATION_COLS = ["threshold_exact", "calib_arl", "calib_censor_rate"]


def load_shift_results(res_glob: str, n_runs: int) -> list[dict]:
    """Merge each shift result.pkl matching res_glob with the stream_metadata.json that sits
    next to the stream_csv it ran on."""
    pkl_paths = sorted(Path(p) for p in glob.glob(res_glob))
    assert len(pkl_paths) == n_runs, f"{res_glob} matched {len(pkl_paths)} runs, expected {n_runs}"
    results = []
    for pkl_path in pkl_paths:
        with pkl_path.open("rb") as f:
            res = pickle.load(f)
        with (Path(res["args"]["stream_csv"]).parent / "stream_metadata.json").open() as f:
            meta = json.load(f)
        results.append(dict(meta) | {
            "stream_length": meta["n_pre"] + meta["n_post"],
            "t_change": meta["n_pre"],
            "res": res,
        })
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-cell summary for a sliding-window baseline.")
    p.add_argument("--shift-res-glob", required=True, help="glob for the shift result.pkls")
    p.add_argument("--n-shift-runs", type=int, required=True,
                   help="number of shift runs the glob must match")
    p.add_argument("--thresholds-csv", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--n-burn-in", type=int, required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--band-confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0, help="curve-bootstrap seed")
    p.add_argument("--k-calib", type=int, default=None, help="number of controls per stream")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("args: %s", vars(args))

    shift_results = load_shift_results(args.shift_res_glob, args.n_shift_runs)
    config = shift_results[0]["res"]["sliding_window_config"]
    assert all(r["res"]["sliding_window_config"]["method"] == config["method"]
               for r in shift_results)
    assert config["n_burn_in"] == args.n_burn_in, (
        f"runs used n_burn_in {config['n_burn_in']}, expected {args.n_burn_in}")
    logging.info("loaded %d shift runs (%s, W=%d)",
                 len(shift_results), config["method"], config["window_size"])
    if config["method"] == "lsdd":
        counts = Counter(r["res"]["lsdd"]["lambda"] for r in shift_results)
        logging.info("selected lambda over shift runs: %s",
                     {lam: n for lam, n in sorted(counts.items(), reverse=True)})

    calibration = pd.read_csv(args.thresholds_csv)
    reachable = calibration[calibration["reachable"]].reset_index(drop=True)
    logging.info("dropped %d unreachable nominal ARL0 targets (NaN thresholds); %d kept",
                 len(calibration) - len(reachable), len(reachable))
    assert len(reachable), "no reachable nominal ARL0 target"

    rows = compute_summary(
        label=args.label,
        null_results=[],
        shift_results=shift_results,
        thresholds=reachable["threshold"].to_numpy(),
        n_bootstrap=args.n_bootstrap,
        band_confidence=args.band_confidence,
        seed=args.seed,
        n_burn_in=args.n_burn_in,
    )
    for row, (_, cal) in zip(rows, reachable.iterrows()):
        row["nominal_arl"] = float(cal["nominal_arl"])
        row.update({col: float(cal[col]) for col in CALIBRATION_COLS})
        row.update({
            "sliding_window_method": config["method"],
            "window_size": config["window_size"],
            "n_reference": config["n_reference"],
            "k_calib": args.k_calib,
        })

    write_rows(rows, args.out_csv)
    logging.info("wrote %s (%d rows)", args.out_csv, len(rows))


if __name__ == "__main__":
    main()
