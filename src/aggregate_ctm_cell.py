"""Aggregate one CTM configuration: load null + shift pickles, sweep thresholds via
`compute_summary`, write `summary.csv`.

Sibling of aggregate_cell.py. The metrics transfer unchanged -- compute_summary reads only
res["history"]["traj"] -- but the descriptive columns do not: a CTM run has none of the six
SCL config fields aggregate_cell.py pulls, and carries scorer/jump_rates instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from llm_prior_monitoring.ctm_config import CTMConfig
from llm_prior_monitoring.metrics import cap_thresholds_at_stop, compute_summary
from llm_prior_monitoring.utils import write_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-cell threshold sweep and summary (CTM).")
    p.add_argument("--null-res-glob", help="glob for null result.pkls (recursive **)")
    p.add_argument("--shift-res-globs", nargs="+", required=True, help="glob for shift result.pkls (recursive **)")
    p.add_argument("--null-dat-glob", help="glob for null stream_metadata.json (recursive **)")
    p.add_argument("--shift-dat-globs", nargs="+", required=True, help="glob for shift stream_metadata.json (recursive **)")
    p.add_argument("--skip-null", action="store_true",
                   help="No null streams: ARL0 columns come out NaN and null globs must be omitted.")
    p.add_argument("--label", required=True)
    p.add_argument("--threshold-min", type=float, default=2.0)
    p.add_argument("--threshold-max", type=float, default=600.0)
    p.add_argument("--n-thresholds", type=int, default=35)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--band-confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0, help="curve-bootstrap seed")
    p.add_argument("--config-json", type=Path, required=True,
                   help="Path to ctm_config.json for this cell")
    p.add_argument("--k-calib", type=int, default=None,
                   help="Calibration pool size (nested axis for sweeps)")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def _merge_per_seed(pkl_paths: list[Path], metadata_paths: list[Path]) -> list[dict]:
    """Merge result.pkl and stream_metadata.json by seed (same shape as aggregate_cell)."""
    match_num = min(len(metadata_paths), len(pkl_paths))
    results = []
    for meta_path, pkl_path in zip(metadata_paths[:match_num], pkl_paths[:match_num]):
        with meta_path.open() as f:
            meta = json.load(f)
        with pkl_path.open("rb") as f:
            result_obj = pickle.load(f)
        merged_dict = dict(meta)
        merged_dict["stream_length"] = meta["n_pre"] + meta["n_post"]
        merged_dict["t_change"] = meta["n_pre"]
        merged_dict["res"] = result_obj
        results.append(merged_dict)
    return results


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("args: %s", vars(args))

    if args.skip_null:
        assert args.null_res_glob is None and args.null_dat_glob is None, \
            "--skip-null takes no null globs"
        null_pkls, null_metas = [], []
    else:
        assert args.null_res_glob and args.null_dat_glob, \
            "null globs are required without --skip-null"
        null_pkls = [Path(p) for p in sorted(glob.glob(args.null_res_glob, recursive=True))]
        null_metas = [Path(p) for p in sorted(glob.glob(args.null_dat_glob, recursive=True))]
        assert null_pkls, f"no pkls match {args.null_res_glob}"
        assert null_metas, f"no metadata match {args.null_dat_glob}"
        logging.info("null res glob %s matched %d pkls", args.null_res_glob, len(null_pkls))
        # assert len(null_pkls) == len(null_metas)
    shift_pkls = [Path(p) for gl in args.shift_res_globs for p in sorted(glob.glob(gl, recursive=True))]
    shift_metas = [Path(p) for gl in args.shift_dat_globs for p in sorted(glob.glob(gl, recursive=True))]
    assert shift_pkls, f"no pkls match {args.shift_res_globs}"
    assert shift_metas, f"no metadata match {args.shift_dat_globs}"
    logging.info("shift res glob %s matched %d pkls", args.shift_res_globs, len(shift_pkls))
    # assert len(shift_pkls) == len(shift_metas)

    config = CTMConfig.from_json(args.config_json)
    logging.info("ctm_config: %s", config)

    null_results = _merge_per_seed(null_pkls, null_metas)
    shift_results = _merge_per_seed(shift_pkls, shift_metas)
    assert args.skip_null or null_results, "no null pickles loaded"
    assert shift_results, "no shift pickles loaded"

    # Log alarm rates at the minimum threshold (loosest bound)
    def _alarm_rate_at_threshold(results: list[dict], threshold: float) -> float:
        n_alarmed = sum(
            1 for r in results
            if np.nanmax(r["res"]["history"]["traj"]) >= threshold
        )
        return n_alarmed / len(results) if results else float("nan")

    if null_results:
        logging.info(
            "null alarm rate at threshold=%.2f: %.1f%% (%d/%d seeds)",
            args.threshold_max,
            100 * _alarm_rate_at_threshold(null_results, args.threshold_max),
            sum(1 for r in null_results if np.nanmax(r["res"]["history"]["traj"]) >= args.threshold_max),
            len(null_results),
        )
    logging.info(
        "shift alarm rate at threshold=%.2f: %.1f%% (%d/%d seeds)",
        args.threshold_max,
        100 * _alarm_rate_at_threshold(shift_results, args.threshold_max),
        sum(1 for r in shift_results if np.nanmax(r["res"]["history"]["traj"]) >= args.threshold_max),
        len(shift_results),
    )

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.n_thresholds)
    kept = cap_thresholds_at_stop(null_results + shift_results, thresholds)
    if len(kept) < len(thresholds):
        logging.info("dropped %d/%d sweep thresholds above the binding "
                     "--max-m-statistic (max kept %.4g)",
                     len(thresholds) - len(kept), len(thresholds), kept[-1])
    thresholds = kept

    rows = compute_summary(
        label=args.label,
        null_results=null_results,
        shift_results=shift_results,
        thresholds=thresholds,
        n_bootstrap=args.n_bootstrap,
        band_confidence=args.band_confidence,
        seed=args.seed,
        n_burn_in=config.n_burn_in,
    )

    # Native CTM columns only -- nothing faked to look like an SCL run. plot_exp_fakenotes
    # builds the CTM method label from scorer/jump_rates.
    config_cols = {
        "scorer": config.scorer,
        "variant": config.variant,
        "jump_rates": ",".join(map(str, config.jump_rates)),
        "alpha": config.alpha,
        "n_reference": config.n_reference,
        "k_calib": args.k_calib,
        "use_sr": config.use_sr,
        "direction_path": config.direction_path,
        "standardize": config.standardize,
    }
    for row in rows:
        row.update(config_cols)

    write_rows(rows, args.out_csv)
    logging.info("wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
