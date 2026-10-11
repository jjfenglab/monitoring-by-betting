"""Aggregate one method configuration: load null + shift pickles, sweep
thresholds via `compute_summary`, write `summary.csv`.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from llm_prior_monitoring.metrics import cap_thresholds_at_stop, compute_summary
from llm_prior_monitoring.utils import write_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-cell threshold sweep and summary.")
    p.add_argument("--null-res-glob", help="glob for null result.pkls (recursive **)")
    p.add_argument("--shift-res-globs", nargs="+", required=True, help="glob for shift result.pkls (recursive **)")
    p.add_argument("--null-dat-glob", help="glob for null stream_metadata.json (recursive **)")
    p.add_argument("--shift-dat-globs", nargs="+", required=True, help="glob for shift stream_metadata.json (recursive **)")
    p.add_argument("--skip-null", action="store_true",
                   help="This arm runs no null streams (SEED_CAPS null: 0): the ARL0 "
                        "columns come out NaN and the null globs must be omitted.")
    p.add_argument("--label", required=True)
    p.add_argument("--post-p1-fraction", type=float, default=float("nan"),
                   help="epsilon of the shift streams summarized here, when the grid "
                        "sweeps several and each is summarized on its own (NaN if pooled)")
    p.add_argument("--threshold-min", type=float, default=2.0)
    p.add_argument("--threshold-max", type=float, default=600.0)
    p.add_argument("--n-thresholds", type=int, default=35)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--band-confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0, help="curve-bootstrap seed")
    p.add_argument("--config-json", type=Path, required=True,
                   help="Path to scl_config.json for this cell")
    p.add_argument("--e-process", required=True)
    p.add_argument("--changepoint-learner", required=True)
    p.add_argument("--cp-feature-source", required=True)
    p.add_argument("--budget-profile", default="na",
                   help="Sweep-cell label for the remove_then_detect budget axis "
                        "(matches the bg_* dir); 'na' for detect_then_remove cells.")
    p.add_argument("--k-calib", type=int, default=None,
                   help="Calibration pool size (nested axis for sweeps)")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def _get_config_value(scl_config: dict, field: str):
    """Extract a field from the SCL config structure."""
    if field in scl_config:
        return scl_config[field]
    if field in scl_config.get("concept_proposer", {}):
        return scl_config["concept_proposer"][field]
    if field in scl_config.get("annotation_selector", {}):
        return scl_config["annotation_selector"][field]
    if field in scl_config.get("classifier", {}):
        return scl_config["classifier"][field]
    if field in scl_config.get("imputer", {}):
        return scl_config["imputer"][field]
    raise KeyError(f"Field '{field}' not found in scl_config")


def summarize_annotation_cost(results: list[dict], suffix: str) -> dict:
    """Mean per-seed annotation cost over one stream kind, as `<metric>_<suffix>` cols.

    `n_annotation_prompts` is the extractor's own counter, bumped before cache/API
    dispatch, so it exists for every arm but counts requests not spend.
    `n_annotation_cells_*` are the distinct (row, concept) cells requested per stream
    section, cache-independent: what a fresh cache would bill.
    """
    def _mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else float("nan")

    cells = [r["res"].get("n_annotation_cells") or {} for r in results]
    return {
        f"n_annotation_prompts_{suffix}": _mean(
            [r["res"].get("n_annotation_prompts") for r in results]),
        f"n_annotation_cells_history_{suffix}": _mean([c.get("history") for c in cells]),
        f"n_annotation_cells_calib_{suffix}": _mean([c.get("calib") for c in cells]),
        f"n_annotation_cells_xt_{suffix}": _mean([c.get("xt") for c in cells]),
    }


def _merge_per_seed(
    pkl_paths: list[Path],
    metadata_paths: list[Path],
) -> list[dict]:
    """Merge result.pkl and stream_metadata.json by seed."""
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

    # Dropping the null arm has to be stated, never inferred from an empty glob: a typo
    # in a null glob would otherwise pass silently as an uncalibrated summary.
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
    shift_pkls = [Path(p) for shift_gl in args.shift_res_globs for p in sorted(glob.glob(shift_gl, recursive=True))]
    shift_metas = [Path(p) for shift_gl in args.shift_dat_globs for p in sorted(glob.glob(shift_gl, recursive=True))]
    assert shift_pkls, f"no pkls match {args.shift_res_globs}"
    assert shift_metas, f"no metadata match {args.shift_dat_globs}"
    logging.info("shift res glob %s matched %d pkls", args.shift_res_globs, len(shift_pkls))
    # assert len(null_pkls) == len(null_metas)
    # assert len(shift_pkls) == len(shift_metas)

    with args.config_json.open() as f:
        scl_config = json.load(f)

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
        n_burn_in=scl_config["n_burn_in"],
    )

    # Removal variant for faceting. Arms that structurally never remove (identity,
    # sae, all_at_once oracle) get NaN so they replicate across removal facets as
    # baselines, like CTM rows; a removal-capable arm (mock, bertopic_llm,
    # random_pool oracle) run with rule "none" is a true ablation and keeps its
    # own "none" facet row -- a NaN there would replicate it into every panel
    # with a legend label identical to the panel's own series.
    # Greedy splits by group size, matching debug.csv's removal_variant.
    removal_rule = _get_config_value(scl_config, "removal_rule")
    greedy_group_size = scl_config.get("concept_proposer", {}).get("greedy_group_size", 1)
    clt = _get_config_value(scl_config, "concept_learner_type")
    oracle_mode = scl_config.get("oracle", {}).get("proposal_mode", "all_at_once")
    if clt in ("identity", "sae") or (clt == "oracle" and oracle_mode == "all_at_once"):
        removal_variant = float("nan")
    elif removal_rule == "greedy":
        removal_variant = "greedy_1" if greedy_group_size == 1 else "greedy_n"
    else:
        removal_variant = removal_rule  # "importance", "random", or "none"

    # Unified budget facet label: the two variants sweep different budget knobs
    # (annotation_rate vs budget_profile), so plots facet on one combined column.
    scl_variant = scl_config["scl_variant"]
    annotation_rate = 1.0
    
    # Null and shift streams run to different horizons, so their cost differs; keep
    # them apart rather than averaging over a mixture.
    config_cols = {
        **summarize_annotation_cost(null_results, "null"),
        **summarize_annotation_cost(shift_results, "shift"),
        "e_process": args.e_process,
        "changepoint_learner": args.changepoint_learner,
        "cp_feature_source": args.cp_feature_source,
        "concept_learner_type": _get_config_value(scl_config, "concept_learner_type"),
        # .get: configs written before the variant field existed default to the baseline.
        "scl_variant": scl_variant,
        "post_p1_fraction": args.post_p1_fraction,
        "budget_profile": args.budget_profile,
        "k_calib": args.k_calib,
        "annotation_rate": annotation_rate,
        "noise_rate": _get_config_value(scl_config, "noise_rate"),
        "classifier_type": _get_config_value(scl_config, "classifier_type"),
        "removal_variant": removal_variant,
        # Indexed directly: "variant" is generic enough that the flat section search in
        # _get_config_value could resolve it against the wrong section later.
        "imputer_variant": scl_config["imputer"]["variant"],
        # .get: configs of other arms written before the oracle section existed are not
        # rebuilt. NaN (not "") for non-oracle rows so _series_label in the plot drops it.
        "oracle_variant": scl_config.get("oracle", {}).get("variant") or float("nan"),
        # True | False for adaptive arms; NaN for non-adaptive so plots don't facet on it.
        "search_all_sizes": (
            scl_config["search_all_sizes"]
        ),
        # rows per step the annotation arms propose/scan/fit on, as text so the legend
        # reads "100" not "100.0"; NaN = full history (and configs predating the field),
        # so the plot label drops it.
        "history_sample_size": (
            str(hss) if (hss := scl_config.get("history_sample_size")) is not None
            else float("nan")
        ),
        # frozen | online for sae rows (configs predating the field are frozen runs);
        # NaN elsewhere so the plot's series label drops it.
        "sae_variant": (
            scl_config.get("sae", {}).get("variant", "frozen")
            if _get_config_value(scl_config, "concept_learner_type") == "sae"
            else float("nan")
        ),
    }
    for row in rows:
        row.update(config_cols)

    write_rows(rows, args.out_csv)
    logging.info("wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
