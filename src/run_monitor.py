"""
Run one monitoring procedure on a provided data stream.
Can specify with or without LLM prior
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "HACHI"))

import numpy as np
import pandas as pd
import torch

from create_lgm_pipeline import (
    CHANGEPOINT_LEARNER_NAMES,
    CP_FEATURE_SOURCE_NAMES,
    make_sequential_concept_learner,
)
from llm_prior_monitoring.concept_lifecycle import (
    ConceptEpisodeRecorder,
    ConceptLifecycleRow,
    CsvRecorder,
    RemovalCandidateRow,
)
from llm_prior_monitoring.e_detector_sr import SymmetricSREDetector
from merge_llm_cache import merge_cache
from llm_prior_monitoring.monitor import AdaptiveMonitoring
from llm_prior_monitoring.scl_config import SCLConfig

# for logging purposes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one LLMGuidedMonitoring procedure on a pre-generated data stream."
    )
    p.add_argument("--seed", type=int, default=0)

    # SCL config (JSON file)
    p.add_argument("--scl-config", type=Path, required=True,
                   help="Path to SCL config JSON file")

    # E-detector args
    p.add_argument("--alpha", type=float, default=0.05, help="ARL0=1/alpha")
    p.add_argument("--lam", type=float, default=0.5, help="weight term in the e-process")
    p.add_argument("--max-m-statistic", type=float, default=None,
                   help="Stop monitoring once the M-statistic reaches this value "
                        "(LLM-cost cap, sibling of --max-steps; None = full stream). "
                        "The post-hoc threshold sweep stays valid only up to this "
                        "value -- aggregate_cell.py drops rows above it.")


    # Changepoint learner args
    p.add_argument("--changepoint-learner", required=True, choices=sorted(CHANGEPOINT_LEARNER_NAMES))
    p.add_argument("--cp-feature-source", default="embedding", choices=sorted(CP_FEATURE_SOURCE_NAMES))
    p.add_argument("--candidate-step", type=int, default=5, help="the grid for skimming over possible changepoints. smallest grid size is 1")
    p.add_argument("--min-block", type=int, default=5, help="minimum num of observations in the pre- and post- changepoint partitions")
    p.add_argument("--n-dirs", type=int, default=20, help="num SW directions checked")
    p.add_argument("--p", type=int, default=2, choices=[1, 2], help="wasserstein order")
    p.add_argument("--gamma", type=float, default=None, help="kernel gamma")

    # Data stream inputs
    p.add_argument("--stream-csv", type=Path, required=True)
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Monitor only the first this-many stream observations "
                        "(None = full stream). The caller owns the horizon: a deployed "
                        "monitor does not know whether its stream is null or shifted.")
    p.add_argument("--cache-file", type=Path,
                   help="Per-stream LLM cache for this run")
    # Output
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--debug-csv", type=Path, required=True)
    p.add_argument("--matrix-snapshot-dir", type=Path, default=None,
                   help="If set, write per-step annotated/imputed matrix CSVs here")
    p.add_argument("--sae-final-checkpoint", type=Path, default=None,
                   help="Online SAE only: save the end-of-stream SAE here (for post-hoc "
                        "auto-interp; the in-stream neurons drift)")
    # Opt-in lifecycle diagnostics: only exp_cebab declares these targets, so every
    # other experiment keeps calling this script unchanged.
    p.add_argument("--concept-lifecycle-csv", type=Path, default=None,
                   help="Per-(t, concept_id) state of the active concept set")
    p.add_argument("--concept-episodes-csv", type=Path, default=None,
                   help="Per-concept_id residency episodes; the concept-name dictionary")
    p.add_argument("--removal-csv", type=Path, default=None,
                   help="If set, write every scored removal candidate here")
    p.add_argument("--keyphrase-csv", type=Path, default=None,
                   help="bertopic_llm only: if set, dump the keyphrase-extraction frame here. "
                        "Per-run: concurrent streams must not share one path")
    args = p.parse_args()
    assert (args.concept_lifecycle_csv is None) == (args.concept_episodes_csv is None), (
        "--concept-lifecycle-csv and --concept-episodes-csv go together: lifecycle "
        "rows carry concept_id, which only resolves through the episodes file"
    )
    assert args.removal_csv is None or args.concept_episodes_csv is not None, (
        "--removal-csv needs --concept-episodes-csv to resolve its member_ids"
    )

    return args


def load_stream_data(args) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load stream data from pre-generated CSV + embeddings.npz."""
    full_df = pd.read_csv(args.stream_csv)
    all_emb = np.load(args.embeddings_path)["emb"]
    control_df = full_df[full_df["is_control"]].reset_index(drop=True)
    stream_df = full_df[~full_df["is_control"]].reset_index(drop=True)
    stream_emb = all_emb[stream_df["emb_idx"].values]
    control_emb = all_emb[control_df["emb_idx"].values]
    return stream_df, control_df, stream_emb, control_emb


def main() -> None:
    t_start = time.monotonic()
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log,
        # level=logging.DEBUG,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    # HACHI and lab_llm log on the root logger (module-level logging.* calls /
    # injected root logger), so DEBUG must be set at the root. Clamp noisy
    # third-party namespaces so only their INFO+ records get through.
    for noisy in ("urllib3", "httpx", "httpcore", "litellm", "openai", "anthropic",
                  "botocore", "matplotlib", "PIL", "asyncio", "filelock"):
        logging.getLogger(noisy).setLevel(logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("args: %s", vars(args))

    # Validate e-process / null-source consistency
    scl_config = SCLConfig.from_json(args.scl_config)
    logging.info("scl_config: %s", scl_config)

    stream_df, calib_df, stream_emb, calib_emb = load_stream_data(args)
    if args.max_steps is not None:
        assert args.max_steps > scl_config.n_burn_in, (
            f"max_steps {args.max_steps} leaves no post-burn-in steps"
        )
        stream_df = stream_df.iloc[:args.max_steps].reset_index(drop=True)
        stream_emb = stream_emb[:args.max_steps]
        logging.info("stream truncated to first %d observations", len(stream_df))

    e_detector = SymmetricSREDetector(
        alpha=args.alpha,
        lam=args.lam,
        n_burn_in=scl_config.n_burn_in,
        max_m_statistic=args.max_m_statistic,
    )

    embedding_dim = stream_emb.shape[1]
    # Build calib_df with required columns for SCL (text and y_true=0 for calibration)
    scl_calib_df = pd.DataFrame({
        "text": calib_df["text"].values,
        "y_true": np.zeros(len(calib_df), dtype=int),
    })
    scl = make_sequential_concept_learner(
        scl_config,
        embedding_dim,
        fixed_calibs=calib_emb,
        calib_df=scl_calib_df,
        min_block=args.min_block,
        cache_file=args.cache_file,
        keyphrase_outputs_file=(None if args.keyphrase_csv is None
                                else str(args.keyphrase_csv)),
    )

    args.debug_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.matrix_snapshot_dir is not None:
        args.matrix_snapshot_dir.mkdir(parents=True, exist_ok=True)

    lifecycle_recorder = None
    if args.concept_lifecycle_csv is not None:
        # Identity's "concepts" are embedding dimensions, so per-concept rows are
        # meaningless; the recorders still write headers, keeping the declared SCons
        # targets uniform across every cell of the grid.
        enabled = scl_config.concept_learner_type != "identity"
        lifecycle_recorder = CsvRecorder(
            args.concept_lifecycle_csv, ConceptLifecycleRow, enabled=enabled)
        scl.episode_recorder = ConceptEpisodeRecorder(
            args.concept_episodes_csv, scl.scl_variant, enabled=enabled)
        if args.removal_csv is not None:
            scl.removal_recorder = CsvRecorder(
                args.removal_csv, RemovalCandidateRow, enabled=enabled)

    monitor = AdaptiveMonitoring(
        sequential_concept_learner=scl,
        e_detector=e_detector,
        cp_feature_source=args.cp_feature_source,
        n_burn_in=scl_config.n_burn_in,
        debug_csv=args.debug_csv,
        matrix_snapshot_dir=args.matrix_snapshot_dir,
        concept_lifecycle_recorder=lifecycle_recorder,
    )

    for t, (structured_obs_t, row) in enumerate(zip(stream_emb, stream_df.itertuples())):
        monitor.update(
            structured_obs_t,
            {
                "text": str(row.text),
                "y_true": int(row.y_true) if row.y_true is not None else None, # only available for synthetic data
                "true_group": int(row.true_group) if row.true_group is not None else None, # only available for synthetic data
                "source_idx": int(row.source_idx),
            },
        )
        # break, not sys.exit: every artifact written below is a declared SCons target.
        if e_detector.force_stop():
            logging.info("EARLY STOP @ t=%d: M=%.3g >= max_m_statistic=%g (%d/%d steps)",
                         t, e_detector.MStatistic, args.max_m_statistic,
                         t + 1, len(stream_df))
            break

    if args.matrix_snapshot_dir is not None:
        assert monitor.snapshot_files, "matrix snapshot dir set but no snapshots written"
        pd.DataFrame({"file": monitor.snapshot_files}).to_csv(
            args.matrix_snapshot_dir / "manifest.csv", index=False
        )

    # Episodes are only complete once the stream ends (an episode still active
    # at the last step has removed_t = NaN), so this file is written once here.
    if scl.episode_recorder is not None:
        scl.episode_recorder.write()

    if args.sae_final_checkpoint is not None:
        monitor.scl.concept_projector.state.sae.save(str(args.sae_final_checkpoint))

    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = {
        "args": args_dict,
        "scl_config": asdict(scl_config),
        "history": {
            "traj": np.asarray(monitor.e_detector.history_MStatistic, dtype=float),
            "t_hat_history": list(monitor.t_hat_history),
            "n_steps_run": len(monitor.e_detector.history_MStatistic),
            "stopped_early": e_detector.force_stop(),
        },
        "all_concepts": monitor.scl.all_concepts,
        "final_sparse_concept_matrix": monitor.scl.sparse_concept_matrix.to_dense(),
        "final_coef": None if (coef := monitor.scl.linear_coef()) is None else np.ravel(coef),
        # distinct (row, concept) cells requested from the annotators, cache-independent
        "n_annotation_cells": {
            "history": monitor.scl.n_annotation_cells_history,
            "calib": monitor.scl.n_annotation_cells_calib,
            "xt": monitor.scl.n_annotation_cells_xt,
        },
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(result, f)
    logging.info("wrote %s", args.out)

    for c in monitor.scl.all_concepts:
        logging.info(f'FINAL CONCEPT {c["concept"]}')

    logging.info("total running time: %.1f s", time.monotonic() - t_start)


if __name__ == "__main__":
    main()
