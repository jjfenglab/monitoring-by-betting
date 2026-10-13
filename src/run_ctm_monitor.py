"""Run the conformal test martingale baseline on one stream."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_prior_monitoring.conformal_martingale import (
    ConformalTestMartingaleMonitor,
    build_scorer,
)
from llm_prior_monitoring.ctm_config import CTMConfig


def load_stream_data(args) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split stream.csv into control rows and stream rows; index into embeddings.npz.

    Duplicated from run_monitor.py rather than imported: that module pulls in torch and
    HACHI at import time, and this baseline is deliberately independent of that stack.
    """
    full_df = pd.read_csv(args.stream_csv)
    all_emb = np.load(args.embeddings_path)["emb"]
    control_df = full_df[full_df["is_control"]].reset_index(drop=True)
    stream_df = full_df[~full_df["is_control"]].reset_index(drop=True)
    return (stream_df, control_df,
            all_emb[stream_df["emb_idx"].values],
            all_emb[control_df["emb_idx"].values])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conformal test martingale baseline.")
    p.add_argument("--ctm-config", type=Path, required=True)
    p.add_argument("--stream-csv", type=Path, required=True)
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-m-statistic", type=float, default=None,
                   help="Stop monitoring once the M-statistic reaches this value "
                        "(None = full stream). Matches run_monitor.py's flag so the "
                        "SCL and CTM sweeps span the same threshold range.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Monitor only the first this-many stream observations "
                        "(None = full stream). The caller owns the horizon: a deployed "
                        "monitor does not know whether its stream is null or shifted.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--debug-csv", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.info("args: %s", vars(args))
    np.random.seed(args.seed)

    config = CTMConfig.from_json(args.ctm_config)
    logging.info("ctm_config: %s", config)

    stream_df, control_df, stream_emb, control_emb = load_stream_data(args)

    if args.max_steps is not None:
        assert args.max_steps > config.n_burn_in, (
            f"max_steps {args.max_steps} leaves no post-burn-in steps"
        )
        stream_df = stream_df.iloc[:args.max_steps].reset_index(drop=True)
        stream_emb = stream_emb[:args.max_steps]
        logging.info("stream truncated to first %d observations", len(stream_df))

    assert 0 < config.n_reference < len(control_emb), (
        f"n_reference={config.n_reference} must leave a non-empty calibration set "
        f"(n_controls={len(control_emb)})"
    )
    reference = control_emb[:config.n_reference]
    calibration = control_emb[config.n_reference:]
    logging.info("reference=%d calibration=%d stream=%d dim=%d n_burn_in=%d",
                 len(reference), len(calibration), len(stream_emb), stream_emb.shape[1],
                 config.n_burn_in)

    directions = None
    if config.direction_path is not None:
        directions = np.load(config.direction_path)["directions"]
        logging.info("loaded %d direction(s) from %s", len(directions), config.direction_path)

    # Anisotropy diagnostic, not a gate: a large spread means the cosine is dominated by the
    # widest coordinates and `standardize` is warranted. all-MiniLM-L6-v2 measures ~2.8x.
    # max/min is useless here -- one MiniLM coordinate is dead, giving a ratio of ~1e31.
    pcts = np.percentile(reference.std(axis=0), [1, 50, 99])
    logging.info("reference coord std p01/p50/p99 = %s", pcts.round(5))

    monitor = ConformalTestMartingaleMonitor(
        reference=reference,
        calibration=calibration,
        scorer=build_scorer(config.scorer, directions, standardize=config.standardize),
        jump_rates=config.jump_rates,
        alpha=config.alpha,
        use_sr=config.use_sr,
        max_m_statistic=args.max_m_statistic,
        n_burn_in=config.n_burn_in,
    )
    for t, x_t in enumerate(stream_emb):
        monitor.update(x_t)
        if monitor.force_stop():
            logging.info("EARLY STOP @ t=%d: M=%.3g >= max_m_statistic=%g (%d/%d steps)",
                         t, monitor.MStatistic, args.max_m_statistic,
                         t + 1, len(stream_emb))
            break

    traj = np.asarray(monitor.history_MStatistic, dtype=float)
    assert len(traj) == len(stream_emb) or monitor.force_stop()

    args.debug_csv.parent.mkdir(parents=True, exist_ok=True)
    # `score` is a NONCONFORMITY score (higher = stranger) and `p_value` follows
    # Vovk2021-Retrain eq. (1), so strange observations score p near 1. Outputs written before
    # that convention switch hold the complement, 1 - p; M_statistic is unaffected.
    pd.DataFrame({
        "t": np.arange(monitor.t),
        "M_statistic": traj,  # the two columns plot_e_statistic.py reads
        "score": monitor.history_score,
        "p_value": monitor.history_pvalue,
        "L_t": monitor.history_increment,
    }).to_csv(args.debug_csv, index=False)

    result = {
        "args": vars(args),
        "ctm_config": asdict(config),  # mirrors run_monitor.py's "scl_config" entry
        "history": {
            "traj": traj,
            "p_values": np.asarray(monitor.history_pvalue, dtype=float),
            "scores": np.asarray(monitor.history_score, dtype=float),
            "n_steps_run": monitor.t,
            "stopped_early": monitor.force_stop(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    logging.info("wrote %s (T=%d, final M=%.4g, rejected=%s)",
                 args.out, monitor.t, traj[-1], monitor.is_rejected())


if __name__ == "__main__":
    main()
