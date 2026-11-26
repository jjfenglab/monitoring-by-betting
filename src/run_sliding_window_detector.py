"""Run an online sliding-window baseline (MMD, LSDD, KS drifted share or cosine) on one stream."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_prior_monitoring.online_kernel_detectors import (
    OnlineLSDDDetector,
    OnlineMMDDetector,
)
from llm_prior_monitoring.online_kernel_detectors import (
    OnlineLSDDDetector,
    OnlineMMDDetector,
)
from run_ctm_monitor import load_stream_data

Detector = (
    OnlineMMDDetector
    | OnlineLSDDDetector
    | OnlineKSShareDetector
    | OnlineCosineDetector
)


def build_detector(args, controls: np.ndarray) -> Detector:
    """Construct the detector named by --method."""
    if args.method == "mmd":
        return OnlineMMDDetector(controls, args.window_size, args.n_burn_in)
    if args.method == "ks_share":
        return OnlineKSShareDetector(
            controls, args.window_size, args.n_burn_in, args.ks_alpha
        )
    if args.method == "cosine":
        return OnlineCosineDetector(controls, args.window_size, args.n_burn_in)
    assert args.n_centers is not None, "--n-centers is required for lsdd"
    return OnlineLSDDDetector(
        controls,
        args.window_size,
        args.n_burn_in,
        n_centers=args.n_centers,
        lambda_rd_max=args.lambda_rd_max,
        n_lambda_splits=args.n_lambda_splits,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online kernel two-sample baseline on one stream."
    )
    p.add_argument(
        "--method", choices=["mmd", "lsdd", "ks_share", "cosine"], required=True
    )
    p.add_argument("--window-size", type=int, required=True)
    p.add_argument("--n-burn-in", type=int, required=True)
    p.add_argument(
        "--n-centers",
        type=int,
        default=None,
        help="lsdd: number of controls used as kernel centres (the first ones)",
    )
    p.add_argument(
        "--lambda-rd-max",
        type=float,
        default=0.2,
        help="lsdd: largest lambda with mean relative difference below this",
    )
    p.add_argument(
        "--n-lambda-splits",
        type=int,
        default=200,
        help="lsdd: random reference/window splits for selecting lambda",
    )
    p.add_argument(
        "--ks-alpha",
        type=float,
        default=0.05,
        help="ks_share: a dimension drifts when its KS p-value is <= this",
    )
    p.add_argument("--stream-csv", type=Path, required=True)
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Monitor only the first this-many stream observations (None = all).",
    )
    p.add_argument("--seed", type=int, default=0)
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

    _, _, stream_emb, control_emb = load_stream_data(args)
    if args.max_steps is not None:
        assert (
            args.max_steps > args.n_burn_in
        ), f"max_steps {args.max_steps} leaves no post-burn-in steps"
        stream_emb = stream_emb[: args.max_steps]
    logging.info(
        "method=%s controls=%d stream=%d dim=%d window=%d n_burn_in=%d",
        args.method,
        len(control_emb),
        len(stream_emb),
        stream_emb.shape[1],
        args.window_size,
        args.n_burn_in,
    )

    detector = build_detector(args, control_emb)
    for x_t in stream_emb:
        detector.update(x_t)
    traj = np.asarray(detector.history_statistic, dtype=float)
    assert len(traj) == len(stream_emb)

    n_reference = len(control_emb) - (args.n_centers if args.method == "lsdd" else 0)
    sliding_window_config = {
        "method": args.method,
        "window_size": args.window_size,
        "n_burn_in": args.n_burn_in,
        "n_controls": len(control_emb),
        "n_reference": n_reference,
    }
    if args.method in ("mmd", "lsdd"):
        sliding_window_config["two_sigma2"] = detector.two_sigma2
        logging.info("kernel bandwidth two_sigma2=%.4g", detector.two_sigma2)
    result = {
        "args": vars(args),
        "sliding_window_config": sliding_window_config,
        "history": {
            "traj": traj,
            "n_steps_run": detector.t,
            "stopped_early": False,
        },
    }
    if args.method == "lsdd":
        result["lsdd"] = detector.selection_info()
        rd_at_lambda = result["lsdd"]["mean_rd"][
            result["lsdd"]["lambda_grid"].index(detector.lam)
        ]
        logging.info(
            "lsdd lambda=%.4g (mean RD %.3f < %.2f); mean RD per lambda: %s",
            detector.lam,
            rd_at_lambda,
            args.lambda_rd_max,
            np.round(result["lsdd"]["mean_rd"], 4).tolist(),
        )

    pd.DataFrame({"t": np.arange(len(traj)), "statistic": traj}).to_csv(
        args.debug_csv, index=False
    )
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    logging.info(
        "wrote %s (T=%d, max statistic=%.4g)", args.out, len(traj), np.nanmax(traj)
    )


if __name__ == "__main__":
    main()
