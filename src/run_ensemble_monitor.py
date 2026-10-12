"""
Ensemble monitor: combines two methods' monitors after the fact.

Reads debug.csv files from two source methods (e.g., bertopic_llm and identity) and
either averages their per-step p-values and recomputes M_statistic (--combine p_value),
or averages their M_statistic trajectories (--combine m_statistic). Outputs a new
debug.csv and result.pkl compatible with the aggregation pipeline.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble two monitors.")
    p.add_argument("--combine", choices=["p_value", "m_statistic"], required=True)
    p.add_argument("--debug-csv-1", type=Path, required=True, help="First method's debug.csv")
    p.add_argument("--debug-csv-2", type=Path, required=True, help="Second method's debug.csv")
    p.add_argument("--lam", type=float, default=0.5, help="Lambda for e-process betting (p_value)")
    p.add_argument("--max-m-statistic", type=float, default=250.0, help="Early stop threshold")
    p.add_argument("--stream-length", type=int, required=True,
                   help="n_pre + n_post; under --combine m_statistic a run that never "
                        "reaches --max-m-statistic must cover the whole stream (asserted)")
    p.add_argument("--out-debug-csv", type=Path, required=True)
    p.add_argument("--out-pkl", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def average_p_values(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    lam: float,
    max_m_statistic: float,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Average p-values and recompute M_statistic trajectory."""
    max_len = min(len(df1), len(df2))

    ensemble_df = pd.DataFrame({
        "t": df1["t"].iloc[:max_len].values,
        "p_t": 0.5 * (df1["p_t"].iloc[:max_len].values + df2["p_t"].iloc[:max_len].values),
        "p_null": 0.5 * (df1["p_null"].iloc[:max_len].values + df2["p_null"].iloc[:max_len].values),
        "true_group": df1["true_group"].iloc[:max_len].values,
    })

    e_processes = np.array([])
    m_statistics = np.full(max_len, np.nan)
    stopped_early = False

    for i in range(max_len):
        if np.isnan(ensemble_df["p_t"].iloc[i]):
            continue
        e_processes = np.concatenate([e_processes, np.array([1.0])])
        bettor_return = 1.0 + lam * (ensemble_df["p_t"].iloc[i] - ensemble_df["p_null"].iloc[i])
        e_processes = e_processes * bettor_return
        m_statistics[i] = np.sum(e_processes)

        if m_statistics[i] >= max_m_statistic:
            stopped_early = True
            break

    ensemble_df["M_statistic"] = m_statistics
    return ensemble_df, m_statistics, stopped_early


def average_m_statistics(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    max_m_statistic: float,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Average the two e-detectors' M_statistic (itself an e-detector), zero-padding the shorter input.

    Once either input reaches its cap c, the average is >= c / 2, so every threshold up to
    half the smaller input cap is crossed before the trajectory ends.
    """
    max_len = max(len(df1), len(df2))
    m1 = np.concatenate([df1["M_statistic"].to_numpy(float), np.zeros(max_len - len(df1))])
    m2 = np.concatenate([df2["M_statistic"].to_numpy(float), np.zeros(max_len - len(df2))])
    assert np.array_equal(np.isnan(m1), np.isnan(m2)), "inputs disagree on burn-in steps"
    m_statistics = 0.5 * (m1 + m2)

    crossed = np.where(m_statistics >= max_m_statistic)[0]
    stopped_early = len(crossed) > 0
    if stopped_early:
        # m_statistics[crossed[0] + 1:] = np.nan
        max_len = crossed[0] + 1

    ensemble_df = pd.DataFrame({
        "t": np.arange(max_len),
        "M_statistic_1": m1[:max_len],
        "M_statistic_2": m2[:max_len],
        # "true_group": df1["true_group"].to_numpy()[:max_len],
        "M_statistic": m_statistics[:max_len],
    })
    return ensemble_df, m_statistics, stopped_early


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

    df1 = pd.read_csv(args.debug_csv_1)
    df2 = pd.read_csv(args.debug_csv_2)
    logging.info("Loaded %d rows from %s", len(df1), args.debug_csv_1)
    logging.info("Loaded %d rows from %s", len(df2), args.debug_csv_2)

    if args.combine == "p_value":
        ensemble_df, m_traj, stopped_early = average_p_values(
            df1, df2, args.lam, args.max_m_statistic
        )
    else:
        ensemble_df, m_traj, stopped_early = average_m_statistics(
            df1, df2, args.max_m_statistic
        )
        # A run that never reaches the cap must cover the whole stream, so the EDD code
        # censors it at the stream end rather than where an input stopped.
        assert stopped_early or len(m_traj) == args.stream_length

    assert len(m_traj) <= args.stream_length
    logging.info("inputs cover %d of %d steps", len(m_traj), args.stream_length)

    args.out_debug_csv.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(args.out_debug_csv, index=False)
    logging.info("Wrote ensemble debug.csv with %d rows", len(ensemble_df))
    result = {
        "args": {"max_m_statistic": args.max_m_statistic},
        "history": {
            "traj": m_traj,
            "n_steps_run": len(m_traj),
            "stopped_early": stopped_early,
        },
        "llm_usage": {},
    }
    with args.out_pkl.open("wb") as f:
        pickle.dump(result, f)
    logging.info("Wrote result.pkl")


if __name__ == "__main__":
    main()
