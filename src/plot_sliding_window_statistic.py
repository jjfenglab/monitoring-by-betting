"""Plot a sliding-window baseline's statistic over time from debug.csv."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

STATISTIC_LABELS = {
    "mmd": "MMD²",
    "lsdd": "LSDD",
    "ks_share": "KS drifted-dim share",
    "cosine": "Cosine distance",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a sliding-window statistic over time.")
    p.add_argument("--debug-csv", type=Path, required=True)
    p.add_argument("--method", choices=list(STATISTIC_LABELS), required=True)
    p.add_argument("--window-size", type=int, required=True)
    p.add_argument("--fig-path", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def plot_sliding_window_statistic(df: pd.DataFrame, method: str,
                                  window_size: int) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    label = STATISTIC_LABELS[method]
    ax.plot(df["t"], df["statistic"], color="tab:blue", lw=1.5)
    ax.set_xlim([0, df["t"].max() + 1])
    ax.set_xlabel("Stream index t")
    ax.set_ylabel(f"{label} (sliding window, W={window_size})")
    ax.set_title(f"Online {label} statistic over time")
    fig.tight_layout()
    return fig


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

    df = pd.read_csv(args.debug_csv).dropna(subset=["statistic"])  # burn-in steps are NaN
    assert len(df) > 0, "no rows with a statistic"

    fig = plot_sliding_window_statistic(df, args.method, args.window_size)
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_path, dpi=150)
    plt.close(fig)
    logging.info("wrote %s (%d rows)", args.fig_path, len(df))


if __name__ == "__main__":
    main()
