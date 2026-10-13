"""Plot the cumulative e-detector statistic M_t over time from debug.csv."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot M_statistic over time from debug.csv.")
    p.add_argument("--debug-csv", type=Path, required=True)
    p.add_argument("--fig-path", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def plot_m_statistic(df: pd.DataFrame) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["t"], df["M_statistic"], color="tab:blue", lw=1.5)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="M = 1 (null)")
    ax.set_xlim([0, df["t"].max() + 1])
    ax.set_xlabel("Stream index t")
    ax.set_ylabel("M_statistic (cumulative e-value)")
    ax.set_title("E-detector statistic over time")
    ax.legend(loc="upper left")

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

    df = pd.read_csv(args.debug_csv)
    df = df.dropna(subset=["M_statistic"])
    assert len(df) > 0, "no rows with M_statistic"

    fig = plot_m_statistic(df)
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_path, dpi=150)
    plt.close(fig)
    logging.info("wrote %s (%d rows)", args.fig_path, len(df))


if __name__ == "__main__":
    main()
