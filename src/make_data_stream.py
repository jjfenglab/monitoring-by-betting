"""Generate a data stream CSV from pre-embedded data or synthetic generation.

Supports two modes:
  1. Real data: loads embeddings.npz + mapping.csv, samples a stream
  2. Synthetic: generates Gaussian embeddings with controlled shift

Outputs:
  stream.csv : per-row metadata (t, emb_idx, text, y_true, true_group, etc.)
               Control rows come first (is_control=True, t=NaN), then stream rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_prior_monitoring.synthetic import (
    SyntheticConfig,
    sample_p0,
    sample_p1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a data stream CSV from embeddings or synthetic data."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pre-p1-fraction", type=float, default=0.0,
                   help="Fraction of p1 samples in pre-shift period")
    p.add_argument("--post-p1-fraction", type=float, default=1.0,
                   help="Fraction of p1 samples in post-shift period")

    p.add_argument("--n-controls", type=int, default=15,
                   help="Number of control rows (sampled from pre-change pool, 0 for none)")
    p.add_argument("--n-pre", type=int, default=100,
                   help="Number of pre-change rows in stream")
    p.add_argument("--n-post", type=int, default=250,
                   help="Number of post-change rows (0 for null/no-shift stream)")

    p.add_argument("--embeddings-path", type=Path, default=None,
                   help="Real mode: path to embeddings.npz")
    p.add_argument("--mapping-path", type=Path, default=None,
                   help="Real mode: path to mapping.csv")

    p.add_argument("--dim", type=int, default=20, help="Synthetic: embedding dimension")
    p.add_argument("--delta", type=float, default=0.0, help="Synthetic: shift magnitude")
    p.add_argument("--n-shift-features", type=int, default=3,
                   help="Synthetic: number of features that shift")
    p.add_argument("--shift-kind", choices=("sparse", "dense"), default="sparse")
    p.add_argument("--gradual-window", type=int, default=0,
                   help="Synthetic: window over which shift ramps up (0=abrupt)")

    p.add_argument("--out-stream-csv", type=Path, required=True)
    p.add_argument("--out-embeddings-path", type=Path, default=None,
                   help="Synthetic mode: output path for generated embeddings.npz")
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out-metadata", type=Path, default=None,
                   help="Output path for JSON metadata of args")
    return p.parse_args()


def make_real_stream(args, rng: np.random.Generator) -> pd.DataFrame:
    """Real-data mode: sample stream from pre-embedded data."""
    emb_data = np.load(args.embeddings_path)
    mapping_df = pd.read_csv(args.mapping_path)

    p0_df = mapping_df[mapping_df["split_label"] == "p0"]
    p1_df = mapping_df[mapping_df["split_label"] == "p1"]

    # Pre-shift period includes controls + pre-stream, all with same p0/p1 mix
    n_preshift_total = args.n_controls + args.n_pre
    n_preshift_p1 = int(n_preshift_total * args.pre_p1_fraction)
    n_preshift_p0 = n_preshift_total - n_preshift_p1

    # Post-shift period
    n_post_p1 = int(args.n_post * args.post_p1_fraction)
    n_post_p0 = args.n_post - n_post_p1

    # Total needed from each pool
    total_p0_needed = n_preshift_p0 + n_post_p0
    total_p1_needed = n_preshift_p1 + n_post_p1

    if total_p0_needed > len(p0_df):
        raise ValueError(
            f"Not enough p0 data: need {n_preshift_p0} preshift + {n_post_p0} post "
            f"= {total_p0_needed}, but only {len(p0_df)} available"
        )
    if total_p1_needed > len(p1_df):
        raise ValueError(
            f"Not enough p1 data: need {n_preshift_p1} preshift + {n_post_p1} post "
            f"= {total_p1_needed}, but only {len(p1_df)} available"
        )

    # Shuffle and partition p0 pool: preshift first, then post
    shuffled_p0_idx = rng.permutation(len(p0_df))
    preshift_p0_idx = shuffled_p0_idx[:n_preshift_p0]
    post_p0_idx = shuffled_p0_idx[n_preshift_p0:n_preshift_p0 + n_post_p0]

    # Shuffle and partition p1 pool: preshift first, then post
    shuffled_p1_idx = rng.permutation(len(p1_df))
    preshift_p1_idx = shuffled_p1_idx[:n_preshift_p1]
    post_p1_idx = shuffled_p1_idx[n_preshift_p1:n_preshift_p1 + n_post_p1]

    # Build preshift rows (controls + pre-stream), then partition
    preshift_p0_rows = p0_df.iloc[preshift_p0_idx].copy()
    preshift_p0_rows["true_group"] = 0
    preshift_p1_rows = p1_df.iloc[preshift_p1_idx].copy() if n_preshift_p1 > 0 else pd.DataFrame(columns=p1_df.columns)
    if n_preshift_p1 > 0:
        preshift_p1_rows["true_group"] = 1
    preshift_rows = pd.concat([preshift_p0_rows, preshift_p1_rows], ignore_index=True)
    preshift_rows = preshift_rows.iloc[rng.permutation(len(preshift_rows))]  # shuffle

    # Partition preshift into controls and pre-stream
    control_rows = preshift_rows.iloc[:args.n_controls]
    pre_rows = preshift_rows.iloc[args.n_controls:]

    if args.n_controls > 0:
        control_out = pd.DataFrame({
            "t": np.nan,
            "emb_idx": control_rows["row_index"].values,
            "text": control_rows["text"].values,
            "y_true": 0,
            "true_group": control_rows["true_group"].values,
            "source_idx": control_rows["source_idx"].values,
            "is_repeat": False,
            "is_control": True,
        })
    else:
        control_out = pd.DataFrame(columns=[
            "t", "emb_idx", "text", "y_true", "true_group", "source_idx", "is_repeat", "is_control"
        ])

    # Build post-shift rows (n_post=0 means null stream)
    if args.n_post > 0:
        post_p0_rows = p0_df.iloc[post_p0_idx].copy() if n_post_p0 > 0 else pd.DataFrame(columns=p0_df.columns)
        if n_post_p0 > 0:
            post_p0_rows["true_group"] = 0
        post_p1_rows = p1_df.iloc[post_p1_idx].copy() if n_post_p1 > 0 else pd.DataFrame(columns=p1_df.columns)
        if n_post_p1 > 0:
            post_p1_rows["true_group"] = 1
        post_rows = pd.concat([post_p0_rows, post_p1_rows], ignore_index=True)
        post_rows = post_rows.iloc[rng.permutation(len(post_rows))]  # shuffle
        stream_rows = pd.concat([pre_rows, post_rows], ignore_index=True)
    else:
        stream_rows = pre_rows.copy()

    stream_out = pd.DataFrame({
        "t": np.arange(len(stream_rows)),
        "emb_idx": stream_rows["row_index"].values,
        "text": stream_rows["text"].values,
        "y_true": (np.arange(len(stream_rows)) >= args.n_pre).astype(int),
        "true_group": stream_rows["true_group"].values,
        "source_idx": stream_rows["source_idx"].values,
        "is_repeat": False,
        "is_control": False,
    })

    out_df = pd.concat([control_out, stream_out], ignore_index=True)
    return out_df


def make_synthetic_stream(
    args, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic mode: generate Gaussian embeddings with controlled shift."""
    config = SyntheticConfig(
        dim=args.dim,
        n_shift_features=args.n_shift_features,
        shift_kind=args.shift_kind,
        gradual_window=args.gradual_window,
        n_pre=args.n_pre,
        n_post=args.n_post,
        n_controls=args.n_controls,
    )

    # Pre-shift period includes controls + pre-stream, all with same p0/p1 mix
    n_preshift_total = args.n_controls + args.n_pre
    n_preshift_p1 = int(n_preshift_total * args.pre_p1_fraction)
    n_preshift_p0 = n_preshift_total - n_preshift_p1

    preshift_p0_emb = sample_p0(rng, n_preshift_p0, config) if n_preshift_p0 > 0 else np.empty((0, args.dim))
    preshift_p1_emb = sample_p1(rng, n_preshift_p1, config, delta=args.delta) if n_preshift_p1 > 0 else np.empty((0, args.dim))
    preshift_true_group = np.array([0] * n_preshift_p0 + [1] * n_preshift_p1)

    # Shuffle preshift and partition into controls and pre-stream
    preshift_order = rng.permutation(n_preshift_total)
    preshift_emb = np.vstack([preshift_p0_emb, preshift_p1_emb])[preshift_order]
    preshift_true_group = preshift_true_group[preshift_order]

    control_emb = preshift_emb[:args.n_controls]
    control_true_group = preshift_true_group[:args.n_controls]
    pre_emb = preshift_emb[args.n_controls:]
    pre_true_group = preshift_true_group[args.n_controls:]

    if args.n_post > 0:
        # Post-shift
        n_post_p1 = int(args.n_post * args.post_p1_fraction)
        n_post_p0 = args.n_post - n_post_p1
        post_p0_emb = sample_p0(rng, n_post_p0, config) if n_post_p0 > 0 else np.empty((0, args.dim))
        post_p1_emb = sample_p1(rng, n_post_p1, config, delta=args.delta) if n_post_p1 > 0 else np.empty((0, args.dim))
        post_true_group = np.array([0] * n_post_p0 + [1] * n_post_p1)

        # Shuffle post-shift
        post_order = rng.permutation(args.n_post)
        post_emb = np.vstack([post_p0_emb, post_p1_emb])[post_order]
        post_true_group = post_true_group[post_order]

        stream_emb = np.vstack([pre_emb, post_emb])
        stream_true_group = np.concatenate([pre_true_group, post_true_group])
        stream_y_true = (np.arange(len(stream_emb)) >= args.n_pre).astype(int)
    else:
        stream_emb = pre_emb
        stream_true_group = pre_true_group
        stream_y_true = np.zeros(len(stream_emb), dtype=int)

    all_emb = np.vstack([control_emb, stream_emb]) if args.n_controls > 0 else stream_emb

    if args.n_controls > 0:
        control_out = pd.DataFrame({
            "t": np.nan,
            "emb_idx": np.arange(args.n_controls),
            "text": [f"control_{i}" for i in range(args.n_controls)],
            "y_true": 0,
            "true_group": control_true_group,
            "source_idx": -1,
            "is_repeat": False,
            "is_control": True,
        })
    else:
        control_out = pd.DataFrame(columns=[
            "t", "emb_idx", "text", "y_true", "true_group", "source_idx", "is_repeat", "is_control"
        ])

    stream_out = pd.DataFrame({
        "t": np.arange(len(stream_emb)),
        "emb_idx": np.arange(args.n_controls, args.n_controls + len(stream_emb)),
        "text": [f"obs_{i}" for i in range(len(stream_emb))],
        "y_true": stream_y_true,
        "true_group": stream_true_group,
        "source_idx": -1,
        "is_repeat": False,
        "is_control": False,
    })

    out_df = pd.concat([control_out, stream_out], ignore_index=True)
    return out_df, all_emb


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

    rng = np.random.default_rng(args.seed)

    is_real_mode = args.embeddings_path is not None
    if is_real_mode:
        assert args.mapping_path is not None, "--mapping-path required with --embeddings-path"
        out_df = make_real_stream(args, rng)
    else:
        assert args.out_embeddings_path is not None, (
            "--out-embeddings-path required for synthetic mode"
        )
        out_df, all_emb = make_synthetic_stream(args, rng)
        args.out_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_embeddings_path, emb=all_emb)
        logging.info("saved synthetic embeddings → %s", args.out_embeddings_path)

    args.out_stream_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_stream_csv, index=False)

    if args.out_metadata is not None:
        args.out_metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        with open(args.out_metadata, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info("saved metadata → %s", args.out_metadata)

    n_controls = out_df["is_control"].sum()
    n_stream = len(out_df) - n_controls
    logging.info(
        "saved stream.csv (controls=%d, stream=%d, t_change=%s) → %s",
        n_controls, n_stream, args.n_pre, args.out_stream_csv,
    )


if __name__ == "__main__":
    main()
