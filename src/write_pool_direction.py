"""Compute the empirical pool-mean-difference drift direction for GuidedScore.

The direction is mean(embeddings of group1 rows) - mean(embeddings of group0 rows),
unit-normalised: the realized shift axis in embedding space, an oracle upper bound for
the text-described directions from write_text_direction.py. Same npz schema (minus the
ic/oc provenance texts), so it plugs into the guided CTM unchanged.

Outputs:
  directions.npz : directions (1, d) unit row; labels (1,) for provenance
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

MIN_DIRECTION_NORM = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute the pool-mean-difference drift direction.")
    p.add_argument("--embeddings-path", type=Path, required=True,
                   help="npz with key 'emb', rows aligned with the mapping CSV.")
    p.add_argument("--mapping-csv", type=Path, required=True)
    p.add_argument("--group-col", type=str, default="split_label")
    p.add_argument("--group0", type=str, default="p0")
    p.add_argument("--group1", type=str, default="p1")
    p.add_argument("--label", type=str, required=True,
                   help="Provenance label stored in the npz.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
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

    emb = np.load(args.embeddings_path)["emb"]
    mapping = pd.read_csv(args.mapping_csv)
    assert len(mapping) == len(emb), (
        f"mapping rows ({len(mapping)}) != embedding rows ({len(emb)})"
    )

    mask0 = (mapping[args.group_col] == args.group0).values
    mask1 = (mapping[args.group_col] == args.group1).values
    assert mask0.sum() > 0 and mask1.sum() > 0, (
        f"empty group: {args.group0}={mask0.sum()}, {args.group1}={mask1.sum()}"
    )

    direction = emb[mask1].mean(axis=0) - emb[mask0].mean(axis=0)
    norm = np.linalg.norm(direction)
    assert norm > MIN_DIRECTION_NORM, f"group means (nearly) coincide: norm={norm}"
    direction = direction / norm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, directions=direction[None, :], labels=np.array([args.label]))
    logging.info("group sizes: %s=%d, %s=%d", args.group0, mask0.sum(),
                 args.group1, mask1.sum())
    logging.info("direction %s: pre-normalisation norm %.4f", args.label, norm)
    logging.info("wrote %s (k=1, dim=%d)", args.out, direction.shape[0])


if __name__ == "__main__":
    main()
