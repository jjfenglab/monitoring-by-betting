"""Embed IC/OC description pairs into unit drift directions for GuidedScore.

A direction is embed(OC description) - embed(IC description): the axis in embedding space
along which the anticipated shift moves observations. Precomputed here rather than inside
run_ctm_monitor.py so the monitor never imports torch, and so the direction is a cacheable,
inspectable artifact.

Outputs:
  directions.npz : directions (k, d) unit rows; labels, ic_texts, oc_texts (k,) for provenance
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import embed

MIN_DIRECTION_NORM = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed IC/OC descriptions into drift directions.")
    p.add_argument("--descriptions-json", type=Path, required=True,
                   help='JSON list of {"label", "ic", "oc"} objects.')
    p.add_argument("--embed-model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Must match the model that produced the stream embeddings.")
    p.add_argument("--local-files-only", action="store_true",
                   help="Resolve the model from the local HF cache only; no Hub call.")
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

    with open(args.descriptions_json) as f:
        specs = json.load(f)
    assert specs, f"no descriptions in {args.descriptions_json}"

    model = SentenceTransformer(args.embed_model, local_files_only=args.local_files_only)
    model.eval()
    ic = embed([s["ic"] for s in specs], model)
    oc = embed([s["oc"] for s in specs], model)

    directions = oc - ic
    norms = np.linalg.norm(directions, axis=1)
    assert np.all(norms > MIN_DIRECTION_NORM), (
        f"IC and OC descriptions embed to (nearly) the same point: norms={norms}"
    )
    directions = directions / norms[:, None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        directions=directions,
        labels=np.array([s["label"] for s in specs]),
        ic_texts=np.array([s["ic"] for s in specs]),
        oc_texts=np.array([s["oc"] for s in specs]),
    )
    for spec, norm in zip(specs, norms):
        logging.info("direction %s: pre-normalisation norm %.4f", spec["label"], norm)
    logging.info("wrote %s (k=%d, dim=%d)", args.out, *directions.shape)


if __name__ == "__main__":
    main()
