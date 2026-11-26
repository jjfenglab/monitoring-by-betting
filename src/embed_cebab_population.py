"""Embed the existing CEBaB population.csv with another sentence-embedding model.

Rows are embedded in file order, so the output npz (key "emb") lines up with
population.csv exactly as population_embeddings.npz does. Pool assembly only slices
embeddings by row, which keeps the documents and streams identical across models.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed population.csv with a given model.")
    p.add_argument("--population-csv", type=Path, required=True)
    p.add_argument("--embed-model", type=str, required=True)
    p.add_argument("--out-embeddings-npz", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--local-files-only", action="store_true",
                   help="Load the model from the local Hugging Face cache only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("args: %s", vars(args))

    texts = pd.read_csv(args.population_csv)["observation"]
    assert texts.notna().all(), "population.csv has missing review text"

    # CPU, matching how the paper's MiniLM embeddings were produced.
    model = SentenceTransformer(args.embed_model, device="cpu",
                                local_files_only=args.local_files_only)
    model.eval()
    logging.info("embedding %d texts with %s", len(texts), args.embed_model)
    emb = model.encode(texts.tolist(), batch_size=args.batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)

    assert len(emb) == len(texts), (len(emb), len(texts))
    assert np.isfinite(emb).all(), "non-finite embedding values"
    args.out_embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_embeddings_npz, emb=emb)
    logging.info("saved embeddings %s -> %s", emb.shape, args.out_embeddings_npz)


if __name__ == "__main__":
    main()
