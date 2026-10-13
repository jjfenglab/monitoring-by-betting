"""Download CEBaB (originals + edits), build a labeled population table, embed it.

Reads the split parquet files directly from the HF Hub at a pinned revision
(bypassing the `datasets` library, whose pyarrow pin is broken in the venv).
Edits only have their edited aspect re-validated; missing ('') aspect labels
are inherited from the row's original where possible and left '' otherwise —
originals themselves are unvalidated on 9-44% of aspects, so consumers must
filter on `{aspect}_label != ""` for whichever aspect they shift. Every unique
review text is embedded into an npz (key "emb") aligned with the CSV row order,
so candidate pool designs can be compared on embedding shift without re-running
the model.

Run remotely, then copy the two outputs back:
    python src/assemble_cebab_population.py \
        --out-population-csv exp_cebab/data/population.csv \
        --out-embeddings-npz exp_cebab/data/population_embeddings.npz \
        --log exp_cebab/data/population.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).parent.parent))

CEBAB_REVISION = "9d2a5a096f1b94344309406886ccc402d9048de8"
CEBAB_SPLITS = [
    "train_exclusive",
    "train_inclusive",
    "train_observational",
    "validation",
    "test",
]
ASPECTS = ["food", "ambiance", "service", "noise"]
MENTION_LABELS = ["Positive", "Negative", "no majority"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and embed the CEBaB population.")
    p.add_argument("--cebab-revision", type=str, default=CEBAB_REVISION)
    p.add_argument("--embed-model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--local-files-only", action="store_true",
                   help="Resolve the model from the local HF cache only; no Hub call. "
                        "Leave off when the CEBaB parquets also need downloading.")
    p.add_argument("--out-population-csv", type=Path, required=True)
    p.add_argument("--out-embeddings-npz", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def download_cebab(revision: str) -> pd.DataFrame:
    """Fetch every split's parquet from the Hub and concatenate them."""
    repo_files = list_repo_files("CEBaB/CEBaB", repo_type="dataset", revision=revision)
    dfs = []
    for split in CEBAB_SPLITS:
        matches = [f for f in repo_files if f.startswith(f"data/{split}-")]
        assert len(matches) == 1, f"expected one parquet for {split}, got {matches}"
        path = hf_hub_download("CEBaB/CEBaB", matches[0], repo_type="dataset",
                               revision=revision)
        dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True)


def build_population(raw: pd.DataFrame) -> pd.DataFrame:
    """Dedup rows/texts, inherit missing aspect labels, derive indicator columns."""
    df = raw.drop_duplicates(subset="id")
    n_dup_text = df["description"].duplicated().sum()
    df = df.drop_duplicates(subset="description").reset_index(drop=True)
    logging.info("unique rows: %d (dropped %d duplicate texts)", len(df), n_dup_text)

    originals = df[df["is_original"]].set_index("original_id")
    pop = pd.DataFrame({
        "observation": df["description"],
        "id": df["id"],
        "original_id": df["original_id"],
        "is_original": df["is_original"],
        "edit_goal": df["edit_goal"],
        "edit_type": df["edit_type"],
        "review_majority": df["review_majority"],
    })
    # numeric stars; NaN where annotators reached no majority on the star rating
    pop["stars"] = pd.to_numeric(df["review_majority"], errors="coerce")

    for a in ASPECTS:
        col = f"{a}_aspect_majority"
        inherited = df["original_id"].map(originals[col])
        label = df[col].where(df[col] != "", inherited).fillna("")
        pop[f"{a}_label"] = label
        labeled = label != ""
        pop[f"{a}_mentioned"] = np.where(labeled, label.isin(MENTION_LABELS), np.nan)
        pop[f"{a}_negative"] = np.where(labeled, label == "Negative", np.nan)
        logging.info(
            "%s: labeled %.3f | mentioned %.3f | negative %.3f (rates among labeled)",
            a, labeled.mean(), pop.loc[labeled, f"{a}_mentioned"].mean(),
            pop.loc[labeled, f"{a}_negative"].mean(),
        )
    logging.info("stars available: %.3f (rest is 'no majority')",
                 pop["stars"].notna().mean())
    return pop


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("args: %s", vars(args))

    raw = download_cebab(args.cebab_revision)
    pop = build_population(raw)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embed_model, local_files_only=args.local_files_only)
    model.eval()
    logging.info("embedding %d texts with %s", len(pop), args.embed_model)
    emb = model.encode(pop["observation"].tolist(), convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    assert len(emb) == len(pop)

    args.out_population_csv.parent.mkdir(parents=True, exist_ok=True)
    pop.to_csv(args.out_population_csv, index=False)
    np.savez(args.out_embeddings_npz, emb=emb)
    logging.info("saved population (%d rows) -> %s", len(pop), args.out_population_csv)
    logging.info("saved embeddings %s -> %s", emb.shape, args.out_embeddings_npz)


if __name__ == "__main__":
    main()
