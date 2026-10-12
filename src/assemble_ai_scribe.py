"""Embed p0/p1 AI-scribe-note pools into embeddings.npz + mapping.csv.

Unlike assemble_cebab.py, there is no population-build or variant-selection
stage here: data/p0_pool.csv (pre-change) and data/p1_pool.csv (post-change)
are the user's own pre-split, gitignored source files. Each row carries at
least two text columns -- an AI-generated draft note and the clinician's
final/updated note -- and --text-mode picks what gets embedded and monitored:

  draft_only        : just the draft column (e.g. abr_1_hpi)
  final_only         : just the clinician-final column (e.g. abr_upd_hpi)
  draft_plus_final   : both columns, tagged and concatenated into one string:
                       "<ai_draft>...</ai_draft><final_note>...</final_note>"
  diff               : only the edit itself -- diff-match-patch's changed spans
                       between draft and final, tagged and joined by " ... ":
                       "<del>removed</del> ... <ins>added</ins>". Unchanged text is
                       dropped entirely, so a note with no clinician edits produces
                       an empty string.
  draft_plus_diff    : the full draft followed by the diff:
                       "<ai_draft>...</ai_draft><changes><del>...</del> ... <ins>...</ins></changes>"
  draft_diff_inline  : the full draft with character-level <del>/<ins> edits inline
  draft_diff_inline_word : the same, but diffed word by word
  diff_structured    : the draft, then each edit numbered as BEFORE/AFTER lines with
                       3 words of context on each side

Runs once per (experiment, text_mode); every downstream cell for that
text_mode loads the resulting cached files.

A deterministic slice of the p0 rows (--sae-frac, --sae-split-seed) is carved
off and labeled "p0_sae" instead of "p0": the SAE trains only on that slice
(src/extract_split_embeddings.py --split-label p0_sae), while every stream
(src/make_data_stream.py, which only ever looks at split_label=="p0") only
ever draws from the remaining "p0" rows. This keeps the SAE's training set
and every monitored document strictly disjoint without touching the source
CSVs -- the split is deterministic (row-index permutation under
--sae-split-seed) and lives entirely in this script.

Outputs:
  embeddings.npz : single file with all embeddings stacked (p0 rows, then p1)
  mapping.csv    : row_index, split_label ("p0" | "p0_sae" | "p1"), text,
                   source_idx, true_group
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from diff_match_patch import diff_match_patch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import embed

TEXT_MODES = ("draft_only", "final_only", "draft_plus_final", "diff", "draft_plus_diff", "draft_diff_inline", "draft_diff_inline_word", "diff_structured")


def diff_only(draft: str, final: str) -> str:
    """Just the changed spans between draft and final, tagged and joined by ' ... '.

    Uses diff-match-patch's diff_cleanupSemantic to merge fragmented
    character-level edits into human-readable, phrase-aligned chunks --
    important for prose, where a raw character diff tends to be choppy
    (e.g. splitting mid-word) unless pre-tokenized by word first.
    """
    dmp = diff_match_patch()
    diffs = dmp.diff_main(draft, final)
    dmp.diff_cleanupSemantic(diffs)
    changed = []
    for op, text in diffs:
        if op == -1:
            changed.append(f"<del>{text}</del>")
        elif op == 1:
            changed.append(f"<ins>{text}</ins>")
    return " ... ".join(changed)

def diff_inline(draft: str, final: str) -> str:
    """Full text with changes marked inline (character-level)."""
    dmp = diff_match_patch()
    diffs = dmp.diff_main(draft, final)
    dmp.diff_cleanupSemantic(diffs)
    parts = []
    for op, text in diffs:
        if op == 0:
            parts.append(text)
        elif op == -1:
            parts.append(f"<del>{text}</del>")
        elif op == 1:
            parts.append(f"<ins>{text}</ins>")
    return "".join(parts)


def diff_inline_word(draft: str, final: str) -> str:
    """Full text with changes marked inline (word-level).

    Uses difflib.SequenceMatcher for true word-level diffing.
    E.g., "2/26/26" -> "2/20/26" becomes "<del>2/26/26</del> <ins>2/20/26</ins>"
    instead of character-level "2/2<del>6</del><ins>0</ins>/26".
    """
    draft_words = re.findall(r'\S+', draft)
    final_words = re.findall(r'\S+', final)

    matcher = difflib.SequenceMatcher(None, draft_words, final_words)
    parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(draft_words[i1:i2]))
        elif tag == "delete":
            parts.append(f"<del>{' '.join(draft_words[i1:i2])}</del>")
        elif tag == "insert":
            parts.append(f"<ins>{' '.join(final_words[j1:j2])}</ins>")
        elif tag == "replace":
            parts.append(f"<del>{' '.join(draft_words[i1:i2])}</del>")
            parts.append(f"<ins>{' '.join(final_words[j1:j2])}</ins>")

    return " ".join(parts)


def diff_structured(draft: str, final: str, tail_len: int = 3) -> str:
    """Structured diff format with explicit sections for draft, final, and numbered edits.

    More explicit format that may be easier for LLMs to parse than inline diffs.
    """
    draft_words = re.findall(r'\S+', draft)
    final_words = re.findall(r'\S+', final)

    matcher = difflib.SequenceMatcher(None, draft_words, final_words)

    edits = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # Get surrounding context
        ctx_before_draft = " ".join(draft_words[max(0, i1-tail_len):i1])
        ctx_after_draft = " ".join(draft_words[i2:i2+tail_len])
        ctx_before_final = " ".join(final_words[max(0, j1-tail_len):j1])
        ctx_after_final = " ".join(final_words[j2:j2+tail_len])

        if tag == "delete":
            deleted_text = " ".join(draft_words[i1:i2])
            edits.append({
                "before": f"{ctx_before_draft} <del>{deleted_text}</del> {ctx_after_draft}".strip(),
                "after": f"{ctx_before_final} [DELETED] {ctx_after_final}".strip()
            })
        elif tag == "insert":
            inserted_text = " ".join(final_words[j1:j2])
            edits.append({
                "before": f"{ctx_before_draft} [INSERTED] {ctx_after_draft}".strip(),
                "after": f"{ctx_before_final} <ins>{inserted_text}</ins> {ctx_after_final}".strip()
            })
        elif tag == "replace":
            old_text = " ".join(draft_words[i1:i2])
            new_text = " ".join(final_words[j1:j2])
            edits.append({
                "before": f"{ctx_before_draft} <del>{old_text}</del> {ctx_after_draft}".strip(),
                "after": f"{ctx_before_final} <ins>{new_text}</ins> {ctx_after_final}".strip()
            })

    # Build structured output
    parts = [
        "=== ORIGINAL AI DRAFT ===",
        draft,
        "",
        "=== EDITS (<ins>=inserted, <del>=deleted) ===",
    ]

    for i, edit in enumerate(edits, 1):
        parts.append(f"EDIT {i}")
        parts.append(f"BEFORE: {edit['before']}")
        parts.append(f"AFTER: {edit['after']}")
        parts.append("")

    return "\n".join(parts)


def build_text(df: pd.DataFrame, text_mode: str, draft_col: str, final_col: str) -> list[str]:
    if text_mode == "draft_only":
        return df[draft_col].astype(str).tolist()
    elif text_mode == "final_only":
        return df[final_col].astype(str).tolist()
    elif text_mode == "draft_plus_final":
        return [
            f"<ai_draft>{d}</ai_draft><final_note>{f}</final_note>"
            for d, f in zip(df[draft_col].astype(str), df[final_col].astype(str))
        ]
    elif text_mode == "diff":
        return [
            diff_only(d, f)
            for d, f in zip(df[draft_col].astype(str), df[final_col].astype(str))
        ]
    elif text_mode == "draft_plus_diff":
        return [
            f"<ai_draft>{d}</ai_draft><changes>{diff_only(d, f)}</changes>"
            for d, f in zip(df[draft_col].astype(str), df[final_col].astype(str))
        ]
    elif text_mode == "draft_diff_inline":
        return [
            diff_inline(d, f)
            for d, f in zip(df[draft_col].astype(str), df[final_col].astype(str))
        ]
    elif text_mode == "draft_diff_inline_word":
        return [
            diff_inline_word(d, f)
            for d, f in zip(df[draft_col].astype(str), df[final_col].astype(str))
        ]
    elif text_mode == "diff_structured":
        return [
            diff_structured(d, f)
            for d, f in zip(df[draft_col].astype(str), df[final_col].astype(str))
        ]
    raise ValueError(f"unknown text_mode: {text_mode!r}, expected one of {TEXT_MODES}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--p0-pool-path", type=Path, required=True,
                    help="pre-change pool CSV (gitignored source data)")
    p.add_argument("--p1-pool-path", type=Path, required=True,
                    help="post-change pool CSV (gitignored source data)")
    p.add_argument("--text-mode", choices=TEXT_MODES, required=True)
    p.add_argument("--draft-col", default="abr_1_hpi", help="AI-draft text column")
    p.add_argument("--final-col", default="abr_upd_hpi", help="clinician-final text column")
    p.add_argument("--embed-model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sae-frac", type=float, default=0.2,
                   help="fraction of p0 rows carved off (deterministically) for "
                        "SAE training, labeled split_label='p0_sae' instead of 'p0' "
                        "so streams never draw them")
    p.add_argument("--sae-split-seed", type=int, default=0,
                   help="seed for the p0 / p0_sae row partition (row-index "
                        "permutation only -- reproducible independent of embedding "
                        "nondeterminism)")
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument("--mapping-path", type=Path, required=True)
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
    )
    logging.info("args: %s", vars(args))

    p0_df = pd.read_csv(args.p0_pool_path)
    p1_df = pd.read_csv(args.p1_pool_path)

    p0_texts = np.array(build_text(p0_df, args.text_mode, args.draft_col, args.final_col))
    p1_texts = np.array(build_text(p1_df, args.text_mode, args.draft_col, args.final_col))
    if args.max_chars is not None:
        p0_texts = p0_texts[np.char.str_len(p0_texts) < args.max_chars]
        p1_texts = p1_texts[np.char.str_len(p1_texts) < args.max_chars]
    texts = np.concatenate([p0_texts, p1_texts])
    logging.info("text_mode=%s  p0=%d rows  p1=%d rows", args.text_mode, len(p0_texts), len(p1_texts))

    n_p0, n_p1 = len(p0_texts), len(p1_texts)

    # Deterministic, seeded row-index permutation: independent of embedding
    # nondeterminism, so the p0/p0_sae partition is reproducible across machines
    # even though the embeddings themselves are not bit-identical across them.
    rng = np.random.default_rng(args.sae_split_seed)
    n_sae = round(n_p0 * args.sae_frac)
    sae_idx = set(rng.permutation(n_p0)[:n_sae].tolist())
    p0_split_labels = ["p0_sae" if i in sae_idx else "p0" for i in range(n_p0)]
    logging.info("p0 split: sae=%d  stream=%d  (frac=%.3f seed=%d)",
                 n_sae, n_p0 - n_sae, args.sae_frac, args.sae_split_seed)

    mapping_df = pd.DataFrame({
        "row_index": np.arange(n_p0 + n_p1),
        "split_label": p0_split_labels + ["p1"] * n_p1,
        "text": texts,
        "source_idx": list(range(n_p0)) + list(range(n_p1)),
        "true_group": [0] * n_p0 + [1] * n_p1,
    })
    
    args.mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(args.mapping_path, index=False)

    # Generate and save embeddings
    model = SentenceTransformer(args.embed_model, local_files_only=True)
    model.eval()
    logging.info("embedding with %s", args.embed_model)

    all_emb = embed(texts, model)

    args.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.embeddings_path, emb=all_emb)

    logging.info(
        "saved embeddings(%d rows, %d dims) -> %s", len(all_emb), all_emb.shape[1], args.embeddings_path
    )
    logging.info("saved mapping (p0=%d [stream=%d, sae=%d], p1=%d) -> %s",
                 n_p0, n_p0 - n_sae, n_sae, n_p1, args.mapping_path)


if __name__ == "__main__":
    main()
