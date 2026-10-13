"""Merge one duckdb LLM cache into another, deduplicating by cache_key.

Folds a costing run's private cache into the shared per-dataset cache so the
grid reuses its spend. The destination is snapshotted to <dest>.bak before any
write; the source is attached read-only and never modified.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb


def merge_cache(src: Path, dest: Path) -> int:
    """Insert src rows absent from dest (by cache_key); return rows added."""
    assert src.exists(), f"source cache not found: {src}"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        con = duckdb.connect(str(dest), read_only=True)
        try:
            return con.execute("SELECT count(*) FROM completion_cache").fetchone()[0]
        finally:
            con.close()

    shutil.copy2(dest, dest.with_name(dest.name + ".bak"))
    con = duckdb.connect(str(dest))
    try:
        src_sql = str(src).replace("'", "''")
        con.execute(f"ATTACH '{src_sql}' AS src (READ_ONLY)")
        n_before = con.execute("SELECT count(*) FROM completion_cache").fetchone()[0]
        con.execute(
            """
            INSERT INTO completion_cache (cache_key, model, response_json, created_at)
            SELECT s.cache_key, s.model, s.response_json, s.created_at
            FROM src.completion_cache s
            WHERE NOT EXISTS (
                SELECT 1 FROM completion_cache d WHERE d.cache_key = s.cache_key
            )
            """
        )
        n_after = con.execute("SELECT count(*) FROM completion_cache").fetchone()[0]
        return n_after - n_before
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="private costing cache DB")
    p.add_argument("--dest", type=Path, required=True, help="shared per-dataset cache DB")
    args = p.parse_args()
    added = merge_cache(args.src, args.dest)
    print(f"merged {args.src} -> {args.dest}: +{added} rows")


if __name__ == "__main__":
    main()
