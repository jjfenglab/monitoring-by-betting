"""Train a TopK sparse autoencoder on a pool embeddings.npz (offline, no LLM).

Wraps hypothesaes.quickstart.train_sae. The checkpoint lands at
<checkpoint-dir>/SAE_M={M}_K={K}.pt with sae_settings.json beside it.
Defaults follow the HypotheSAEs README guidance of (M=64, K=4) for ~1k docs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hypothesaes.quickstart import train_sae
from hypothesaes.sae import get_sae_checkpoint_name


def parse_args() -> argparse.Namespace:
    """CLI arguments for offline SAE training."""
    p = argparse.ArgumentParser(description="Train a TopK SAE on pool embeddings.")
    p.add_argument("--embeddings-path", type=Path, required=True,
                   help="Pool embeddings .npz with key 'emb'")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--M", type=int, default=64, help="Total SAE neurons")
    p.add_argument("--K", type=int, default=4, help="Active neurons per input")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Train the SAE on the pool embeddings and save checkpoint + settings JSON."""
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    emb = np.load(args.embeddings_path)["emb"]
    perm = np.random.permutation(len(emb))
    n_val = int(len(emb) * args.val_frac)
    val_emb, train_emb = emb[perm[:n_val]], emb[perm[n_val:]]

    train_sae(
        train_emb,
        M=args.M,
        K=args.K,
        checkpoint_dir=str(args.checkpoint_dir),
        overwrite_checkpoint=True,
        val_embeddings=val_emb if n_val > 0 else None,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Cached auto-interp glosses describe the old checkpoint's neurons; clear them.
    for stale in args.checkpoint_dir.glob("neuron_names_*.json"):
        stale.unlink()

    checkpoint_path = args.checkpoint_dir / get_sae_checkpoint_name(args.M, args.K)
    settings = {
        "checkpoint_path": str(checkpoint_path),
        "embeddings_path": str(args.embeddings_path),
        "input_dim": int(emb.shape[1]),
        "n_train": len(train_emb),
        "n_val": n_val,
        "M": args.M,
        "K": args.K,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    with open(args.checkpoint_dir / "sae_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Saved checkpoint {checkpoint_path} and sae_settings.json")


if __name__ == "__main__":
    main()
