"""Sentence-transformer embedding helper shared by the pool-assembly and direction scripts."""

from __future__ import annotations

import numpy as np


def embed(inputs: list, model) -> np.ndarray:
    """Encode texts (or PIL images, for a CLIP checkpoint) to unit-norm embeddings."""
    return model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
