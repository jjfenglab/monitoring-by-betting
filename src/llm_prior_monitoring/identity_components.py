"""Identity SCL subcomponents: bypass all LLM/HACHI calls, use embeddings directly.

Assemble these into a SubsetScanSCL to get a baseline monitor where the input
embeddings ARE the concept features — no keyphrase extraction, no concept proposal,
no sparse extraction. The classifier fits on raw embeddings at n_burn_in.
IdentityConceptProjector passes embeddings through unchanged as features.
"""

from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

from .feature_extractors import SparseExtractionMatrix


class IdentityConceptGenerator:
    """Concept generator that uses embedding dimension indices as concept names.

    At n_burn_in, returns [{"concept": "dim_0"}, ..., {"concept": "dim_{d-1}"}].
    Subsequent calls return [] so the concept list never grows.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.config = SimpleNamespace(
            training=SimpleNamespace(num_greedy_holdout=0, train_frac=1.0),
            concept=SimpleNamespace(max_meta_concepts=embedding_dim),
        )

    def generate_initial_concepts(
        self, init_seed: int, data_df: pd.DataFrame, concept_selector
    ):
        return [{"concept": f"dim_{i}"} for i in range(self.embedding_dim)]

    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[Dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[Dict]:
        return []


class IdentityConceptSelector:
    """Placeholder concept_selector: nothing calls or reads it."""

    def __init__(self):
        self.n_init_concepts = None
        self.n_new_concepts_per_update = None


class IdentitySparseFeatureExtractor:
    """Sparse extraction no-op: returns an empty (all-NaN) extraction matrix."""

    def extract_sync(
        self,
        data_df: pd.DataFrame,
        all_concepts: List[Dict],
        obs_concept_to_extract,
        text_column: str = "text",
        error_if_missing: bool = True,
    ) -> SparseExtractionMatrix:
        n_obs, n_concepts = obs_concept_to_extract.shape
        return SparseExtractionMatrix((n_obs, n_concepts))


class IdentityConceptProjector:
    """Identity projector: embeddings pass through unchanged as concept features."""

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings
