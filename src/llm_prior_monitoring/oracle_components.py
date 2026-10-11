"""Oracle SCL subcomponents: user-supplied ground-truth columns are the concepts.

The concept set is given up front (pool-CSV column names) and there is no LLM
anywhere. Two proposal modes exist: OracleConceptGenerator (all_at_once)
returns every supplied column at burn-in and never proposes again, while
OraclePoolConceptGenerator (random_pool) proposes columns sequentially from
the pool — n_init at burn-in, then the not-currently-active columns each step
(the SCL subsamples to n_new). The full oracle concept matrix is precomputed
at construction (text-keyed lookup over the pool CSVs).

Ground-truth NaN (an aspect not annotated for a row) is left as NaN in the
concept matrix.

Assembled by create_lgm_pipeline.make_sequential_concept_learner for
concept_learner_type == "oracle", reusing MockKeyphrase and IdentityConceptSelector.
"""

from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .feature_extractors import SparseExtractionMatrix


class OracleState:
    """Text-keyed lookup of ground-truth concept rows, built from the pool CSVs."""

    def __init__(self, pool_csvs, concept_cols, text_column: str = "observation"):
        self.concept_cols = list(concept_cols)
        self.col_idx = {c: j for j, c in enumerate(self.concept_cols)}
        self.text_to_concepts: Dict[str, np.ndarray] = {}
        for path in pool_csvs:
            df = pd.read_csv(path)
            missing = [c for c in self.concept_cols if c not in df.columns]
            assert not missing, f"{path} lacks concept columns {missing}"
            values = df[self.concept_cols].to_numpy(dtype=float)
            valid = values[~np.isnan(values)]
            assert np.isin(valid, (0.0, 1.0)).all(), (
                f"non-binary concept values in {path}"
            )
            for text, row in zip(df[text_column], values):
                prev = self.text_to_concepts.get(text)
                assert prev is None or np.array_equal(prev, row, equal_nan=True), (
                    f"duplicate pool text with conflicting concepts: {text!r}"
                )
                self.text_to_concepts[text] = row

    def lookup(self, texts) -> np.ndarray:
        """Ground-truth rows (n, k) for texts; KeyError = text not in the pools."""
        return np.stack([self.text_to_concepts[t] for t in texts])


class OracleConceptGenerator:
    """Emits every supplied column as a concept at burn-in; never proposes again."""

    def __init__(self, state: OracleState):
        self.state = state
        self.config = SimpleNamespace(
            training=SimpleNamespace(num_greedy_holdout=0, train_frac=1.0),
            concept=SimpleNamespace(max_meta_concepts=len(state.concept_cols)),
        )

    def generate_initial_concepts(self, init_seed, data_df: pd.DataFrame, concept_selector):
        return [{"concept": col} for col in self.state.concept_cols]

    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[Dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[Dict]:
        return []


class OraclePoolConceptGenerator:
    """Random-pool proposal: n_init at burn-in, then every not-currently-active
    column each step (the SCL subsamples to n_new). Stateless w.r.t. removal:
    removed columns leave all_concepts and become re-eligible (recycling)."""

    def __init__(
        self,
        state: OracleState,
        n_init_concepts: int,
        n_new_concepts_per_update: int = 0,
        informative_cols=(),
    ):
        self.state = state
        self.n_init_concepts = n_init_concepts
        self.n_new_concepts_per_update = n_new_concepts_per_update
        self.informative_cols = set(informative_cols)
        unknown = self.informative_cols - set(state.concept_cols)
        assert not unknown, f"informative_cols not in pool: {unknown}"
        self.config = SimpleNamespace(
            training=SimpleNamespace(num_greedy_holdout=0, train_frac=1.0),
            concept=SimpleNamespace(max_meta_concepts=n_init_concepts),
        )

    def _build_concept_dicts(self, cols) -> List[Dict]:
        # is_informative feeds the frac_noninformative_* debug columns only;
        # omitted entirely when no informative columns are configured.
        if not self.informative_cols:
            return [{"concept": c} for c in cols]
        return [{"concept": c, "is_informative": c in self.informative_cols}
                for c in cols]

    def generate_initial_concepts(self, init_seed, data_df: pd.DataFrame, concept_selector):
        assert self.n_init_concepts <= len(self.state.concept_cols)
        cols = np.random.choice(self.state.concept_cols, size=self.n_init_concepts, replace=False)
        return self._build_concept_dicts(list(cols))

    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[Dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[Dict]:
        active = {c["concept"] for c in current_concepts}
        return self._build_concept_dicts(
            [c for c in self.state.concept_cols if c not in active])


class OracleSparseFeatureExtractor:
    """Reveals ground-truth values for the selected (obs, concept) cells."""

    def __init__(self, state: OracleState):
        self.state = state

    def extract_sync(
        self,
        data_df: pd.DataFrame,
        all_concepts: List[Dict],
        obs_concept_to_extract: sp.sparray,
        text_column: str = "text",
        error_if_missing: bool = True,
    ) -> SparseExtractionMatrix:
        n_obs, n_concepts = obs_concept_to_extract.shape
        result = SparseExtractionMatrix((n_obs, n_concepts))
        mask = (obs_concept_to_extract.toarray()
                if sp.issparse(obs_concept_to_extract) else obs_concept_to_extract)
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return result
        truth = self.state.lookup(data_df[text_column])
        for obs_idx, concept_idx in zip(rows, cols):
            # Map by concept NAME, not position, so column order never drifts.
            value = truth[obs_idx, self.state.col_idx[all_concepts[concept_idx]["concept"]]]
            if not np.isnan(value):  # NaN ground truth stays unextracted -> imputed
                result[obs_idx, concept_idx] = int(value)
        return result
