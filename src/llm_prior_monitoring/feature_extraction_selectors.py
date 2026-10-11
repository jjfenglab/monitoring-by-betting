import numpy as np
import scipy.sparse as sp


class FeatureExtractionSelector:
    """Selects which (observation, concept) pairs to extract.

    Annotate-once-frozen at a fixed rate: each cell in the new-concept columns is
    offered exactly once (all current rows), annotated with probability
    `annotation_rate`, and never revisited. Past-concept columns are frozen and are
    never re-offered, which keeps the marginal missing rate uniform across the
    generations a row has seen. Uses the global `np.random` (seeded in run_monitor).
    """

    def select_features(
        self,
        n_obs: int,
        n_concepts: int,
        new_concept_start: int,
        annotation_rate: float,
    ) -> sp.lil_array:
        """Bernoulli(annotation_rate) over columns [new_concept_start:] for all rows.

        Args:
            n_obs: number of observations (rows) to offer the new columns to.
            n_concepts: total number of concepts (columns).
            new_concept_start: first column index of this step's new generation; columns
                [new_concept_start:] are offered, columns before it are frozen. Use 0 to
                offer all columns (a fresh observation with nothing annotated yet).
            annotation_rate: probability each offered cell is selected for extraction.

        Returns:
            sparse (n_obs, n_concepts) mask, 1=extract, 0=don't; nonzero only in the
            new-column block.
        """
        n_new = n_concepts - new_concept_start
        dense = np.zeros((n_obs, n_concepts), dtype=np.int8)
        if n_obs > 0 and n_new > 0:
            dense[:, new_concept_start:] = np.random.random((n_obs, n_new)) < annotation_rate
        return sp.lil_array(dense)
