"""Mock implementations of HACHI subcomponents for testing SequentialConceptLearner.

Real counterparts live in HACHI (not modified). These mocks have the same interface
so a real SequentialConceptLearner can be assembled with them for LLM-free testing.
"""
from types import SimpleNamespace

from .concept_selectors import MockConceptSelector, MockConceptState


class MockConceptGenerator:
    """Concept generator that skips word count fitting and LLM calls.

    Delegates concept proposal entirely to MockConceptSelector.query_for_new_cand,
    bypassing HACHI's get_word_count_data and data split steps.

    Carries a minimal config namespace mirroring HACHI's generator interface.
    """

    def __init__(self, concept_selector: MockConceptSelector):
        self.concept_selector = concept_selector
        self.config = SimpleNamespace(
            training=SimpleNamespace(
                num_greedy_holdout=0,
                train_frac=1.0,
            ),
            concept=SimpleNamespace(
                max_meta_concepts=concept_selector.state.n_init_concepts,
            ),
        )

    def generate_initial_concepts(self, init_seed, data_df, concept_selector):
        """Propose initial concepts without word count fitting or LLM calls."""
        concept_selector.state.post_seen = bool(data_df["y_true"].any())
        return concept_selector.query_for_new_cand("", [], max_new_tokens=0)

    def generate_candidate_concepts(
        self, init_data_df, concept_selector, current_concepts, current_extractions
    ):
        """Propose new candidate concepts without residual fitting or LLM calls."""
        concept_selector.state.post_seen = bool(init_data_df["y_true"].any())
        return concept_selector.query_for_new_cand("", [], max_new_tokens=0)


class MockKeyphrase:
    """Passthrough keyphrase extractor that returns data_df unchanged.

    Adds a placeholder llm_output column ("mocking") if absent: SubsetScanSCL only
    uses rows whose llm_output is non-empty.
    """

    def extract(self, data_df, train_idxs):
        if "llm_output" not in data_df.columns:
            data_df = data_df.copy()
            data_df["llm_output"] = "mocking"
        return data_df
