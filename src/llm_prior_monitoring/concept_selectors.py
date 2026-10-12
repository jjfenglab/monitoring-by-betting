import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

# from HACHI.src.ensemble_trainer.model_utils import fit_residual  # lazy: imported inside RankedConceptSelector methods

class Concept(BaseModel):
    definition: str = Field(description="yes no question that defines the concept")
    keyphrases: List[str] = Field(description="keyphrases that are related to this concept and should be excluded when doing keyphrase-based changepoint estimation")

class CandidateConcepts(BaseModel):
    reasoning: Optional[str] = Field(
        description="Reasoning through each one of the attributes"
    )
    concepts: List[Concept] = Field(
        description="List of candidate concepts with associated keyphrases"
    )


class RankedConceptSelector:
    """Concept selector that ranks features by logistic regression coefficient magnitude.

    Uses fit_residual() to identify the most discriminative word features before
    prompting the LLM, rather than passing alphabetically-ordered features.
    """

    def __init__(
        self,
        llm_dict,
        prompt_iter_file: str | Path,
        prompt_init_file: str | Path,
        residual_model_type: str = "l2",
        num_top: int = 20,
        penalty_downweight_factor: float = 1000.0,
        cv: int = 5,
        seed: int = 0,
    ):
        self.llm_dict = llm_dict
        self.prompt_iter_file = Path(prompt_iter_file)
        self.prompt_init_file = Path(prompt_init_file)
        self._prompt_iter_template = self._load_prompt_template(self.prompt_iter_file)
        self._prompt_init_template = self._load_prompt_template(self.prompt_init_file)
        self.force_keep_columns = None
        # Set by SequentialConceptLearner to inject into prompt
        self.n_init_concepts: int = 10
        # Overrides num_replace=0 that HACHI passes when num_greedy_holdout is set to 0
        self.n_new_concepts_per_update: Optional[int] = None
        self.residual_model_type = residual_model_type
        self.num_top = num_top
        self.penalty_downweight_factor = penalty_downweight_factor
        self.cv = cv
        self.seed = seed

    def _load_prompt_template(self, path: Path) -> str:
        with open(path, "r") as f:
            return f.read()

    def scrub_vectorized_sentences(self, X_words, word_names, concept_dicts):
        """Return inputs unchanged."""
        return X_words, word_names

    def query_for_new_cand(self, prompt, top_feat_names, max_new_tokens) -> list[dict]:
        """Query LLM for new candidate concepts."""
        from src.llm_utils import SYSTEM_PROMPT

        llm = self.llm_dict["extraction"]
        try:
            llm_response = llm.run(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                response_format=CandidateConcepts,
                strict_response_format=True,
            )
            return [
                {"concept": c.definition, "keyphrases": c.keyphrases}
                for c in llm_response.concepts
            ]
        except Exception as e:
            logging.info(f"LLM call for new concepts failed so returning no new concepts: {e}")
            return []

    def make_initial_concept_prompt(self, X_words, y, sample_weight, data_df, feat_names):
        """Generate prompt with features ranked by logistic regression coefficient."""
        from HACHI.src.ensemble_trainer.model_utils import fit_residual
        feat_names_list = feat_names.tolist() if hasattr(feat_names, "tolist") else list(feat_names)
        top_df = fit_residual(
            self.residual_model_type,
            feat_names_list,
            None,
            X_words,
            y,
            sample_weight=sample_weight,
            penalty_downweight_factor=100,
            is_multiclass=np.unique(y).size > 2,
            num_top=self.num_top,
            cv=self.cv,
            seed=self.seed,
        )
        norm = np.max(np.abs(top_df.coef))
        if norm > 0:
            top_df["coef"] = top_df.coef / norm
        top_features_text = top_df[["feature_name", "coef"]].to_csv(index=False, float_format="%.3f")
        prompt = self._prompt_init_template.format(
            top_features_text=top_features_text,
            max_meta_concepts=self.n_init_concepts,
        )
        return prompt, top_features_text, top_df.feature_name

    def make_new_concept_prompt(
        self,
        X_extracted,
        X_words,
        y,
        sample_weight,
        data_df,
        extract_feature_names,
        feat_names,
        num_replace,
    ):
        """Generate prompt with residual features ranked after controlling for existing concepts."""
        from HACHI.src.ensemble_trainer.model_utils import fit_residual
        feat_names_list = feat_names.tolist() if hasattr(feat_names, "tolist") else list(feat_names)
        top_df = fit_residual(
            self.residual_model_type,
            feat_names_list,
            X_extracted,
            X_words,
            y,
            sample_weight=sample_weight,
            penalty_downweight_factor=self.penalty_downweight_factor,
            is_multiclass=np.unique(y).size > 2,
            num_top=self.num_top,
            cv=self.cv,
            seed=self.seed,
        )
        norm = np.max(np.abs(top_df.coef))
        if norm > 0:
            top_df["coef"] = top_df.coef / norm
        top_features_text = top_df[["feature_name", "coef"]].to_csv(index=False, float_format="%.3f")
        meta_concepts = ", ".join(extract_feature_names)
        effective_num_replace = (
            self.n_new_concepts_per_update
            if self.n_new_concepts_per_update is not None
            else num_replace
        )
        prompt = self._prompt_iter_template.format(
            meta_concepts=meta_concepts,
            top_features_text=top_features_text,
            max_meta_concepts=2,
            num_concepts_fixed=len(extract_feature_names),
            num_concepts=len(extract_feature_names)+1,
        )
        return prompt, meta_concepts, top_features_text, top_df.feature_name


@dataclass
class MockConceptState:
    """Shared state between MockConceptSelector and MockSparseFeatureExtractor.

    MockConceptSelector stamps each proposed concept with its role (strength s/w/noise,
    is_informative) and appends the flag to informative_flags; MockSparseFeatureExtractor
    labels cells from the concept's strength.

    field(default_factory=...) is required for mutable defaults in dataclasses —
    without it, all instances would share the same list/generator object.
    """

    p_informative: float = 0.5  # unused since the fixed strong/weak/noise batches
    noise_rate: float = 0.1
    n_init_concepts: int = 10
    n_new_concepts_per_update: int = 2
    # P(weak concept != its strong partner's latent value); controls how weakly the
    # weak concept correlates with (and predicts through) its strong partner.
    weak_flip_rate: float = 0.3
    post_seen: bool = False  # set by MockConceptGenerator: a post-change sample is in history
    # One bool per proposed concept; True = tracks y_true (informative), False = Bernoulli(0.5) noise
    informative_flags: list = field(default_factory=list)
    # Strong-concept latent cache: (text, strong_id, y_val) -> 0/1. Lets the weak
    # concept be a noisy copy of its partner's realized value even though cells are
    # extracted lazily across calls, and keeps recurring texts self-consistent.
    latent_values: dict = field(default_factory=dict)
    # Monotonic count of concepts ever proposed; names concepts uniquely. Unlike
    # len(informative_flags) it never resets when concepts are removed, so mock_k_T/F
    # names stay distinct across removals (debug traceability).
    total_proposed: int = 0


class MockConceptSelector:
    """Concept selector that returns oracle concept dicts without calling an LLM.

    Stamps each concept's role at proposal time and records it in the shared
    MockConceptState; MockSparseFeatureExtractor labels cells from the concept's strength.
    """

    def __init__(self, state: MockConceptState):
        self.state = state
        self.n_init_concepts = state.n_init_concepts
        self.n_new_concepts_per_update = state.n_new_concepts_per_update

    def query_for_new_cand(self, prompt, top_feat_names, max_new_tokens) -> list[dict]:
        """Return mock concept dicts without calling an LLM.

        Uses n_init_concepts on the first call (no flags yet) and
        n_new_concepts_per_update on all subsequent calls. Pre-change proposals are
        all noise; each post-change batch with n >= 4 is a fixed bundle of 1 strong
        informative, 1 weak informative (a noisy copy of the strong one's latent,
        linked via partner_id), and n - 2 noise concepts. With n in {2, 3} there's no
        room for the weak concept, so the batch is just 1 strong informative and
        n - 1 noise concepts.

        Names come from the monotonic total_proposed counter (never reused after a
        removal); it runs in lockstep with the SCL's id counter, so a batch's first
        concept (the strong one) is guaranteed id start_id and the weak dict can
        record partner_id before the SCL stamps ids. The _T_s/_T_w/_F suffix and the
        is_informative/strength keys snapshot the role at proposal time so it stays
        readable after removal reindexes the state.
        """
        n = (
            self.state.n_init_concepts
            if len(self.state.informative_flags) == 0
            else self.state.n_new_concepts_per_update
        )
        start_id = self.state.total_proposed
        self.state.total_proposed += n
        if not self.state.post_seen:
            # Concepts born before any post-change sample is in history are noise forever.
            self.state.informative_flags.extend([False] * n)
            return [
                {"concept": f"mock_{start_id + i}_F", "is_informative": False}
                for i in range(n)
            ]
        assert n >= 2, (
            f"post-change mock batches need n_new_concepts_per_update >= 2 "
            f"(1 strong + >=1 noise), got {n}"
        )
        if n < 4:
            self.state.informative_flags.extend([True] + [False] * (n - 1))
            batch = [
                {"concept": f"mock_{start_id}_T_s", "is_informative": True,
                 "strength": "s"},
            ]
            batch += [
                {"concept": f"mock_{start_id + i}_F", "is_informative": False}
                for i in range(1, n)
            ]
            return batch
        self.state.informative_flags.extend([True, True] + [False] * (n - 2))
        batch = [
            {"concept": f"mock_{start_id}_T_s", "is_informative": True,
             "strength": "s"},
            {"concept": f"mock_{start_id + 1}_T_w", "is_informative": True,
             "strength": "w", "partner_id": start_id},
        ]
        batch += [
            {"concept": f"mock_{start_id + i}_F", "is_informative": False}
            for i in range(2, n)
        ]
        return batch
