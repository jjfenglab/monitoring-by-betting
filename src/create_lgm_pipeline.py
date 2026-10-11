"""Factory functions for assembling the adaptive monitoring pipeline.

Provides three component factories (make_e_detector, make_changepoint_learner,
make_sequential_concept_learner) and a convenience assembler (make_monitor).
CHANGEPOINT_LEARNER_NAMES are defined here and imported
by run_monitor.py for argparse choices.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llm_prior_monitoring.changepoint import (
    ChangepointLearner,
    KernelCPDLearner,
    MeanShiftChangepointLearner,
    SlicedWassersteinCPDLearner,
)
from llm_prior_monitoring.classifiers import DimensionAdaptiveLogistic, DirectionFixedLogistic
from llm_prior_monitoring.concept_selectors import MockConceptSelector, MockConceptState
from llm_prior_monitoring.e_detector import EDetector
from llm_prior_monitoring.e_detector_sr import (
    SymmetricSREDetector,
)
from llm_prior_monitoring.feature_extractors import MockSparseFeatureExtractor
from llm_prior_monitoring.identity_components import (
    IdentityConceptGenerator,
    IdentityConceptProjector,
    IdentityConceptSelector,
    IdentitySparseFeatureExtractor,
)
from llm_prior_monitoring.mock_HACHI import MockConceptGenerator, MockKeyphrase
from llm_prior_monitoring.scl_config import SCLConfig
from llm_prior_monitoring.sequential_concept_learner import (
    SequentialConceptLearner,
    SubsetScanSCL,
)

CHANGEPOINT_LEARNER_NAMES: frozenset[str] = frozenset({
    "mean_shift",
    "kernel_cosine",
    "kernel_rbf",
    "sw_stored",
    "sw_random",
})

# The representation the changepoint learner scans for the shift: raw embeddings, or
# the imputer concept space (real annotations override imputed values).
CP_FEATURE_SOURCE_NAMES: frozenset[str] = frozenset({
    "embedding",
    "concept",
})

SCL_TYPES: frozenset[str] = frozenset({"identity", "mock", "sae", "oracle", "bertopic_llm"})


def make_classifier(config: SCLConfig) -> BaseEstimator:
    """Build the SCL's classifier from config, optionally CV-tuning it at each fit."""
    clf_cfg = config.classifier
    classifier_type = clf_cfg.classifier_type
    assert classifier_type in ("logistic", "random_forest", "direction_fixed"), (
        f"Unknown classifier_type: {classifier_type!r}"
    )
    if classifier_type == "logistic":
        lr = DimensionAdaptiveLogistic(
            max_unpenalized_dim=clf_cfg.max_unpenalized_dim,
            l1_ratio=clf_cfg.l1_ratio,
            cv=clf_cfg.cv,
            cv_folds=clf_cfg.cv_folds,
            cv_n_cs=clf_cfg.cv_n_cs,
            random_state=config.seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", lr)])

    if classifier_type == "direction_fixed":
        clf = DirectionFixedLogistic()
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    base = RandomForestClassifier(
        n_estimators=clf_cfg.rf_n_estimators, class_weight="balanced", random_state=config.seed,
    )
    if not clf_cfg.cv:
        return base
    return RandomizedSearchCV(
        base,
        param_distributions={
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_leaf": [3, 5, 10],
        },
        n_iter=clf_cfg.rf_n_iter, cv=clf_cfg.cv_folds, scoring="neg_log_loss",
        refit=True, random_state=config.seed,
    )


def make_sequential_concept_learner(
    config: SCLConfig,
    embedding_dim: int,
    cache_file: str,
    fixed_calibs: np.ndarray | None = None,
    calib_df: pd.DataFrame | None = None,
    min_block: int | None = None,
    keyphrase_outputs_file: str | None = None,
) -> SequentialConceptLearner:
    """Assemble a SequentialConceptLearner variant from config.

    identity     — embeddings are the concepts; no LLM calls.
    mock         — Bernoulli concept simulation based on per-row y_true; no LLM calls.
    sae          — pretrained SAE latents are the concepts; no LLM calls. With
                   config.sae.refit_every set, the SAE is warm-started on the history
                   each step (online variant); removal is supported either way.
    oracle       — user-supplied ground-truth pool columns are the concepts; no LLM calls.
    bertopic_llm — cluster keyphrases with BERTopic, formalize top shifted topics into
                   yes/no concepts via LLM, annotate via LLM.

    min_block is the CP learner's min_block, passed through by run_monitor so the SCL
    and CP learner agree by construction (required for scl_variant=remove_then_detect).

    keyphrase_outputs_file is the per-run keyphrase dump (write-only debug output).
    None writes nothing; run_monitor points it into the run's own output dir so
    concurrent streams do not race on one path.
    """
    concept_learner_type = config.concept_learner_type
    assert concept_learner_type in SCL_TYPES, (
        f"Unknown concept_learner_type: {concept_learner_type!r}"
    )
    assert config.scl_variant == "subset_scan", (
        f"Only subset_scan variant is supported, got {config.scl_variant!r}"
    )
    assert min_block is not None, "subset_scan needs min_block from run_monitor"
    ann_cfg = config.annotation_selector
    classifier = make_classifier(config)
    prop_cfg = config.concept_proposer

    if concept_learner_type == "identity":
        # Identity: embedding dimensions are the "concepts", no annotation or proposal
        # subset_size = embedding_dim ensures no concepts are filtered out
        return SubsetScanSCL(
            min_block=min_block,
            subset_size=embedding_dim,
            search_all_sizes=config.search_all_sizes,
            concept_projector=IdentityConceptProjector(),
            keyphrase_extractor=MockKeyphrase(),
            concept_generator=IdentityConceptGenerator(embedding_dim),
            concept_selector=IdentityConceptSelector(),
            sparse_feature_extractor=IdentitySparseFeatureExtractor(),
            classifier=classifier,
            n_burn_in=config.n_burn_in,
            n_init_concepts=embedding_dim,
            n_new_concepts_per_update=0,  # No new concepts - use embeddings as-is
            fixed_calibs=fixed_calibs,
            calib_df=calib_df,
            seed=config.seed,
        )
    elif concept_learner_type == "sae":
        # SAE: pretrained SAE latents are the "concepts", no annotation or proposal
        from llm_prior_monitoring.sae_components import (
            SAEConceptGenerator,
            SAEConceptProjector,
            SAEState,
        )
        sae_cfg = config.sae
        state = SAEState(
            sae_cfg.checkpoint_path, sae_cfg.embeddings_path, sae_cfg.mapping_path,
            side_embeddings_path=sae_cfg.side_embeddings_path,
            refit_every=sae_cfg.refit_every,
            refit_epochs=sae_cfg.refit_epochs,
            refit_lr=sae_cfg.refit_lr,
            refit_batch_size=sae_cfg.refit_batch_size,
        )
        # subset_size = n_init_concepts ensures no concepts are filtered out
        return SubsetScanSCL(
            min_block=min_block,
            subset_size=prop_cfg.n_init_concepts,
            search_all_sizes=config.search_all_sizes,
            concept_projector=SAEConceptProjector(state),
            keyphrase_extractor=MockKeyphrase(),
            concept_generator=SAEConceptGenerator(
                state,
                selection_method=sae_cfg.selection_method,
                interpret=sae_cfg.interpret,
                interpreter_model=sae_cfg.interpreter_model,
            ),
            concept_selector=IdentityConceptSelector(),
            sparse_feature_extractor=IdentitySparseFeatureExtractor(),
            classifier=classifier,
            n_burn_in=config.n_burn_in,
            n_init_concepts=prop_cfg.n_init_concepts,
            n_new_concepts_per_update=0,  # No new concepts - use SAE latents as-is
            fixed_calibs=fixed_calibs,
            calib_df=calib_df,
            seed=config.seed,
        )
    elif concept_learner_type == "oracle":
        from llm_prior_monitoring.oracle_components import (
            OracleConceptGenerator,
            OraclePoolConceptGenerator,
            OracleSparseFeatureExtractor,
            OracleState,
        )
        oracle_cfg = config.oracle
        assert oracle_cfg.pool_csvs and oracle_cfg.concept_cols, (
            "oracle learner needs oracle.pool_csvs and oracle.concept_cols"
        )
        state = OracleState(
            oracle_cfg.pool_csvs, oracle_cfg.concept_cols,
            text_column=oracle_cfg.pool_text_column,
        )
        if oracle_cfg.proposal_mode == "random_pool":
            # Mimics an LLM-driven propose/remove loop at zero LLM cost:
            # n_init random pool columns at burn-in, n_new random not-active
            # columns per step, removal enabled (removed columns recycle).
            assert prop_cfg.n_init_concepts <= len(oracle_cfg.concept_cols)
            return SubsetScanSCL(
                min_block=min_block,
                search_all_sizes=config.search_all_sizes,
                keyphrase_extractor=MockKeyphrase(),
                concept_generator=OraclePoolConceptGenerator(
                    state,
                    n_init_concepts=prop_cfg.n_init_concepts,
                    n_new_concepts_per_update=prop_cfg.n_new_concepts_per_update,
                    informative_cols=oracle_cfg.informative_cols,
                ),
                concept_selector=IdentityConceptSelector(),
                sparse_feature_extractor=OracleSparseFeatureExtractor(state),
                classifier=classifier,
                n_burn_in=config.n_burn_in,
                n_init_concepts=prop_cfg.n_init_concepts,
                n_new_concepts_per_update=prop_cfg.n_new_concepts_per_update,
                history_sample_size=config.history_sample_size,
                fixed_calibs=fixed_calibs,
                calib_df=calib_df,
                seed=config.seed,
            )
        # all_at_once: every column at burn-in, no proposal after
        return SubsetScanSCL(
            min_block=min_block,
            search_all_sizes=config.search_all_sizes,
            keyphrase_extractor=MockKeyphrase(),
            concept_generator=OracleConceptGenerator(state),
            concept_selector=IdentityConceptSelector(),
            sparse_feature_extractor=OracleSparseFeatureExtractor(state),
            classifier=classifier,
            n_burn_in=config.n_burn_in,
            n_init_concepts=len(oracle_cfg.concept_cols),
            n_new_concepts_per_update=0,
            history_sample_size=config.history_sample_size,
            fixed_calibs=fixed_calibs,
            calib_df=calib_df,
            seed=config.seed,
        )
    elif concept_learner_type == "mock":
        state = MockConceptState(
            p_informative=prop_cfg.p_informative,
            noise_rate=prop_cfg.noise_rate,
            weak_flip_rate=prop_cfg.weak_flip_rate,
            n_init_concepts=prop_cfg.n_init_concepts,
            n_new_concepts_per_update=prop_cfg.n_new_concepts_per_update,
        )
        selector = MockConceptSelector(state)
        return SubsetScanSCL(
            min_block=min_block,
            search_all_sizes=config.search_all_sizes,
            keyphrase_extractor=MockKeyphrase(),
            concept_generator=MockConceptGenerator(selector),
            concept_selector=selector,
            sparse_feature_extractor=MockSparseFeatureExtractor(state),
            classifier=classifier,
            n_burn_in=config.n_burn_in,
            n_init_concepts=prop_cfg.n_init_concepts,
            n_new_concepts_per_update=prop_cfg.n_new_concepts_per_update,
            history_sample_size=config.history_sample_size,
            fixed_calibs=fixed_calibs,
            calib_df=calib_df,
            seed=config.seed,
        )
    elif concept_learner_type == "bertopic_llm":
        # bertopic_llm — cluster keyphrases with BERTopic, rank topics by shift,
        # query LLM to formalize topics into yes/no concepts, annotate via LLM
        from llm_prior_monitoring.bertopic_state import BertopicState
        from llm_prior_monitoring.bertopic_llm_components import BertopicLLMConceptGenerator
        from llm_prior_monitoring.concept_selectors import RankedConceptSelector
        from llm_prior_monitoring.feature_extractors import SingleConceptSparseFeatureExtractor
        from HACHI.src.ensemble_trainer.config import LLMConfig
        from src.keyphrase import Keyphrase
        from src.keyphrase_config import KeyphraseConfig
        from src.llm_utils import create_llm_clients

        llm_config = LLMConfig(
            llm_iter_type=prop_cfg.llm_proposer.llm_model,
            llm_extraction_type=config.annotator.llm_annotator.llm_model,
            cache_file=cache_file,
        )
        llm_dict = create_llm_clients(llm_config)
        annotator_llm_dict = create_llm_clients(llm_config)

        keyphrase_config = KeyphraseConfig(
            llm_model_type=prop_cfg.llm_keyphrase.llm_model,
            prompt_file=prop_cfg.prompt_keyphrase_file,
            cache_file=cache_file,
            seed=config.seed,
            batch_size=prop_cfg.keyphrase_batch_size,
            num_new_tokens=prop_cfg.llm_keyphrase.max_new_tokens,
            prompt_placeholder=prop_cfg.keyphrase_prompt_placeholder,
            text_column=prop_cfg.keyphrase_text_column,
            is_image=config.is_image,
            index_col=0,
            llm_outputs_file=keyphrase_outputs_file,
            num_retries=0,
            retry_delay_seconds=60,
            log_file="_output/keyphrase.log",
            retry_prompt_file=prop_cfg.prompt_keyphrase_retry_file,
        )
        keyphrase_extractor = Keyphrase(keyphrase_config)
        
        bertopic_state = BertopicState(
            n_topics=prop_cfg.n_bertopic_topics,
            seed=config.seed,
            embedding_model=prop_cfg.embedding_model_path,
        )
        concept_generator = BertopicLLMConceptGenerator(
            state=bertopic_state,
            keyphrase_extractor=keyphrase_extractor,
            llm_dict=llm_dict,
            prompt_init_file=prop_cfg.prompt_proposer_init_file,
            prompt_iter_file=prop_cfg.prompt_proposer_iter_file,
            n_init_concepts=prop_cfg.n_init_concepts,
            n_new_concepts_per_update=prop_cfg.n_new_concepts_per_update,
            min_block=min_block,
        )

        bettor_extractor = SingleConceptSparseFeatureExtractor(
            annotator_llm_dict,
            prompt_file=config.annotator.prompt_annotator_file,
            is_image=config.is_image,
            max_new_tokens=config.annotator.llm_annotator.max_new_tokens,
            batch_size=config.annotator.batch_size,
            num_retries=config.annotator.llm_annotator.num_retries,
            retry_delay_seconds=config.annotator.llm_annotator.retry_delay_seconds,
            retry_prompt_file=config.annotator.prompt_annotator_retry_file,
        )

        return SubsetScanSCL(
            min_block=min_block,
            search_all_sizes=config.search_all_sizes,
            bettor_feature_extractor=bettor_extractor,
            sparse_feature_extractor=bettor_extractor,
            keyphrase_extractor=keyphrase_extractor,
            concept_generator=concept_generator,
            concept_selector=RankedConceptSelector(
                llm_dict,
                prompt_iter_file=prop_cfg.prompt_proposer_iter_file,
                prompt_init_file=prop_cfg.prompt_proposer_init_file,
                cv=prop_cfg.cv,
            ),
            classifier=classifier,
            fixed_calibs=fixed_calibs,
            calib_df=calib_df,
            n_burn_in=config.n_burn_in,
            n_init_concepts=prop_cfg.n_init_concepts,
            n_new_concepts_per_update=prop_cfg.n_new_concepts_per_update,
            real_query_budget_ceiling=ann_cfg.real_query_budget_ceiling,
            history_sample_size=config.history_sample_size,
            seed=config.seed,
        )
