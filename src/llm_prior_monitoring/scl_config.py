"""Configuration dataclasses for SequentialConceptLearner components."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

@dataclass
class LLMConfig:
    """Config for a single LLM endpoint."""
    llm_model: str = ""
    max_new_tokens: int = 5000
    num_retries: int = 2
    retry_delay_seconds: int = 60

@dataclass
class ConceptProposerConfig:
    """Config for concept generation and lifecycle management.

    Includes keyphrase extraction and concept-generator config for the LLM-based
    bertopic_llm arm.
    """
    cv: int = 3
    n_init_concepts: int = 5
    n_new_concepts_per_update: int = 2
    # mock mode only
    p_informative: float = 0.8 # unused since fixed strong/weak/noise mock batches
    noise_rate: float = 0.1 # how much noise is in an informative concept
    weak_flip_rate: float = 0.3 # P(weak concept != its strong partner's latent)
    # concept lifecycle (removal after freeze)
    freeze_time: int | None = None
    removal_rule: Literal["none", "importance", "greedy", "random"] = "none"
    rf_importance: Literal["permutation", "impurity"] = "permutation"
    greedy_group_size: int = 1
    greedy_max_group_candidates: int = 50
    removal_n_jobs: int = 1  # joblib workers for greedy candidate scoring
    # keyphrase extraction (bertopic_llm mode)
    prompt_keyphrase_file: str | None = None
    prompt_keyphrase_retry_file: str | None = None
    keyphrase_batch_size: int = 20
    keyphrase_prompt_placeholder: str = "{note}"
    keyphrase_text_column: str = "text"
    # concept generator / selector (bertopic_llm mode) — prompt files
    prompt_proposer_iter_file: str | None = None
    prompt_proposer_init_file: str | None = None
    # LLM configs for bertopic_llm mode
    llm_keyphrase: LLMConfig = field(default_factory=LLMConfig)
    llm_proposer: LLMConfig = field(default_factory=LLMConfig)
    # BERTopic embedding model (local path to avoid downloading)
    embedding_model_path: str | None = "local_all_MiniLM_L6_v2"
    # bertopic_llm mode: number of BERTopic topics (can exceed n_init_concepts)
    n_bertopic_topics: int = 25

@dataclass
class AnnotationSelectorConfig:
    """Config for annotation-related settings."""
    # Safety ceiling for real LLM queries per step
    real_query_budget_ceiling: int = 500


@dataclass
class AnnotatorConfig:
    """Config for the annotation/extraction process.

    bertopic_llm mode uses sparse_feature_extractor with an LLM prompt file.
    Mock mode generates synthetic labels without config.
    """
    prompt_annotator_file: str | None = None
    prompt_annotator_retry_file: str | None = None
    batch_size: int = 20  # prompts per LLM batch (= max parallel requests)
    llm_annotator: LLMConfig = field(default_factory=LLMConfig)
    # single_concept_mode: if True, each (obs, concept) pair gets its own LLM call
    # instead of batching multiple concepts per observation into one call
    single_concept_mode: bool = False


@dataclass
class ImputerConfig:
    """Config for the concept imputer (fills missing labels from embeddings)."""
    hidden_dims_grid: tuple[tuple[int, ...], ...] = ((64, 32),)
    epochs_grid: tuple[int, ...] = (50,)
    batch_size: int = 32
    lr: float = 1e-3
    # Sweep-cell label, matching the imp_* output directory. Carried so summaries can
    # tell tuning variants apart without reconstructing the grids.
    variant: str = "off"


@dataclass
class ClassifierConfig:
    """Config for the classifier"""
    classifier_type: Literal["logistic", "random_forest", "direction_fixed"] = "logistic"
    seed: int = 0
    max_unpenalized_dim: int = 10
    l1_ratio: float = 0.0
    cv: bool = False
    cv_folds: int = 5
    cv_n_cs: int = 10
    rf_n_estimators: int = 50
    rf_n_iter: int = 6

@dataclass
class SAEConfig:
    """Config for the SAE concept learner: checkpoint + pool files for text-to-embedding lookup."""
    checkpoint_path: str | None = None
    embeddings_path: str | None = None
    mapping_path: str | None = None
    selection_method: Literal["correlation", "lasso", "separation_score"] = "correlation"
    interpret: bool = False
    # hypothesaes' own OpenAI client (env OPENAI_KEY_SAE), not the Versa gateway
    interpreter_model: str = "gpt-4.1-mini"
    # Sweep-cell label, derived from refit_every in __post_init__ (kept as a field so
    # it round-trips through the JSON); summaries read it to tell the arms apart.
    variant: str = "frozen"
    # Online variant: warm-start the checkpoint every `refit_every` SCL updates on
    # side_embeddings_path (the disjoint SAE-side pool) + the stream history through
    # t-1. None = frozen checkpoint. lr/batch default to train_sae.py's offline values.
    refit_every: int | None = None
    refit_epochs: int = 5
    refit_lr: float = 5e-4
    refit_batch_size: int = 64
    side_embeddings_path: str | None = None

    def __post_init__(self):
        self.variant = "online" if self.refit_every is not None else "frozen"
        assert self.refit_every is None or self.side_embeddings_path is not None, (
            "online SAE needs side_embeddings_path"
        )
        assert self.refit_epochs >= 1
        assert not (self.interpret and self.refit_every is not None), (
            "auto-interp glosses are cached per checkpoint; the online SAE drifts, so "
            "name its final checkpoint (run_monitor --sae-final-checkpoint) post hoc"
        )


@dataclass
class OracleConfig:
    """Config for the oracle concept learner: ground-truth columns from pool CSVs."""
    pool_csvs: tuple[str, ...] = ()
    concept_cols: tuple[str, ...] = ()
    pool_text_column: str = "observation"
    # Sweep-cell label (e.g. "oracle_true"); summaries use it to tell variants apart.
    variant: str = ""
    # all_at_once: every column at burn-in, no proposal after, removal unsupported.
    # random_pool: n_init random columns at burn-in, then random proposal from the
    # not-currently-active pool each step; removal supported (removed columns recycle).
    proposal_mode: Literal["all_at_once", "random_pool"] = "all_at_once"
    # Columns stamped is_informative=True at proposal (random_pool only); feeds the
    # frac_noninformative_* debug columns.
    informative_cols: tuple[str, ...] = ()


@dataclass
class SCLConfig:
    """Top-level config for SequentialConceptLearner.

    Maps to the main responsibilities:
    0. concept_proposer: generate/manage concepts
    1. annotator: perform extraction (config minimal; LLM arms use injected objects)
    2. classifier: train classifier + form null reference
    """
    concept_learner_type: Literal["identity", "mock", "sae", "oracle", "bertopic_llm"]
    # subset_scan: joint feature selection and changepoint detection via subset scan.
    scl_variant: Literal["subset_scan"] = "subset_scan"
    n_burn_in: int = 30
    seed: int = 0
    cache_file: str = "cache_v1api.db" # DEPRECATED
    cache_mode: Literal["shared", "per_stream"] = "shared"
    # observations are image paths: LLM calls attach the image as a content
    # part, and the text column carries the path (dataset-level property, so it
    # governs both keyphrase extraction and concept annotation)
    is_image: bool = False
    # subset scan: if True, search over subset sizes 1..K with penalty (stat - k)
    search_all_sizes: bool = False
    # annotation-based learners: each update proposes, annotates, scans and fits on a
    # fresh uniform sample of this many history rows, so annotation cost is linear in
    # t instead of quadratic. None = full history (the default behavior).
    history_sample_size: int | None = None
    concept_proposer: ConceptProposerConfig = field(default_factory=ConceptProposerConfig)
    annotation_selector: AnnotationSelectorConfig = field(default_factory=AnnotationSelectorConfig)
    annotator: AnnotatorConfig = field(default_factory=AnnotatorConfig)
    imputer: ImputerConfig = field(default_factory=ImputerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)

    def to_json(self, path: str | Path) -> None:
        """Write config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SCLConfig":
        """Load config from JSON file."""
        with open(path) as f:
            d = json.load(f)

        cp = d["concept_proposer"]
        concept_proposer = ConceptProposerConfig(
            n_init_concepts=cp["n_init_concepts"],
            n_new_concepts_per_update=cp["n_new_concepts_per_update"],
            p_informative=cp["p_informative"],
            noise_rate=cp["noise_rate"],
            weak_flip_rate=cp["weak_flip_rate"],
            freeze_time=cp["freeze_time"],
            removal_rule=cp["removal_rule"],
            rf_importance=cp["rf_importance"],
            greedy_group_size=cp["greedy_group_size"],
            greedy_max_group_candidates=cp["greedy_max_group_candidates"],
            removal_n_jobs=cp["removal_n_jobs"],
            prompt_keyphrase_file=cp["prompt_keyphrase_file"],
            prompt_keyphrase_retry_file=cp.get("prompt_keyphrase_retry_file"),
            keyphrase_batch_size=cp["keyphrase_batch_size"],
            keyphrase_prompt_placeholder=cp["keyphrase_prompt_placeholder"],
            keyphrase_text_column=cp["keyphrase_text_column"],
            prompt_proposer_iter_file=cp["prompt_proposer_iter_file"],
            prompt_proposer_init_file=cp["prompt_proposer_init_file"],
            llm_keyphrase=LLMConfig(**cp["llm_keyphrase"]),
            llm_proposer=LLMConfig(**cp["llm_proposer"]),
            embedding_model_path=cp.get("embedding_model_path", "local_all_MiniLM_L6_v2"),
            n_bertopic_topics=cp["n_bertopic_topics"],
        )

        ann = d["annotator"]
        annotator = AnnotatorConfig(
            prompt_annotator_file=ann["prompt_annotator_file"],
            prompt_annotator_retry_file=ann.get("prompt_annotator_retry_file"),
            batch_size=ann["batch_size"],
            llm_annotator=LLMConfig(**ann["llm_annotator"]),
            single_concept_mode=ann.get("single_concept_mode", False),
        )

        imp = d["imputer"]
        imputer = ImputerConfig(
            hidden_dims_grid=tuple(tuple(x) for x in imp["hidden_dims_grid"]),
            epochs_grid=tuple(imp["epochs_grid"]),
            batch_size=imp["batch_size"],
            lr=imp["lr"],
            variant=imp["variant"],
        )

        oracle_d = d.get("oracle", {})
        oracle = OracleConfig(
            pool_csvs=tuple(oracle_d.get("pool_csvs", ())),
            concept_cols=tuple(oracle_d.get("concept_cols", ())),
            pool_text_column=oracle_d.get("pool_text_column", "observation"),
            variant=oracle_d.get("variant", ""),
            proposal_mode=oracle_d.get("proposal_mode", "all_at_once"),
            informative_cols=tuple(oracle_d.get("informative_cols", ())),
        )

        return cls(
            concept_learner_type=d["concept_learner_type"],
            scl_variant=d["scl_variant"],
            n_burn_in=d["n_burn_in"],
            seed=d["seed"],
            cache_mode=d["cache_mode"],
            # migration: configs written before is_image was hoisted to the top
            # level carry it as concept_proposer.keyphrase_is_image; text-only
            # configs written before either field existed carry neither
            is_image=d.get("is_image", cp.get("keyphrase_is_image", False)),
            search_all_sizes=d.get("search_all_sizes", False),
            history_sample_size=d.get("history_sample_size"),
            concept_proposer=concept_proposer,
            annotation_selector=AnnotationSelectorConfig(
                real_query_budget_ceiling=d["annotation_selector"].get("real_query_budget_ceiling", 500),
            ),
            annotator=annotator,
            imputer=imputer,
            classifier=ClassifierConfig(**d["classifier"]),
            sae=SAEConfig(**d.get("sae", {})),
            oracle=oracle,
        )
