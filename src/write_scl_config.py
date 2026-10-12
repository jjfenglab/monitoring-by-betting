"""Write an SCLConfig JSON file from command-line arguments.

Used by sconscript to generate per-run config files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_prior_monitoring.scl_config import (
    AnnotatorConfig,
    ClassifierConfig,
    ConceptProposerConfig,
    LLMConfig,
    OracleConfig,
    SAEConfig,
    SCLConfig,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    # Top-level
    p.add_argument("--concept-learner-type", required=True, choices=["identity", "mock", "sae", "oracle", "bertopic_llm"])
    p.add_argument("--scl-variant", default="subset_scan", choices=["subset_scan"])
    p.add_argument("--n-burn-in", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--is-image", action="store_true",
                   help="Observations are image paths; LLM calls attach the image as a content part.")
    p.add_argument("--search-all-sizes", action="store_true",
                   help="Search over subset sizes 1..K with penalty (stat - k) instead of fixed size K.")
    # Concept proposer
    p.add_argument("--n-init-concepts", type=int, default=5)
    p.add_argument("--n-new-concepts-per-update", type=int, default=2)
    p.add_argument("--p-informative", type=float, default=0.8)
    p.add_argument("--noise-rate", type=float, default=0.0)
    p.add_argument("--weak-flip-rate", type=float, default=0.3)
    p.add_argument("--prompt-keyphrase-file", default=None)
    p.add_argument("--prompt-keyphrase-retry-file", default=None)
    p.add_argument("--keyphrase-batch-size", type=int, default=20)
    p.add_argument("--prompt-proposer-iter-file", default=None)
    p.add_argument("--prompt-proposer-init-file", default=None)
    # BERTopic LLM config
    p.add_argument("--n-bertopic-topics", type=int, default=50,
                   help="Number of BERTopic topics (can exceed n-init-concepts for bertopic_llm).")
    p.add_argument("--num-top-topics", type=int, default=10,
                   help="Number of top shifted topics to show in LLM prompt for bertopic_llm.")
    # Annotator
    p.add_argument("--prompt-annotator-file", default=None)
    p.add_argument("--prompt-annotator-retry-file", default=None)
    p.add_argument("--annotator-batch-size", type=int, default=20)
    p.add_argument("--single-concept-mode", action="store_true",
                   help="Each (obs, concept) pair in the data stream history gets its own LLM call instead of "
                        "batching multiple concepts per observation. All obs in calibration set and new obs get single concept mode regardless.")
    # Classifier
    p.add_argument("--classifier-type", default="logistic", choices=["logistic"])
    p.add_argument("--classifier-l1-ratio", type=float, default=0.0)
    p.add_argument("--max-unpenalized-dim", type=int, default=10,
                   help="Fit an unpenalized logistic regression when the classifier's "
                        "input width (n concepts, or embedding dim for identity) is at "
                        "most this; ridge above it.")
    # SAE concept learner
    p.add_argument("--sae-checkpoint-path", default=None)
    p.add_argument("--sae-embeddings-path", default=None)
    p.add_argument("--sae-mapping-path", default=None)
    p.add_argument("--sae-selection-method", default="correlation",
                   choices=["correlation", "lasso", "separation_score"])
    p.add_argument("--sae-refit-every", type=int, default=None,
                   help="Online SAE: warm-start every N SCL updates (None = frozen)")
    p.add_argument("--sae-refit-epochs", type=int, default=5)
    p.add_argument("--sae-refit-lr", type=float, default=5e-4)
    p.add_argument("--sae-refit-batch-size", type=int, default=64)
    p.add_argument("--sae-side-embeddings-path", default=None,
                   help="SAE-side pool embeddings .npz, kept in every online refit")
    # Oracle concept learner
    p.add_argument("--oracle-pool-csvs", nargs="+", default=[])
    p.add_argument("--oracle-concept-cols", nargs="+", default=[])
    p.add_argument("--oracle-pool-text-column", default="observation")
    p.add_argument("--oracle-variant", default="",
                   help="Sweep-cell label recorded in the config (matches the clt_* dir).")
    p.add_argument("--oracle-proposal-mode", default="all_at_once",
                   choices=["all_at_once", "random_pool"])
    p.add_argument("--oracle-informative-cols", nargs="*", default=[])
    # LLM cache mode: one shared per-dataset cache, or a per-stream copy seeded from it
    p.add_argument("--cache-mode", default="shared", choices=["shared", "per_stream"])
    # LLM: keyphrase
    p.add_argument("--llm-keyphrase-model", default="")
    p.add_argument("--llm-keyphrase-max-new-tokens", type=int, default=5000)
    # LLM: proposer
    p.add_argument("--llm-proposer-model", default="")
    p.add_argument("--llm-proposer-max-new-tokens", type=int, default=5000)
    # LLM: annotator
    p.add_argument("--llm-annotator-model", default="")
    p.add_argument("--llm-annotator-max-new-tokens", type=int, default=5000)
    # LLM: retry settings (shared across all LLM configs)
    p.add_argument("--llm-num-retries", type=int, default=2,
                   help="Number of retries for LLM API calls on rate limit errors.")
    p.add_argument("--llm-retry-delay-seconds", type=int, default=60,
                   help="Seconds to wait between retries.")

    args = p.parse_args()

    config = SCLConfig(
        concept_learner_type=args.concept_learner_type,
        scl_variant=args.scl_variant,
        n_burn_in=args.n_burn_in,
        seed=args.seed,
        cache_mode=args.cache_mode,
        is_image=args.is_image,
        search_all_sizes=args.search_all_sizes,
        concept_proposer=ConceptProposerConfig(
            n_init_concepts=args.n_init_concepts,
            n_new_concepts_per_update=args.n_new_concepts_per_update,
            p_informative=args.p_informative,
            noise_rate=args.noise_rate,
            weak_flip_rate=args.weak_flip_rate,
            prompt_keyphrase_file=args.prompt_keyphrase_file,
            prompt_keyphrase_retry_file=args.prompt_keyphrase_retry_file,
            keyphrase_batch_size=args.keyphrase_batch_size,
            prompt_proposer_iter_file=args.prompt_proposer_iter_file,
            prompt_proposer_init_file=args.prompt_proposer_init_file,
            n_bertopic_topics=args.n_bertopic_topics,
            llm_keyphrase=LLMConfig(
                llm_model=args.llm_keyphrase_model,
                max_new_tokens=args.llm_keyphrase_max_new_tokens,
                num_retries=args.llm_num_retries,
                retry_delay_seconds=args.llm_retry_delay_seconds,
            ),
            llm_proposer=LLMConfig(
                llm_model=args.llm_proposer_model,
                max_new_tokens=args.llm_proposer_max_new_tokens,
                num_retries=args.llm_num_retries,
                retry_delay_seconds=args.llm_retry_delay_seconds,
            ),
        ),
        annotator=AnnotatorConfig(
            prompt_annotator_file=args.prompt_annotator_file,
            prompt_annotator_retry_file=args.prompt_annotator_retry_file,
            batch_size=args.annotator_batch_size,
            llm_annotator=LLMConfig(
                llm_model=args.llm_annotator_model,
                max_new_tokens=args.llm_annotator_max_new_tokens,
                num_retries=args.llm_num_retries,
                retry_delay_seconds=args.llm_retry_delay_seconds,
            ),
            single_concept_mode=args.single_concept_mode,
        ),
        classifier=ClassifierConfig(
            classifier_type=args.classifier_type,
            max_unpenalized_dim=args.max_unpenalized_dim,
            l1_ratio=args.classifier_l1_ratio,
        ),
        sae=SAEConfig(
            checkpoint_path=args.sae_checkpoint_path,
            embeddings_path=args.sae_embeddings_path,
            mapping_path=args.sae_mapping_path,
            selection_method=args.sae_selection_method,
            refit_every=args.sae_refit_every,
            refit_epochs=args.sae_refit_epochs,
            refit_lr=args.sae_refit_lr,
            refit_batch_size=args.sae_refit_batch_size,
            side_embeddings_path=args.sae_side_embeddings_path,
        ),
        oracle=OracleConfig(
            pool_csvs=tuple(args.oracle_pool_csvs),
            concept_cols=tuple(args.oracle_concept_cols),
            pool_text_column=args.oracle_pool_text_column,
            variant=args.oracle_variant,
            proposal_mode=args.oracle_proposal_mode,
            informative_cols=tuple(args.oracle_informative_cols),
        ),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config.to_json(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
