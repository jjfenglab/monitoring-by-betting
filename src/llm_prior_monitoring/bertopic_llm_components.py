"""BERTopic + LLM concept generation: cluster keyphrases, rank topics by shift, formalize via LLM.

Combines BERTopic clustering with LLM-based concept formalization:
1. Extract keyphrases from text (existing Keyphrase class, LLM-based)
2. Cluster keyphrases with BERTopic (each keyphrase = one document)
3. Create binary topic features (presence/absence per observation)
4. Rank topics by before/after shift at estimated changepoint
5. Query LLM with top shifted topics + representative docs to formalize yes/no concepts
6. Annotation uses SingleConceptSparseFeatureExtractor

BERTopic is re-fit via partial_fit each iteration. Topic-to-concept mapping is tracked
to avoid re-proposing topics already formalized into concepts.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, FrozenSet, List, Set, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from .changepoint import MeanShiftChangepointLearner, SubsetScanChangepointLearner
from .bertopic_state import BertopicState
from .concept_selectors import CandidateConcepts
from src.llm_utils import SYSTEM_PROMPT, ResponseFormatError

def _get_keyphrases(keyphrase_field):
    return [kp.strip().lower() for kp in str(keyphrase_field).split(",") if kp]


class BertopicLLMConcept(BaseModel):
    definition: str = Field(description="The yes/no question defining the concept")
    topic_indices: List[int] = Field(description="BERTopic topic indices used for this concept")


class BertopicLLMCandidateConcepts(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning")
    concepts: List[BertopicLLMConcept] = Field(description="List of candidate concepts")


class BertopicLLMConceptGenerator:
    """Concept generator that clusters keyphrases via BERTopic, then formalizes via LLM.

    Flow:
    1. Fit/partial_fit BERTopic on keyphrases
    2. Build binary topic features per observation
    3. Estimate changepoint and rank topics by shift
    4. Query LLM with top shifted topics to formalize into yes/no concepts
    5. Track which topics have been used to avoid re-proposing
    """

    def __init__(
        self,
        state: BertopicState,
        keyphrase_extractor,
        llm_dict: Dict,
        prompt_init_file: str | Path,
        prompt_iter_file: str | Path,
        n_init_concepts: int = 10,
        n_new_concepts_per_update: int = 2,
        min_block: int = 5,
    ):
        """
        Args:
            state: Shared BertopicState holding the model.
            keyphrase_extractor: Keyphrase instance for LLM-based extraction.
            llm_dict: Dict with 'extraction' key pointing to LLM client.
            prompt_init_file: Path to initial concept proposal prompt template.
            prompt_iter_file: Path to iterative concept proposal prompt template.
            n_init_concepts: Number of concepts to generate at burn-in.
            n_new_concepts_per_update: Number of new concepts to propose per iteration.
            min_block: Minimum observations for changepoint detection.
        """
        self.bertopic_state = state
        self.keyphrase_extractor = keyphrase_extractor
        self.llm_dict = llm_dict
        self.prompt_init_file = Path(prompt_init_file)
        self.prompt_iter_file = Path(prompt_iter_file)
        self.n_init_concepts = n_init_concepts
        self.n_new_concepts_per_update = n_new_concepts_per_update
        self.min_block = min_block

        self._prompt_init_template = self._load_prompt_template(self.prompt_init_file)
        self._prompt_iter_template = self._load_prompt_template(self.prompt_iter_file)

        # Cached hierarchy dataframe from BERTopic (set by _build_topic_features)
        self._hierarchy_df: pd.DataFrame | None = None

        # Minimal config namespace for SCL compatibility
        self.config = SimpleNamespace(
            training=SimpleNamespace(num_greedy_holdout=0, train_frac=1.0),
            concept=SimpleNamespace(max_meta_concepts=n_init_concepts),
        )

    def _load_prompt_template(self, path: Path) -> str:
        with open(path, "r") as f:
            return f.read()
    
    def _extractions_to_matrix(
        self,
        extractions: Dict[str, np.ndarray],
        concepts: List[dict],
    ) -> np.ndarray:
        """
        TODO: repeat code. copied from changepoint_concept_generator.py
        Convert extractions dict to a (n_obs, n_concepts) matrix.

        Args:
            extractions: Dict of concept_name -> (n_obs, 1) array
            concepts: List of concept dicts with 'concept' key

        Returns:
            (n_obs, n_concepts) matrix, columns ordered by concepts list
        """
        if not concepts or not extractions:
            # No existing concepts - return empty matrix with correct n_obs
            n_obs = next(iter(extractions.values())).shape[0] if extractions else 0
            return np.zeros((n_obs, 0))

        columns = []
        for concept in concepts:
            concept_name = concept["concept"]
            if concept_name in extractions:
                col = extractions[concept_name].flatten()
            else:
                # Concept not in extractions - use zeros
                n_obs = next(iter(extractions.values())).shape[0]
                col = np.zeros(n_obs)
            columns.append(col)

        return np.column_stack(columns) if columns else np.zeros((0, 0))

    def generate_initial_concepts(
        self,
        init_seed: int,
        data_df: pd.DataFrame,
        concept_selector,
    ) -> List[Dict]:
        """Fit BERTopic, rank topics by shift, query LLM for formal concepts.

        Args:
            init_seed: Random seed (unused, kept for interface compatibility).
            data_df: DataFrame with 'llm_output' column containing keyphrases.
            concept_selector: Unused, kept for interface compatibility.

        Returns:
            List of concept dicts with 'concept' and 'topic_indices' keys.
        """
        all_keyphrases = self._collect_keyphrases(data_df)
        self.bertopic_state.partial_fit(all_keyphrases)
        self._update_topic_docs(all_keyphrases)

        num_candidate_topics = self.config.concept.max_meta_concepts * 2
        topic_features, all_topic_ids = self._build_topic_features(data_df)
        t_hat = self._find_changepoint_mean_shift(topic_features, n_sparse=num_candidate_topics)
        logging.info(f"BertopicLLM KEYPHRASE init: T_HAT={t_hat}, n_topics={len(all_topic_ids)}")

        top_topics_df = self._rank_topics_by_shift(topic_features, t_hat, all_topic_ids).head(num_candidate_topics)
        topics_text = self._format_topics_for_prompt(top_topics_df)

        prompt = self._prompt_init_template.format(
            topics_text=topics_text,
            max_meta_concepts=self.config.concept.max_meta_concepts,
        )
        logging.info(f"BertopicLLM init prompt:\n{prompt}")
        concepts = self._query_llm(prompt)
        concepts = concepts[:self.config.concept.max_meta_concepts]
        
        logging.info(
            f"BertopicLLMConceptGenerator.generate_initial_concepts: "
            f"{len(all_keyphrases)} keyphrases -> {len(concepts)} concepts, "
        )
        assert len(concepts) == self.config.concept.max_meta_concepts

        return concepts

    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[Dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """Re-fit BERTopic, rank unused topics by shift, query LLM for new concepts.

        Args:
            init_data_df: DataFrame with 'llm_output' column containing keyphrases.
            concept_selector: Unused, kept for interface compatibility.
            current_concepts: Existing concepts.
            current_extractions: Existing extractions (unused).

        Returns:
            List of new concept dicts.
        """
        all_keyphrases = self._collect_keyphrases(init_data_df)
        new_keyphrases = self._collect_keyphrases(init_data_df.iloc[-1:])
        topic_descriptions = self.bertopic_state.partial_fit(new_keyphrases)
        print("topic_descriptions", len(topic_descriptions))
        print(len(self.bertopic_state.model_.get_topics()))
        self._update_topic_docs(all_keyphrases)

        # Randomly drop one so refinement of old concepts can occur??
        rand_keep_concepts = np.random.choice(current_concepts, size=len(current_concepts) - 1, replace=False)
        rand_keep_concepts = rand_keep_concepts.tolist()
        
        num_candidate_topics = self.n_new_concepts_per_update * 2
        topic_features, all_topic_ids = self._build_topic_features(
            init_data_df,
            exclude_topics=[concept["topic_indices"] for concept in current_concepts]
        )
        X_concepts = self._extractions_to_matrix(current_extractions, rand_keep_concepts)
        agg_features = np.hstack([X_concepts, topic_features])
        t_hat = self._find_changepoint_mean_shift(agg_features, n_sparse=num_candidate_topics + self.config.concept.max_meta_concepts)
        logging.info(f"BertopicLLM KEYPHRASE iter: T_HAT={t_hat}, n_topics={len(all_topic_ids)}")

        top_topics_df = self._rank_topics_by_shift(topic_features, t_hat, all_topic_ids).head(num_candidate_topics)
        topics_text = self._format_topics_for_prompt(top_topics_df)

        meta_concepts_text = "\n".join(
            f"* X{i} = {c['concept']}" for i, c in enumerate(rand_keep_concepts)
        )
        prompt = self._prompt_iter_template.format(
            meta_concepts=meta_concepts_text,
            topics_text=topics_text,
            max_meta_concepts=self.n_new_concepts_per_update,
            num_concepts_fixed=len(rand_keep_concepts),
            num_concepts=len(rand_keep_concepts) + 1,
        )
        logging.info(f"BertopicLLM iter prompt:\n{prompt}")
        print(prompt)

        concepts = self._query_llm(prompt)

        logging.info(
            f"BertopicLLMConceptGenerator.generate_candidate_concepts: "
            f"generated {len(concepts)} new concepts"
        )
        return concepts

    def _collect_keyphrases(self, data_df: pd.DataFrame) -> List[str]:
        """Collect individual keyphrases from all observations."""
        keyphrases = []
        for llm_output in data_df["llm_output"].fillna(""):
            keyphrases += _get_keyphrases(llm_output)
        return keyphrases

    def _update_topic_docs(self, keyphrases: List[str]) -> None:
        """Update topic-to-docs mapping by transforming keyphrases and recording assignments.

        Since partial_fit doesn't populate BERTopic's representative_docs_, we track them manually.
        """
        if not keyphrases:
            return

        self._topic_to_docs = {}
        topics, _ = self.bertopic_state.model_.transform(keyphrases)
        for kp, topic_id in zip(keyphrases, topics):
            if topic_id < 0:
                continue
            if topic_id not in self._topic_to_docs:
                self._topic_to_docs[topic_id] = []
            self._topic_to_docs[topic_id].append(kp)

    def _build_topic_features(
        self,
        data_df: pd.DataFrame,
        exclude_topics: List[List[int]] | None = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Build binary topic feature matrix from observations using hierarchical topics.

        Creates features for both leaf topics AND merged topics in the hierarchy.
        This allows changepoint detection to find shifts at any level of topic granularity.

        Args:
            data_df: DataFrame with 'llm_output' column containing keyphrases.
            exclude_topics: List of topic index lists to exclude (e.g., from existing concepts).

        Returns:
            Tuple of:
                - (n_obs, n_all_topics) binary matrix covering leaves + hierarchy nodes
                - List of topic IDs (leaf IDs first, then merged parent IDs)
        """
        if self.bertopic_state.model_ is None:
            raise ValueError("BERTopic model not fitted")

        topic_info = self.bertopic_state.model_.get_topic_info()
        leaf_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

        all_keyphrases = self._collect_keyphrases(data_df)

        # create full mapping of nodes to topics
        # self._hierarchy_df = self.bertopic_state.model_.hierarchical_topics(all_keyphrases)
        # self._hierarchy_df["Parent_ID"] = self._hierarchy_df["Parent_ID"].astype(int)
        # # add in leaf nodes to this hierarhcy df
        leaf_rows = [{"Parent_ID": t, "Topics": [t]} for t in leaf_topics]
        # self._hierarchy_df = pd.concat(
        #     [self._hierarchy_df, pd.DataFrame(leaf_rows)], ignore_index=True
        # )
        self._hierarchy_df = pd.DataFrame(leaf_rows)

        # Flatten exclude_topics into a set of hierarchy row indices to skip
        excluded_indices: Set[int] = set()
        if exclude_topics:
            excluded_topic_set = set()
            for topic_list in exclude_topics:
                excluded_topic_set.update(topic_list)
            for idx, row in self._hierarchy_df.iterrows():
                if row["Parent_ID"] in excluded_topic_set:
                    excluded_indices.add(idx)

        leaf_topics_set = set(leaf_topics)
        parent_to_leaves = {}
        for _, row in self._hierarchy_df.iterrows():
            parent_id = row["Parent_ID"]
            parent_to_leaves[parent_id] = set(row["Topics"])

        # Exclude leaf self-references from parent list to avoid duplicates
        # Map each leaf topic to columns for itself and all parent topics containing it
        topic_to_cols = {t: [i] for i, t in enumerate(leaf_topics)}
        n_leaves = len(leaf_topics)
        for parent_col_idx, (parent_id, leaves) in enumerate(parent_to_leaves.items()):
            for leaf in leaves:
                topic_to_cols[leaf].append(parent_id)

        features = np.zeros((len(data_df), len(self._hierarchy_df)), dtype=np.float32)
        for i, (_, row) in enumerate(data_df.iterrows()):
            obs_keyphrases = _get_keyphrases(row["llm_output"])
            if not obs_keyphrases:
                continue

            topics, _ = self.bertopic_state.model_.transform(obs_keyphrases)
            obs_leaf_topics = [t for t in topics if t >= 0]
            for leaf_topic in obs_leaf_topics:
                features[i, topic_to_cols[leaf_topic]] = 1.0

        # Return Parent_ID values (not row indices) for downstream lookups
        all_parent_ids = self._hierarchy_df["Parent_ID"].values
        if excluded_indices:
            all_row_indices = np.arange(len(all_parent_ids))
            keep_mask = ~np.isin(all_row_indices, list(excluded_indices))
            features = features[:, keep_mask]
            all_parent_ids = all_parent_ids[keep_mask]

        return features, all_parent_ids

    def _find_changepoint_mean_shift(self, X: np.ndarray, n_sparse: int) -> int:
        """Find best changepoint via mean shift statistic on topic features."""
        keyphrase_cp_learner = SubsetScanChangepointLearner(
            MeanShiftChangepointLearner(min_block=self.min_block),
            subset_size=n_sparse,
            search_all_sizes=True, #False
        )

        n = X.shape[0]
        if n < 2 * self.min_block:
            return n // 2

        best_split, best_stat, best_subset = keyphrase_cp_learner.estimate_with_subset(X)
        return best_split

    def _rank_topics_by_shift(
        self, X: np.ndarray, t_hat: int, all_topic_ids: List[int]
    ) -> pd.DataFrame:
        """Rank topics (leaves + hierarchy) by their before/after prevalence shift.

        Args:
            X: (n_obs, n_all_topics) feature matrix.
            t_hat: Estimated changepoint.
            all_topic_ids: List of topic IDs corresponding to columns in X.

        Returns:
            DataFrame with columns ['topic_idx', 'shift', 'abs_shift', 'is_hierarchical']
            sorted by abs_shift.
        """
        mean_before = X[:t_hat].mean(axis=0)
        mean_after = X[t_hat:].mean(axis=0)
        shifts = mean_after - mean_before

        topic_info = self.bertopic_state.model_.get_topic_info()
        leaf_topics = set(topic_info[topic_info["Topic"] != -1]["Topic"].tolist())

        records = []
        for i, topic_id in enumerate(all_topic_ids):
            records.append({
                "topic_idx": topic_id,
                "shift": shifts[i],
                "abs_shift": np.abs(shifts[i]),
                "is_hierarchical": topic_id not in leaf_topics,
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = df.sort_values("abs_shift", ascending=False)
        return df

    def _get_topic_signature(self, topic_idx: int) -> str:
        """Get a stable signature for a topic based on its representative docs.

        Works for both leaf topics and hierarchical (merged) topics.
        """
        doc_counts_dict = self._get_topic_representative_docs(topic_idx)
        return "+".join(list(doc_counts_dict.keys()))

    def _get_topic_representative_docs(self, topic_idx: int) -> List[Tuple[str, int]]:
        """Get representative docs for a topic (leaf or hierarchical) with frequency counts.

        Returns:
            List of (doc, count) tuples sorted by count descending.
        """
        hier_row = self._hierarchy_df[self._hierarchy_df["Parent_ID"] == topic_idx]
        descendant_leaves = hier_row.iloc[0]["Topics"]
        all_docs = Counter()
        for leaf_id in descendant_leaves:
            if leaf_id in self._topic_to_docs:
                all_docs.update(self._topic_to_docs[leaf_id])

        return sorted(all_docs.items(), key=lambda x: -x[1])

    def _format_topics_for_prompt(self, top_topics_df: pd.DataFrame) -> str:
        """Format top topics with representative docs and frequency counts for LLM prompt."""
        lines = []

        for _, row in top_topics_df.iterrows():
            topic_idx = int(row["topic_idx"])
            shift = row["shift"]
            is_hier = row.get("is_hierarchical", False)
            sign = "+" if shift >= 0 else ""

            topic_type = "merged topic" if is_hier else "topic"
            lines.append(f"{topic_type} {topic_idx} (shift: {sign}{shift:.2f}):")

            doc_counts = self._get_topic_representative_docs(topic_idx)
            for j, (doc, count) in enumerate(doc_counts[:7]):
                lines.append(f"  - \"{doc}\" (n={count})")
            lines.append("")

        return "\n".join(lines)

    def _query_llm(self, prompt: str) -> List[Dict]:
        """Query LLM for candidate concepts with associated topic indices."""
        llm = self.llm_dict["extraction"]
        try:
            llm_response = llm.run(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                response_format=BertopicLLMCandidateConcepts,
                strict_response_format=True,
            )
            return [
                {"concept": c.definition, "topic_indices": c.topic_indices}
                for c in llm_response.concepts
            ]
        except ResponseFormatError as e:
            logging.warning("LLM call failed for BertopicLLMCandidateConcepts: %s", e)
            logging.warning("Full LLM output for BertopicLLMCandidateConcepts: %s", e.raw_content)
            try:
                # second attempt
                llm_response = llm.run(
                    prompt,
                    max_tokens=2048,
                    response_format=BertopicLLMCandidateConcepts,
                    strict_response_format=True,
                    temperature=1,
                )
                return [
                    {"concept": c.definition, "topic_indices": c.topic_indices}
                    for c in llm_response.concepts
                ]
            except ResponseFormatError as e:
                logging.warning("LLM call failed for BertopicLLMCandidateConcepts: %s", e)
                logging.warning("Full LLM output for BertopicLLMCandidateConcepts: %s", e.raw_content)
                return []