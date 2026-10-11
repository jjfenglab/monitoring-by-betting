"""BERTopic-backed SCL subcomponents: cluster keyphrases into topics as concepts.

An intermediate approach between identity (raw embeddings) and full HACHI
(LLM-proposed concepts). Keyphrases are extracted via LLM, then clustered
with BERTopic. Topics become concepts; feature extraction uses BERTopic's
topic assignments (deterministic, no LLM annotation needed).

Flow:
1. Extract keyphrases from text (existing Keyphrase class, LLM-based)
2. Cluster all keyphrases with BERTopic (each keyphrase = one document)
3. Topic representations (top keywords) become concept definitions
4. Feature extraction: for each observation, assign its keyphrases to topics
   via BERTopic transform, create binary features (1 if any keyphrase in topic)
5. Changepoint estimation on the binary topic features

BERTopic is re-fit each iteration. Concept slots stay stable, but their
descriptions update based on the new clustering.
"""

import logging
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

from .feature_extractors import SparseExtractionMatrix

class Float64IncrementalPCA(IncrementalPCA):
    """IncrementalPCA that outputs float64 for sklearn compatibility."""
    
    def fit_transform(self, X, y=None):
        return super().fit_transform(X, y).astype(np.float64)
    
    def transform(self, X):
        return super().transform(X).astype(np.float64)


class BertopicState:
    """Shared BERTopic model state."""

    def __init__(
        self,
        n_topics: int = 25,
        seed: int = 0,
        num_pca: int = 5,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            n_topics: Target number of topics (BERTopic may return fewer).
            seed: Random seed for reproducibility.
            embedding_model: Sentence-transformers model name or path for embedding keyphrases.
        """
        self.n_topics = n_topics
        self.seed = seed
        self.embedding_model = embedding_model
        self.model_: BERTopic | None = None
        self.concept_refs_: List[Dict] = []
        self.all_topics_: List[int] = []
        self.all_keyphrases_: List[str] = []

        # Need underlying methods that allow online learning
        umap_model = Float64IncrementalPCA(n_components=num_pca)
        cluster_model = MiniBatchKMeans(n_clusters=self.n_topics, random_state=self.seed)
        vectorizer_model = OnlineCountVectorizer(stop_words="english")
        self.model_ = BERTopic(
            # n_gram_range=(1,3),
            # min_topic_size=5,
            # nr_topics=n_topics,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            # verbose=True,
            calculate_probabilities=False,
            embedding_model=self.embedding_model,
        )

    # def fit(self, keyphrases: List[str]) -> List[str]:
    #     """Fit BERTopic on individual keyphrases.

    #     Args:
    #         keyphrases: List of keyphrases (each treated as a document).

    #     Returns:
    #         List of topic description strings (one per topic found).
    #     """
    #     logging.info("fitting...")
    #     self.model_.fit(keyphrases)
    #     self.all_keyphrases_.extend(keyphrases)
    #     self.all_topics_.extend(self.model_.topics_)
    #     self.model_.topics_ = self.all_topics_
    #     topics = self.model_.get_topics()
        
    #     topic_info = self.model_.get_topic_info()
    #     topic_info = topic_info[topic_info["Topic"] != -1]

    #     descriptions = []
    #     for _, row in topic_info.iterrows():
    #         topic_id = row["Topic"]
    #         topic_words = self.model_.get_topic(topic_id)
    #         top_words = [word for word, _ in topic_words[:5]]
    #         keyphrases_in_topic = [
    #             kp for kp, t in zip(keyphrases, topics) if t == topic_id
    #         ]
    #         description = f"topic about: {', '.join(top_words)}"
    #         descriptions.append(description)

    #     logging.info(
    #         f"BertopicState.fit: {len(keyphrases)} keyphrases -> "
    #         f"{len(descriptions)} topics"
    #     )
    #     return descriptions


    def partial_fit(self, keyphrases: List[str]) -> List[str]:
        """Partial fit (online learning) BERTopic on individual keyphrases.

        Args:
            keyphrases: List of keyphrases (each treated as a document).

        Returns:
            List of topic description strings (one per topic found).
        """
        logging.info("fitting...")
        self.model_.partial_fit(keyphrases)
        self.all_keyphrases_.extend(keyphrases)
        self.all_topics_.extend(self.model_.topics_)
        self.model_.topics_ = self.all_topics_
        topics = self.model_.get_topics()

        topic_info = self.model_.get_topic_info()
        topic_info = topic_info[topic_info["Topic"] != -1]

        descriptions = []
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            topic_words = self.model_.get_topic(topic_id)
            top_words = [word for word, _ in topic_words[:5]]
            keyphrases_in_topic = [
                kp for kp, t in zip(keyphrases, topics) if t == topic_id
            ]
            description = f"topic about: {', '.join(top_words)}"
            descriptions.append(description)

        logging.info(
            f"BertopicState.partial_fit: {len(keyphrases)} keyphrases -> "
            f"{len(descriptions)} topics"
        )
        return descriptions

    def get_topic_keyphrases(self, topic_idx: int, keyphrases: List[str], topics: List[int]) -> List[str]:
        """Get keyphrases belonging to a topic."""
        return [kp for kp, t in zip(keyphrases, topics) if t == topic_idx]