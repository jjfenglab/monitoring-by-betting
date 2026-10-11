"""SAE-backed SCL subcomponents: SAE latents are the concepts (HypotheSAEs).

A TopK sparse autoencoder pretrained offline on the disjoint SAE-side pool
(src/train_sae.py) maps each observation's embedding to M latent activations.
Concepts are the neurons most discriminative for the pseudo-label y
(hypothesaes.select_neurons on the observed history); their activations are
deterministic functions of the embedding — no LLM calls anywhere (except
optional auto-interp naming of selected neurons, off by default).

Two variants share this module, selected by SAEConfig.refit_every:
  frozen (None): the offline checkpoint is used read-only for the whole stream.
  online (int):  every `refit_every` SCL updates the checkpoint is warm-started
                 (continued training, persistent Adam) on the side pool plus the
                 stream history through t-1. Neuron *indices* persist across
                 refits; their meanings drift.

Assembled by create_lgm_pipeline.make_sequential_concept_learner for
concept_learner_type == "sae", reusing MockKeyphrase, IdentityConceptSelector,
and IdentitySparseFeatureExtractor for the passthrough surfaces. SAEConceptProjector
maps embeddings to SAE activations.

Known limitation: embeddings are looked up by text (the generator only receives
data_df, never the host's `history` embeddings), so pool texts must map to a
unique embedding — SAEState asserts this at construction.
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from hypothesaes.sae import load_model
from hypothesaes.select_neurons import select_neurons
from torch.utils.data import DataLoader, TensorDataset


class SAEState:
    """Shared SAE model, pool text-to-embedding lookup, neuron registry, and warm-start refit."""

    def __init__(
        self,
        checkpoint_path: str,
        embeddings_path: str,
        mapping_path: str,
        *,
        side_embeddings_path: str | None = None,
        refit_every: int | None = None,
        refit_epochs: int = 5,
        refit_lr: float = 5e-4,
        refit_batch_size: int = 64,
    ):
        self.checkpoint_path = checkpoint_path
        self.sae = load_model(checkpoint_path, device="cpu")
        emb = np.load(embeddings_path)["emb"]
        mapping = pd.read_csv(mapping_path)
        self.text_to_emb: Dict[str, np.ndarray] = {}
        for text, row_index in zip(mapping["text"], mapping["row_index"]):
            e = emb[int(row_index)]
            prev = self.text_to_emb.get(text)
            # p0/p1 are independent draws over one population and may share texts;
            # batched encoding makes re-embedded duplicates differ at float32 level.
            assert prev is None or np.allclose(prev, e, atol=1e-5), (
                f"duplicate pool text with differing embeddings: {text!r}"
            )
            self.text_to_emb[text] = e
        self.neuron_idxs: List[int] = []  # selected neurons, concept-order
        self.refit_every = refit_every
        assert refit_every in [0,1,None]
        self.refit_epochs = refit_epochs
        self.refit_batch_size = refit_batch_size
        if refit_every is not None:
            assert side_embeddings_path is not None, "online SAE needs the side-pool embeddings"
            self.side_emb = np.load(side_embeddings_path)["emb"].astype(np.float32)
            assert self.side_emb.shape[1] == self.sae.input_dim
            # One optimizer for the whole stream: Adam moments carry across refits, so
            # each refit continues the previous one rather than restarting.
            self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=refit_lr)

    def activations(self, embeddings: np.ndarray) -> np.ndarray:
        """Full (n, M) SAE activation matrix for a batch of embeddings.

        hypothesaes' forward bumps its dead-neuron counter on every call, inference
        included, so the several scoring batches per step (history, x_t, the calib
        pool) would reset it for every neuron that fires and sae_dead_ratio would read
        a structural 0; snapshot/restore so it counts training batches only. The
        counter is a diagnostic here and nothing more: these checkpoints are trained
        with aux_k=None, so hypothesaes' dead-neuron revival term is absent and a
        neuron that stops winning the top-K gets no gradient to bring it back.
        """
        counter = self.sae.steps_since_activation.clone()
        acts = self.sae.get_activations(embeddings, show_progress=False)
        self.sae.steps_since_activation.copy_(counter)
        return acts

    def lookup(self, texts) -> np.ndarray:
        """Pool embeddings for the given texts; KeyError = text not in the pool."""
        return np.stack([self.text_to_emb[t] for t in texts])

    def refit(self, history_emb: np.ndarray) -> dict:
        """Warm-start the SAE on side pool ∪ history for refit_epochs; returns diagnostics.

        Reproduces hypothesaes' SparseAutoencoder.fit minus its data re-initialisation
        (fit() calls initialize_weights_ every time, so it cannot continue training).
        sae_refit_loss is the mean batch loss over the whole refit (all epochs).
        """
        train = np.concatenate([self.side_emb, history_emb.astype(np.float32)])
        assert len(train) >= self.refit_batch_size
        # drop_last: hypothesaes normalises each batch's MSE by that batch's own
        # variance, so a 1-row remainder batch divides by zero and poisons the weights.
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train)), batch_size=self.refit_batch_size,
            shuffle=True, drop_last=True,
        )
        self.sae.train()
        losses = []
        for _ in range(self.refit_epochs):
            for (x,) in loader:
                recon, info = self.sae(x)
                loss = self.sae.compute_loss(x, recon, info, aux_coef=1 / 32, multi_coef=0.0)
                # Per batch, before the step: one non-finite batch would poison the
                # persistent Adam moments for the rest of the stream.
                assert torch.isfinite(loss), "SAE refit batch loss is not finite"
                self.optimizer.zero_grad()
                loss.backward()
                self.sae.adjust_decoder_gradient_()
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                self.optimizer.step()
                self.sae.normalize_decoder_()
                losses.append(loss.item())
        self.sae.eval()
        mean_loss = float(np.mean(losses))
        dead = self.sae.steps_since_activation > self.sae.dead_neuron_threshold_steps
        return {
            "sae_refit_loss": mean_loss,
            "sae_dead_ratio": float(dead.float().mean()),
            "sae_n_train": len(train),
        }


class SAEConceptGenerator:
    """Proposes SAE neurons discriminative for the pseudo-label y as concepts."""

    def __init__(
        self,
        state: SAEState,
        selection_method: str = "correlation",
        interpret: bool = False,
        interpreter_model: str = "gpt-4.1-mini",
        text_column: str = "text",
    ):
        self.state = state
        self.selection_method = selection_method
        self.interpret = interpret
        self.interpreter_model = interpreter_model
        self._interpreter = None  # built on first _name_concepts call
        self.text_column = text_column
        self.config = SimpleNamespace(
            training=SimpleNamespace(num_greedy_holdout=0, train_frac=1.0),
            concept=SimpleNamespace(max_meta_concepts=0),
        )

    def generate_initial_concepts(self, init_seed, data_df: pd.DataFrame, concept_selector):
        """Select all SAE neurons as concepts."""
        acts = self.state.activations(self.state.lookup(data_df[self.text_column]))
        n_neurons = acts.shape[1]
        self.state.neuron_idxs = list(range(n_neurons))
        return [{"concept": f"sae_neuron_{i}", "neuron_idx": i} for i in range(n_neurons)]

    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[Dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """Propose new concepts."""
        # no new concepts, since they are already pointing to the right thing
        if self.state.refit_every is not None:
            # NOTE: assumes that refit_every == 1
            # just refit the concepts if asked to "generate new concepts"
            self.state.refit(self.state.lookup(init_data_df[self.text_column]))
        return []

    # def _select_new(self, data_df: pd.DataFrame, n_new: int) -> List[Dict]:
    #     """Top finite-scored neurons for y not already concepts; [] when none are eligible.

    #     Neurons with non-finite scores (dead/constant activations on this history) are
    #     never selected; if fewer than n_new eligible neurons remain, propose what is
    #     eligible. A single-class y is a hard error, not a silent all-NaN ranking.
    #     """
    #     acts = self.state.activations(self.state.lookup(data_df[self.text_column]))
    #     y = data_df["y"].to_numpy()
    #     assert len(np.unique(y)) >= 2, (
    #         "pseudo-label y has a single class: t_hat sits at an extreme of the "
    #         "history, so neuron scores are undefined. Callers must ensure the "
    #         "history straddles t_hat before proposing concepts."
    #     )
    #     taken = set(self.state.neuron_idxs)
    #     n_select = min(len(taken) + n_new, acts.shape[1])
    #     idxs, scores = select_neurons(
    #         acts, y, n_select=n_select, method=self.selection_method, classification=True,
    #     )
    #     n_dead = int(np.count_nonzero(~np.isfinite(scores)))
    #     if n_dead:
    #         logging.info(
    #             "sae concept selection: excluded %d dead (non-finite score) neurons", n_dead
    #         )
    #     new = [
    #         int(i) for i, s in zip(idxs, scores) if np.isfinite(s) and i not in taken
    #     ][:n_new]
    #     self.state.neuron_idxs.extend(new)
    #     concepts = [{"concept": f"sae_neuron_{i}", "neuron_idx": i} for i in new]
    #     if self.interpret and concepts:
    #         self._name_concepts(concepts, data_df[self.text_column].tolist(), acts)
    #     return concepts

    def _name_concepts(self, concepts: List[Dict], texts: List[str], acts: np.ndarray) -> None:
        """Auto-interp naming via HypotheSAEs' NeuronInterpreter (LLM calls; off by default).

        hypothesaes has its own OpenAI client, separate from the repo's Versa gateway:
        it needs OPENAI_KEY_SAE in the environment (OPENAI_BASE_URL optional, for an
        OpenAI-compatible gateway). Glosses are cached per (checkpoint, neuron, model)
        in neuron_names_<model>.json beside the checkpoint: a gloss describes the
        neuron, a fixed function of the frozen SAE, so every run of a dataset shares
        one set of names and the grid pays at most M naming calls total. Concurrent
        runs race benignly (worst case: duplicate calls). train_sae.py clears the file
        on retrain; delete it to force re-naming. Only meaningful for the frozen
        variant: under online refits a neuron's meaning drifts, so name the run's
        final checkpoint (run_monitor --sae-final-checkpoint) post hoc instead.
        """
        cache_path = (
            Path(self.state.checkpoint_path).parent
            / f"neuron_names_{self.interpreter_model}.json"
        )
        names: Dict[str, str] = (
            json.loads(cache_path.read_text()) if cache_path.exists() else {}
        )
        missing = [c["neuron_idx"] for c in concepts if str(c["neuron_idx"]) not in names]
        if missing:
            from hypothesaes.interpret_neurons import NeuronInterpreter

            if self._interpreter is None:
                self._interpreter = NeuronInterpreter(interpreter_model=self.interpreter_model)
            interpretations = self._interpreter.interpret_neurons(
                texts=texts, activations=acts, neuron_indices=missing,
            )
            for i in missing:
                candidates = [s for s in interpretations[i] if s]
                if candidates:
                    names[str(i)] = candidates[0]
            # Re-read + merge so a concurrent writer's names survive; pid-unique tmp +
            # os.replace keeps the file whole under parallel scons runs.
            if cache_path.exists():
                names = {**json.loads(cache_path.read_text()), **names}
            tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(names, indent=2))
            os.replace(tmp, cache_path)
        for c in concepts:
            gloss = names.get(str(c["neuron_idx"]))
            if gloss:
                c["concept"] = f"sae_neuron_{c['neuron_idx']}: {gloss}"


class SAEConceptProjector:
    """Projects embeddings to SAE activations of selected neurons."""

    def __init__(self, state: SAEState):
        self.state = state

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """SAE activations of selected neurons, shape (n, n_concepts)."""
        return self.state.activations(embeddings)[:, list(self.state.neuron_idxs)]
