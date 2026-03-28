"""KNN baseline: predict expression by averaging similar training perturbations.

Gene names are embedded with a lightweight sentence-transformer model
("all-MiniLM-L6-v2") to obtain semantic similarity between perturbation names,
allowing the baseline to generalise to unseen genes that share functional
context with training perturbations.
"""

from typing import List

import numpy as np


class KNNBaseline:
    """Predict expression by averaging the K most similar training perturbations.

    Similarity is measured in the embedding space of perturbation gene names
    produced by the ``all-MiniLM-L6-v2`` sentence-transformer.  For each test
    perturbation the model returns the (weighted) mean expression profile of
    the K nearest training perturbations.

    Parameters
    ----------
    k:
        Default number of nearest neighbours to average.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._train_perts: List[str] = []
        self._train_profiles: np.ndarray = np.empty(0)
        self._train_embeddings: np.ndarray = np.empty(0)
        self._embedder = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        train_perts: List[str],
        train_profiles: np.ndarray,
    ) -> None:
        """Store training perturbation profiles and pre-compute embeddings.

        Parameters
        ----------
        train_perts:
            List of training perturbation gene names (length N_train).
        train_profiles:
            Mean expression profiles for each training perturbation,
            shape ``(N_train, n_genes)``.
        """
        if len(train_perts) != train_profiles.shape[0]:
            raise ValueError(
                f"train_perts has {len(train_perts)} entries but "
                f"train_profiles has {train_profiles.shape[0]} rows."
            )
        self._train_perts = list(train_perts)
        self._train_profiles = np.asarray(train_profiles, dtype=float)
        self._train_embeddings = self._embed(train_perts)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, test_perts: List[str], k: int = 5) -> np.ndarray:
        """Return predicted expression profiles for test perturbations.

        Parameters
        ----------
        test_perts:
            List of test perturbation names (length N_test).
        k:
            Number of nearest training perturbations to average.

        Returns
        -------
        np.ndarray
            Predicted profiles, shape ``(N_test, n_genes)``.
        """
        if len(self._train_perts) == 0:
            raise RuntimeError("KNNBaseline has not been fitted.  Call fit() first.")

        k = min(k, len(self._train_perts))
        test_embeddings = self._embed(test_perts)

        # Cosine similarity between test and train embeddings
        test_norm = np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-12
        train_norm = np.linalg.norm(self._train_embeddings, axis=1, keepdims=True) + 1e-12
        sims = (test_embeddings / test_norm) @ (self._train_embeddings / train_norm).T
        # sims: (N_test, N_train)

        preds = np.empty((len(test_perts), self._train_profiles.shape[1]))
        for i in range(len(test_perts)):
            top_k_idx = np.argsort(sims[i])[-k:]
            preds[i] = self._train_profiles[top_k_idx].mean(axis=0)

        return preds

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, gene_names: List[str]) -> np.ndarray:
        """Embed gene names using sentence-transformers all-MiniLM-L6-v2."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for KNNBaseline.\n"
                    "Install with: pip install sentence-transformers"
                ) from exc

        embeddings = self._embedder.encode(gene_names, show_progress_bar=False)
        return np.asarray(embeddings, dtype=float)
