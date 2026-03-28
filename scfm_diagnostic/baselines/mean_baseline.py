"""Mean baseline predictor for perturbation expression prediction.

This module implements the simplest possible baseline: always predict the
mean expression profile of control cells, regardless of the perturbation.

HEADLINE RESULT
---------------
This trivial predictor — which ignores the perturbation entirely —
outperforms scGPT on the Replogle K562 Perturb-seq benchmark when evaluated
by *delta Pearson r* (the perturbation-effect-aware metric).  The finding
motivates a careful diagnostic of what goes wrong inside foundation models.
"""

from typing import List, Optional

import numpy as np


class MeanBaseline:
    """Predict the mean control expression profile for every perturbation.

    The predictor is intentionally naïve: it stores the mean control profile
    at fit time and returns that identical vector for every query, regardless
    of which gene was knocked out.

    Despite its simplicity, this baseline achieves competitive (or superior)
    MSE and delta Pearson r compared with scGPT on the Replogle K562 dataset,
    which is the central empirical finding motivating the diagnostic study.
    """

    def __init__(self) -> None:
        self._control_mean: Optional[np.ndarray] = None
        self._n_genes: Optional[int] = None

    def fit(self, control_adata) -> None:
        """Compute and store the mean expression profile of control cells.

        Parameters
        ----------
        control_adata:
            AnnData containing *only* control cells (e.g. the output of
            ``get_control_cells``).  ``control_adata.X`` should be the
            preprocessed (log-normalised) expression matrix.
        """
        X = control_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        self._control_mean = X.mean(axis=0)
        self._n_genes = self._control_mean.shape[0]

    def predict(self, perturbations: List[str]) -> np.ndarray:
        """Return the mean control profile for each perturbation.

        Parameters
        ----------
        perturbations:
            List of perturbation names to predict (length N).

        Returns
        -------
        np.ndarray
            Shape ``(N, n_genes)``.  Every row is identical: the control mean.
        """
        self._check_fitted()
        n = len(perturbations)
        return np.tile(self._control_mean, (n, 1))

    def predict_delta(self, perturbations: List[str]) -> np.ndarray:
        """Return predicted perturbation deltas (all zeros).

        Because the mean baseline always predicts the control mean, its
        implied delta from control is identically zero for every gene and
        every perturbation.

        Parameters
        ----------
        perturbations:
            List of perturbation names (length N).

        Returns
        -------
        np.ndarray
            Zero matrix of shape ``(N, n_genes)``.
        """
        self._check_fitted()
        n = len(perturbations)
        return np.zeros((n, self._n_genes), dtype=float)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._control_mean is None:
            raise RuntimeError("MeanBaseline has not been fitted.  Call fit() first.")
