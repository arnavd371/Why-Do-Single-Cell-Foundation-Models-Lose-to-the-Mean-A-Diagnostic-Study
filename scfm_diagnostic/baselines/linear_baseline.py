"""Ridge regression baseline for perturbation expression prediction.

Maps a one-hot encoding of the perturbation identity to the full expression
profile via a regularised linear model.  Unlike the mean baseline this
predictor can in principle learn perturbation-specific effects, but is still
a simple linear model.
"""

import numpy as np
from sklearn.linear_model import Ridge


class LinearBaseline:
    """Ridge regression from perturbation one-hot encoding to expression.

    Parameters
    ----------
    alpha:
        Ridge (L2) regularisation strength.  Defaults to 1.0.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Ridge model.

        Parameters
        ----------
        X:
            One-hot perturbation matrix of shape ``(n_cells, n_perts)``.
            Each row has exactly one non-zero entry indicating which
            perturbation was applied to that cell.
        y:
            Expression matrix of shape ``(n_cells, n_genes)``.  Should be
            log-normalised expression values.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._model.fit(X, y)
        self._fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression profiles.

        Parameters
        ----------
        X:
            One-hot (or arbitrary real-valued) perturbation matrix,
            shape ``(n_samples, n_perts)``.

        Returns
        -------
        np.ndarray
            Predicted expression matrix, shape ``(n_samples, n_genes)``.
        """
        if not self._fitted:
            raise RuntimeError("LinearBaseline has not been fitted.  Call fit() first.")
        return self._model.predict(np.asarray(X, dtype=float))
