"""Evaluation metrics for perturbation expression prediction.

PRIMARY METRIC: delta_pearson_r
-------------------------------
``delta_pearson_r`` measures whether a model captures the *perturbation
effect* — the deviation from the control mean — rather than just the
baseline expression level.

    delta_pearson_r(pred, true, ctrl) =
        Pearson r( (pred - ctrl), (true - ctrl) )

A model that always predicts the control mean gets ``delta_pearson_r = 0.0``
by definition (its delta is identically zero, yielding zero correlation with
any non-zero true delta).  A perfect model gets ``1.0``.

This metric exposes the failure of scGPT: despite achieving acceptable MSE
by staying close to the control mean, its delta_pearson_r is near zero,
meaning it captures almost no perturbation-specific signal.
"""

from typing import Dict

import numpy as np
from scipy.stats import pearsonr


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error between predicted and true expression profiles.

    Parameters
    ----------
    pred:
        Predicted expression, any shape (flattened internally).
    true:
        Ground-truth expression, same shape as *pred*.

    Returns
    -------
    float
        Mean squared error.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean((pred - true) ** 2))


def pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Pearson correlation coefficient across genes.

    Computes Pearson r between predicted and true expression for each gene
    (column), then averages.

    Parameters
    ----------
    pred:
        Predicted expression, shape ``(n_samples, n_genes)`` or ``(n_genes,)``.
    true:
        Ground-truth expression, same shape.

    Returns
    -------
    float
        Mean Pearson r across genes.  Returns 0.0 for degenerate inputs.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
        true = true.reshape(1, -1)

    if pred.shape[0] == 1:
        # Only one sample — Pearson r across genes
        p, t = pred[0], true[0]
        if p.std() < 1e-12 or t.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(p, t)[0, 1])

    # Multiple samples — average Pearson r across genes (column-wise)
    rs = []
    for g in range(pred.shape[1]):
        p, t = pred[:, g], true[:, g]
        if p.std() < 1e-12 or t.std() < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(pearsonr(p, t)[0]))
    return float(np.mean(rs))


def delta_pearson_r(
    pred: np.ndarray,
    true: np.ndarray,
    control_mean: np.ndarray,
) -> float:
    """Pearson correlation on perturbation deltas (primary metric).

    Computes Pearson r between ``(pred − control_mean)`` and
    ``(true − control_mean)``, measuring whether the model correctly
    captures the *direction and magnitude of the perturbation effect*.

    A model that always predicts the control mean has a zero delta vector,
    and therefore ``delta_pearson_r = 0.0``.  This correctly penalises
    regression-to-mean behavior that MSE alone does not detect.

    Parameters
    ----------
    pred:
        Predicted expression profiles, shape ``(n_perts, n_genes)``.
    true:
        Ground-truth expression profiles, shape ``(n_perts, n_genes)``.
    control_mean:
        Mean control expression vector, shape ``(n_genes,)``.
        Subtracted from BOTH *pred* and *true* before correlation.

    Returns
    -------
    float
        Pearson r on the delta (perturbation-effect) vectors, averaged
        across perturbations.  Returns 0.0 for degenerate inputs.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ctrl = np.asarray(control_mean, dtype=float)

    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
        true = true.reshape(1, -1)

    # Subtract control mean from BOTH prediction and truth
    pred_delta = pred - ctrl[np.newaxis, :]
    true_delta = true - ctrl[np.newaxis, :]

    rs = []
    for i in range(pred_delta.shape[0]):
        pd = pred_delta[i]
        td = true_delta[i]
        if pd.std() < 1e-12 or td.std() < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(pearsonr(pd, td)[0]))

    return float(np.mean(rs)) if rs else 0.0


def compute_all_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    control_mean: np.ndarray,
    model_name: str,
) -> Dict[str, object]:
    """Compute MSE, Pearson r, and delta Pearson r for a model.

    Parameters
    ----------
    pred:
        Predicted expression profiles.
    true:
        Ground-truth expression profiles.
    control_mean:
        Control mean expression vector.
    model_name:
        Human-readable name included in the returned dict.

    Returns
    -------
    Dict[str, object]
        ``{"model": model_name, "mse": float, "pearson_r": float,
           "delta_pearson_r": float}``
    """
    return {
        "model": model_name,
        "mse": mse(pred, true),
        "pearson_r": pearson_r(pred, true),
        "delta_pearson_r": delta_pearson_r(pred, true, control_mean),
    }
