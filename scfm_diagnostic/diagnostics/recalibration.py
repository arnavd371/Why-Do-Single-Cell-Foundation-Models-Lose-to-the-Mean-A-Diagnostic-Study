"""Post-hoc recalibration of scGPT predictions.

MOTIVATION
----------
scGPT predictions are biased toward the control mean (regression-to-mean
behavior documented in Failure Mode 1).  A simple per-gene linear
recalibration can correct this bias without any additional model training.

The recalibration model is:

    true ≈ alpha * scgpt_pred + (1 - alpha) * control_mean + beta

where *alpha* and *beta* are per-gene scalars fitted on the **validation**
set.  Evaluating the recalibrated predictions on the **test** set ensures
there is no data leakage.

IMPORTANT: recalibration is ALWAYS fit on the validation set and evaluated
on the test set.  Fitting on the test set would constitute data leakage.
"""

from typing import Dict, Tuple

import numpy as np


def fit_recalibration(
    scgpt_preds: np.ndarray,
    true_profiles: np.ndarray,
    control_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the per-gene linear recalibration parameters on the validation set.

    Solves:
        true[i, g] = alpha[g] * scgpt_preds[i, g]
                     + (1 - alpha[g]) * control_mean[g]
                     + beta[g]
                     + residual

    by rearranging into:
        (true[i, g] - control_mean[g]) = alpha[g] * (scgpt_preds[i, g] - control_mean[g])
                                         + beta[g]

    and fitting a per-gene ordinary least squares regression.

    Parameters
    ----------
    scgpt_preds:
        scGPT predictions on the validation set, shape ``(n_val, n_genes)``.
    true_profiles:
        Ground-truth mean profiles for the validation perturbations,
        shape ``(n_val, n_genes)``.
    control_mean:
        Control mean expression vector, shape ``(n_genes,)``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(alpha, beta)`` — each of shape ``(n_genes,)``.
    """
    preds = np.asarray(scgpt_preds, dtype=float)
    true = np.asarray(true_profiles, dtype=float)
    ctrl = np.asarray(control_mean, dtype=float)

    n_val, n_genes = preds.shape

    # Centred versions
    preds_c = preds - ctrl[np.newaxis, :]  # (n_val, n_genes)
    true_c = true - ctrl[np.newaxis, :]

    # Per-gene OLS: y = alpha * x + beta
    # alpha = cov(x, y) / var(x)
    # beta  = mean(y) - alpha * mean(x)
    x_mean = preds_c.mean(axis=0)  # (n_genes,)
    y_mean = true_c.mean(axis=0)

    cov_xy = ((preds_c - x_mean) * (true_c - y_mean)).mean(axis=0)
    var_x = ((preds_c - x_mean) ** 2).mean(axis=0)

    alpha = np.where(var_x > 1e-12, cov_xy / var_x, 1.0)
    beta = y_mean - alpha * x_mean

    return alpha, beta


def apply_recalibration(
    scgpt_preds: np.ndarray,
    control_mean: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Apply fitted recalibration parameters to (test) scGPT predictions.

    Parameters
    ----------
    scgpt_preds:
        Raw scGPT predictions, shape ``(n_test, n_genes)``.
    control_mean:
        Control mean expression vector, shape ``(n_genes,)``.
    alpha:
        Per-gene scale factor, shape ``(n_genes,)``.
    beta:
        Per-gene bias, shape ``(n_genes,)``.

    Returns
    -------
    np.ndarray
        Recalibrated predictions, shape ``(n_test, n_genes)``.
    """
    preds = np.asarray(scgpt_preds, dtype=float)
    ctrl = np.asarray(control_mean, dtype=float)

    preds_c = preds - ctrl[np.newaxis, :]
    recalibrated_c = alpha[np.newaxis, :] * preds_c + beta[np.newaxis, :]
    return recalibrated_c + ctrl[np.newaxis, :]


def evaluate_recalibration(
    raw_preds: np.ndarray,
    recalibrated_preds: np.ndarray,
    true_profiles: np.ndarray,
    control_mean: np.ndarray,
) -> Dict[str, float]:
    """Compare raw scGPT, recalibrated scGPT, and the mean baseline.

    Metrics computed for each model:
    - **MSE**: mean squared error.
    - **pearson_r**: mean Pearson r across genes.
    - **delta_pearson_r**: Pearson r on ``(pred - control)`` vs
      ``(true - control)``.  This is the primary metric because it measures
      whether the model captures the *perturbation effect* rather than the
      baseline expression level.

    Parameters
    ----------
    raw_preds:
        Raw scGPT predictions, shape ``(n_test, n_genes)``.
    recalibrated_preds:
        Recalibrated scGPT predictions, shape ``(n_test, n_genes)``.
    true_profiles:
        Ground-truth profiles, shape ``(n_test, n_genes)``.
    control_mean:
        Control mean vector, shape ``(n_genes,)``.

    Returns
    -------
    Dict[str, float]
        Keys: ``raw_mse``, ``raw_pearson_r``, ``raw_delta_pearson_r``,
        ``recalib_mse``, ``recalib_pearson_r``, ``recalib_delta_pearson_r``,
        ``mean_mse``, ``mean_pearson_r``, ``mean_delta_pearson_r``.
    """
    from scfm_diagnostic.evaluation.metrics import mse, pearson_r, delta_pearson_r

    ctrl = np.asarray(control_mean, dtype=float)
    n = true_profiles.shape[0]
    mean_preds = np.tile(ctrl, (n, 1))

    result: Dict[str, float] = {}
    for prefix, preds in [
        ("raw", raw_preds),
        ("recalib", recalibrated_preds),
        ("mean", mean_preds),
    ]:
        result[f"{prefix}_mse"] = mse(preds, true_profiles)
        result[f"{prefix}_pearson_r"] = pearson_r(preds, true_profiles)
        result[f"{prefix}_delta_pearson_r"] = delta_pearson_r(preds, true_profiles, ctrl)

    return result
