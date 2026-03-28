"""Failure Mode 1: Fine-tuning objective mismatch.

HYPOTHESIS
----------
scGPT is pretrained with a masked-gene modelling objective (predict masked
gene expression) but fine-tuned for perturbation prediction (predict the full
post-perturbation profile).  These objectives are misaligned: the model
learns to fill in missing values from context, not to predict counterfactual
expression states.  As a consequence, scGPT predictions regress toward the
control (context) mean rather than capturing perturbation-specific effects.

This module quantifies that regression-to-mean behavior.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


def compute_finetuning_loss_landscape(
    wrapper,
    control_adata,
    test_perts: List[str],
    true_profiles: Optional[np.ndarray] = None,
    adata_full=None,
) -> Dict[str, float]:
    """Compare scGPT MSE against mean-baseline MSE for each test perturbation.

    For each test perturbation the function:
    - Runs scGPT to obtain a predicted expression profile.
    - Computes the MSE between scGPT prediction and the true mean profile.
    - Computes the MSE of the mean baseline (control mean) to the true mean.
    - Records the ratio  ``scGPT_MSE / mean_MSE``.

    A ratio > 1.0 means the mean baseline wins for that perturbation.

    Parameters
    ----------
    wrapper:
        Fitted :class:`~scfm_diagnostic.models.scgpt_wrapper.SCGPTWrapper`
        instance.
    control_adata:
        AnnData of control cells.
    test_perts:
        List of perturbation names to evaluate.
    true_profiles:
        Optional pre-computed true mean profiles, shape
        ``(len(test_perts), n_genes)``.  If None, they are computed from
        *adata_full*.
    adata_full:
        Full AnnData (needed when *true_profiles* is None).

    Returns
    -------
    Dict[str, float]
        Keys: ``mean_ratio``, ``std_ratio``, ``fraction_where_mean_wins``.
    """
    ctrl_X = control_adata.X
    if hasattr(ctrl_X, "toarray"):
        ctrl_X = ctrl_X.toarray()
    control_mean = np.asarray(ctrl_X).mean(axis=0)

    ratios = []
    for i, pert in enumerate(test_perts):
        # True mean profile
        if true_profiles is not None:
            true_mean = true_profiles[i]
        elif adata_full is not None:
            mask = adata_full.obs["perturbation"] == pert
            cells = adata_full[mask].X
            if hasattr(cells, "toarray"):
                cells = cells.toarray()
            true_mean = np.asarray(cells).mean(axis=0)
        else:
            warnings.warn(
                "Neither true_profiles nor adata_full provided; skipping.",
                stacklevel=2,
            )
            continue

        scgpt_pred = wrapper.predict_perturbation(control_adata, pert)
        if scgpt_pred is None:
            continue

        scgpt_mse = float(np.mean((scgpt_pred - true_mean) ** 2))
        mean_mse = float(np.mean((control_mean - true_mean) ** 2))

        if mean_mse < 1e-12:
            continue  # degenerate case

        ratios.append(scgpt_mse / mean_mse)

    if len(ratios) == 0:
        return {"mean_ratio": float("nan"), "std_ratio": float("nan"), "fraction_where_mean_wins": float("nan")}

    ratios = np.array(ratios)
    return {
        "mean_ratio": float(ratios.mean()),
        "std_ratio": float(ratios.std()),
        "fraction_where_mean_wins": float((ratios > 1.0).mean()),
    }


def analyze_prediction_regression_to_mean(
    scgpt_predictions: np.ndarray,
    true_profiles: np.ndarray,
    control_mean: np.ndarray,
) -> Dict[str, float]:
    """Quantify how much scGPT predictions regress toward the control mean.

    Three metrics are computed:

    1. **cosine_sim_to_control** — mean cosine similarity between each
       scGPT prediction and the control mean vector.  High values indicate
       the model output resembles the control rather than the perturbation.
    2. **cosine_sim_to_truth** — mean cosine similarity between each
       prediction and the corresponding true perturbed profile.
    3. **regression_to_mean_score** — ratio
       ``‖prediction − control_mean‖ / ‖true_profile − control_mean‖``.
       Values < 1.0 mean the model's prediction is closer to the control
       mean than the ground truth is; i.e. the model underestimates the
       perturbation effect.

    Parameters
    ----------
    scgpt_predictions:
        Predicted profiles, shape ``(n_perts, n_genes)``.
    true_profiles:
        Ground-truth mean profiles, shape ``(n_perts, n_genes)``.
    control_mean:
        Control mean expression vector, shape ``(n_genes,)``.

    Returns
    -------
    Dict[str, float]
        Keys: ``cosine_sim_to_control``, ``cosine_sim_to_truth``,
        ``regression_to_mean_score``.
    """
    n = scgpt_predictions.shape[0]
    cos_sims_ctrl = []
    cos_sims_truth = []
    rtm_scores = []

    for i in range(n):
        pred = scgpt_predictions[i]
        true = true_profiles[i]
        ctrl = control_mean

        # Cosine similarity = 1 - cosine distance
        sim_ctrl = 1.0 - float(cosine_distance(pred, ctrl))
        sim_truth = 1.0 - float(cosine_distance(pred, true))
        cos_sims_ctrl.append(sim_ctrl)
        cos_sims_truth.append(sim_truth)

        denom = np.linalg.norm(true - ctrl)
        if denom > 1e-12:
            rtm_scores.append(np.linalg.norm(pred - ctrl) / denom)

    result = {
        "cosine_sim_to_control": float(np.mean(cos_sims_ctrl)),
        "cosine_sim_to_truth": float(np.mean(cos_sims_truth)),
    }
    if rtm_scores:
        result["regression_to_mean_score"] = float(np.mean(rtm_scores))
    else:
        result["regression_to_mean_score"] = float("nan")

    return result
