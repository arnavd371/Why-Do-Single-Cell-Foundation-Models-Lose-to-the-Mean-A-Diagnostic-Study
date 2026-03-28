"""Full evaluation runner for perturbation prediction models.

Evaluates all models on the test perturbations and produces a summary
DataFrame printed to stdout.
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from scfm_diagnostic.evaluation.metrics import compute_all_metrics


def run_full_evaluation(
    adata,
    test_perts: List[str],
    control_mean: np.ndarray,
    scgpt_wrapper,
    mean_baseline,
    recalibration_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    """Evaluate all models on the test perturbations.

    For each test perturbation the function:
    1. Computes the true mean expression profile.
    2. Obtains predictions from scGPT (may return None on failure).
    3. Obtains predictions from the mean baseline.
    4. Optionally applies recalibration to scGPT predictions.
    5. Computes MSE, Pearson r, and delta Pearson r for each model.

    Parameters
    ----------
    adata:
        Full preprocessed AnnData (used to compute true profiles).
    test_perts:
        List of test perturbation names.
    control_mean:
        Control mean expression vector, shape ``(n_genes,)``.
    scgpt_wrapper:
        Fitted :class:`~scfm_diagnostic.models.scgpt_wrapper.SCGPTWrapper`
        (or ``None`` to skip scGPT evaluation).
    mean_baseline:
        Fitted :class:`~scfm_diagnostic.baselines.mean_baseline.MeanBaseline`.
    recalibration_params:
        ``(alpha, beta)`` arrays from :func:`~scfm_diagnostic.diagnostics.recalibration.fit_recalibration`,
        or ``None`` to skip recalibrated evaluation.

    Returns
    -------
    pd.DataFrame
        Columns: ``perturbation``, ``model``, ``mse``, ``pearson_r``,
        ``delta_pearson_r``.
    """
    # Filter to perturbations that actually exist in adata
    available_perts = set(adata.obs["condition"].unique())
    valid_test_perts = [p for p in test_perts if p in available_perts]
    if len(valid_test_perts) < len(test_perts):
        missing = set(test_perts) - set(valid_test_perts)
        warnings.warn(f"Skipping {len(missing)} test perturbations not found in adata.", stacklevel=2)

    # Build control AnnData for scGPT context
    control_adata = adata[adata.obs["control"].astype(bool)].copy()

    rows = []
    for pert in valid_test_perts:
        # True profile
        pert_cells = adata[adata.obs["condition"] == pert].X
        if hasattr(pert_cells, "toarray"):
            pert_cells = pert_cells.toarray()
        true_profile = np.asarray(pert_cells).mean(axis=0)  # (n_genes,)

        # Mean baseline prediction
        mean_pred = mean_baseline.predict([pert])[0]  # (n_genes,)
        for name, pred in [("Mean baseline", mean_pred)]:
            metrics = compute_all_metrics(
                pred[np.newaxis, :],
                true_profile[np.newaxis, :],
                control_mean,
                name,
            )
            rows.append({
                "perturbation": pert,
                "model": name,
                "mse": metrics["mse"],
                "pearson_r": metrics["pearson_r"],
                "delta_pearson_r": metrics["delta_pearson_r"],
            })

        # scGPT prediction
        if scgpt_wrapper is not None:
            scgpt_pred = scgpt_wrapper.predict_perturbation(control_adata, pert)
            if scgpt_pred is not None:
                metrics = compute_all_metrics(
                    scgpt_pred[np.newaxis, :],
                    true_profile[np.newaxis, :],
                    control_mean,
                    "scGPT (raw)",
                )
                rows.append({
                    "perturbation": pert,
                    "model": "scGPT (raw)",
                    "mse": metrics["mse"],
                    "pearson_r": metrics["pearson_r"],
                    "delta_pearson_r": metrics["delta_pearson_r"],
                })

                # Recalibrated scGPT
                if recalibration_params is not None:
                    from scfm_diagnostic.diagnostics.recalibration import apply_recalibration

                    alpha, beta = recalibration_params
                    recalib_pred = apply_recalibration(
                        scgpt_pred[np.newaxis, :], control_mean, alpha, beta
                    )[0]
                    metrics = compute_all_metrics(
                        recalib_pred[np.newaxis, :],
                        true_profile[np.newaxis, :],
                        control_mean,
                        "scGPT (recalibrated)",
                    )
                    rows.append({
                        "perturbation": pert,
                        "model": "scGPT (recalibrated)",
                        "mse": metrics["mse"],
                        "pearson_r": metrics["pearson_r"],
                        "delta_pearson_r": metrics["delta_pearson_r"],
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No evaluation results to display.")
        return df

    # Print summary table
    summary = df.groupby("model")[["mse", "pearson_r", "delta_pearson_r"]].mean()
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'MSE':>10} {'Pearson r':>12} {'Delta Pearson r':>16}")
    print("=" * 70)
    for model_name, row in summary.iterrows():
        print(
            f"{model_name:<25} {row['mse']:>10.4f} {row['pearson_r']:>12.4f} {row['delta_pearson_r']:>16.4f}"
        )
    print("=" * 70 + "\n")

    return df
