"""Failure Mode 3: Continuous value tokenisation artifacts.

HYPOTHESIS
----------
scGPT tokenises continuous gene expression values into ``n_bins`` discrete
bins (default 51).  This quantisation destroys information for genes whose
perturbation effects are subtle: small expression changes that fall within
the same bin are invisible to the model.

This module quantifies that information loss.
"""

from typing import Any, Dict, Tuple

import numpy as np


def bin_expression(
    expression: np.ndarray,
    n_bins: int = 51,
    value_range: Tuple[float, float] = (0.0, 10.0),
) -> np.ndarray:
    """Quantise and dequantise continuous expression values.

    The expression array is first discretised into *n_bins* equal-width bins
    spanning *value_range*, then dequantised back to the bin-centre values.
    This mirrors the tokenisation step in scGPT's forward pass.

    Parameters
    ----------
    expression:
        Continuous expression values, any shape.
    n_bins:
        Number of discrete bins.
    value_range:
        ``(min, max)`` of the expression range over which bins are defined.

    Returns
    -------
    np.ndarray
        Dequantised array, same shape as *expression*.
    """
    lo, hi = value_range
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Clip to range before digitising
    clipped = np.clip(expression, lo, hi - 1e-9)
    bin_indices = np.digitize(clipped, bin_edges[1:-1])  # in [0, n_bins-1]
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return bin_centers[bin_indices]


def analyze_tokenization_information_loss(
    adata,
    n_bins: int = 51,
) -> Dict[str, Any]:
    """Measure how much variance is lost due to expression binning.

    For each gene the function computes:

    - ``true_var``: variance of mean expression across perturbations.
    - ``binned_var``: variance after quantising to *n_bins* and dequantising.
    - ``information_loss_ratio``: ``1 - binned_var / (true_var + eps)``
      — fraction of variance destroyed by binning (0 = no loss, 1 = all lost).

    Parameters
    ----------
    adata:
        AnnData object.  ``adata.X`` should be log-normalised expression.
        ``adata.obs["condition"]`` identifies the perturbation of each cell.
    n_bins:
        Number of discrete expression bins (matching scGPT default of 51).

    Returns
    -------
    Dict[str, Any]
        Keys:
        - ``mean_information_loss`` (float)
        - ``genes_with_high_loss`` (int): genes with > 50 % information loss
        - ``perturbation_genes_information_loss`` (Dict[str, float]):
          information loss specifically for genes that were knocked out
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    perts = adata.obs["condition"].values
    unique_perts = np.unique(perts)

    # Compute per-perturbation mean profiles
    pert_means = np.vstack(
        [X[perts == p].mean(axis=0) for p in unique_perts]
    )  # (n_perts, n_genes)

    # Expression range for binning
    val_min = float(X.min())
    val_max = float(X.max())
    value_range = (val_min, val_max + 1e-8)

    # True variance across perturbations per gene
    true_var = pert_means.var(axis=0)  # (n_genes,)

    # Variance after binning
    binned_means = bin_expression(pert_means, n_bins=n_bins, value_range=value_range)
    binned_var = binned_means.var(axis=0)

    eps = 1e-12
    loss_ratio = 1.0 - binned_var / (true_var + eps)
    loss_ratio = np.clip(loss_ratio, 0.0, 1.0)

    genes_with_high_loss = int((loss_ratio > 0.5).sum())
    mean_loss = float(loss_ratio.mean())

    # Information loss for knocked-out genes specifically
    gene_names = list(adata.var_names)
    gene_name_set = set(gene_names)
    pert_gene_loss: Dict[str, float] = {}
    for pert in unique_perts:
        if pert in gene_name_set:
            idx = gene_names.index(pert)
            pert_gene_loss[pert] = float(loss_ratio[idx])

    return {
        "mean_information_loss": mean_loss,
        "genes_with_high_loss": genes_with_high_loss,
        "perturbation_genes_information_loss": pert_gene_loss,
    }


def compute_delta_detectability(
    control_mean: np.ndarray,
    perturbed_means: np.ndarray,
    n_bins: int = 51,
    expression_range: Tuple[float, float] = (0.0, 10.0),
) -> Dict[str, Any]:
    """Determine what fraction of perturbation deltas survive binning.

    A delta (perturbed − control) is considered "detectable" if the
    perturbed value falls into a different bin than the control value for a
    given gene.

    Parameters
    ----------
    control_mean:
        Control expression vector, shape ``(n_genes,)``.
    perturbed_means:
        Mean expression of each perturbation, shape ``(n_perts, n_genes)``.
    n_bins:
        Number of discrete bins (scGPT default: 51).
    expression_range:
        ``(min, max)`` of the expression range for binning.

    Returns
    -------
    Dict[str, Any]
        Keys:
        - ``fraction_detectable`` (float)
        - ``mean_delta_magnitude`` (float)
        - ``correlation_delta_detectability`` (float): Pearson r between
          ``|delta|`` magnitude and whether the delta is detectable
    """
    lo, hi = expression_range
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    def _digitize(arr: np.ndarray) -> np.ndarray:
        clipped = np.clip(arr, lo, hi - 1e-9)
        return np.digitize(clipped, bin_edges[1:-1])

    ctrl_bins = _digitize(control_mean)  # (n_genes,)
    pert_bins = _digitize(perturbed_means)  # (n_perts, n_genes)

    # detectable[i, j] = True if gene j of pert i crosses a bin boundary
    detectable = pert_bins != ctrl_bins[np.newaxis, :]  # (n_perts, n_genes)

    delta = perturbed_means - control_mean[np.newaxis, :]
    delta_magnitude = np.abs(delta)

    fraction_detectable = float(detectable.mean())
    mean_delta_mag = float(delta_magnitude.mean())

    # Correlation between |delta| and detectability (flattened)
    flat_mag = delta_magnitude.ravel()
    flat_det = detectable.ravel().astype(float)
    if flat_mag.std() > 1e-12 and flat_det.std() > 1e-12:
        corr = float(np.corrcoef(flat_mag, flat_det)[0, 1])
    else:
        corr = float("nan")

    return {
        "fraction_detectable": fraction_detectable,
        "mean_delta_magnitude": mean_delta_mag,
        "correlation_delta_detectability": corr,
    }
