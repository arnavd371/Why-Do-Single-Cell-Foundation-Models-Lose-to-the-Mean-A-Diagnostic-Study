"""Standard scRNA-seq preprocessing for the Replogle K562 dataset.

Wraps the canonical Scanpy preprocessing pipeline and provides a helper
to compute the mean control expression profile — the "mean baseline" that
outperforms scGPT on the Replogle perturbation prediction benchmark.
"""

from typing import Optional

import anndata
import numpy as np
import scanpy as sc


_TARGET_COUNT_SUM = 1e4  # Cells normalised to this total count


def preprocess(
    adata: anndata.AnnData,
    n_top_genes: int = 5000,
    normalize: bool = True,
) -> anndata.AnnData:
    """Apply the standard scRNA-seq preprocessing pipeline.

    Steps (when *normalize* is True):
    1. Filter cells with fewer than 200 detected genes.
    2. Filter genes detected in fewer than 3 cells.
    3. Normalise each cell's total counts to 10 000.
    4. Log1p transform.
    5. Select the top *n_top_genes* highly variable genes.
    6. Subset the AnnData to those genes.

    Parameters
    ----------
    adata:
        Raw AnnData (cells × genes).
    n_top_genes:
        Number of highly variable genes to retain.
    normalize:
        If False, skip steps 1-4 (useful for already-processed data).

    Returns
    -------
    anndata.AnnData
        Preprocessed AnnData (cells × HVGs).
    """
    adata = adata.copy()

    if normalize:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=_TARGET_COUNT_SUM)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    return adata


def get_mean_control_profile(adata: anndata.AnnData) -> np.ndarray:
    """Compute the mean expression profile of control cells.

    This is the "mean baseline" — the trivially simple predictor that
    always returns the average control expression regardless of the
    perturbation.  Counterintuitively, this outperforms scGPT on the
    Replogle K562 perturbation prediction benchmark.

    Parameters
    ----------
    adata:
        AnnData with ``obs["control"]`` boolean column.  Only control
        cells (``obs["control"] == True``) are included in the mean.

    Returns
    -------
    np.ndarray
        Mean expression vector of shape ``(n_genes,)``.
    """
    ctrl_mask = adata.obs["control"].astype(bool)
    if not ctrl_mask.any():
        raise ValueError("No control cells found in the provided AnnData.")

    ctrl_data = adata[ctrl_mask].X
    if hasattr(ctrl_data, "toarray"):
        ctrl_data = ctrl_data.toarray()
    return np.asarray(ctrl_data).mean(axis=0)
