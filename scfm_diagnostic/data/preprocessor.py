"""Standard scRNA-seq preprocessing for the Replogle K562 dataset."""
from typing import Optional
import anndata
import numpy as np
import scanpy as sc

_TARGET_COUNT_SUM = 1e4

def preprocess(
    adata: anndata.AnnData,
    n_top_genes: int = 5000,
    normalize: bool = True,
    min_genes: int = 200,
    min_cells: int = 3,
) -> anndata.AnnData:
    """Apply the standard scRNA-seq preprocessing pipeline.

    Parameters
    ----------
    adata:
        Raw AnnData (cells x genes).
    n_top_genes:
        Number of highly variable genes to retain.
    normalize:
        If False, skip filtering and normalisation steps.
    min_genes:
        Minimum genes per cell (for filter_cells). Lower for smoke tests.
    min_cells:
        Minimum cells per gene (for filter_genes). Lower for smoke tests.

    Returns
    -------
    anndata.AnnData
        Preprocessed AnnData with obs columns preserved.
    """
    adata = adata.copy()

    # Back up obs before any filtering — scanpy can drop custom columns
    obs_backup = adata.obs.copy()

    if normalize:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_total(adata, target_sum=_TARGET_COUNT_SUM)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Restore obs columns that survived cell filtering
    surviving_cells = adata.obs.index
    for col in obs_backup.columns:
        if col not in adata.obs.columns:
            adata.obs[col] = obs_backup.loc[surviving_cells, col]

    return adata


def get_mean_control_profile(adata: anndata.AnnData) -> np.ndarray:
    """Compute the mean expression profile of control cells.

    Parameters
    ----------
    adata:
        AnnData with obs["control"] boolean column.

    Returns
    -------
    np.ndarray
        Mean expression vector of shape (n_genes,).
    """
    # Support both "control" boolean column and "perturbation"/"condition" string
    if "control" in adata.obs.columns:
        ctrl_mask = adata.obs["control"].astype(bool)
    elif "perturbation" in adata.obs.columns:
        ctrl_mask = adata.obs["perturbation"] == "control"
    elif "condition" in adata.obs.columns:
        ctrl_mask = adata.obs["condition"].isin(["ctrl", "control"])
    else:
        raise ValueError(
            "Cannot identify control cells. "
            "Expected obs column 'control', 'perturbation', or 'condition'."
        )

    if not ctrl_mask.any():
        raise ValueError("No control cells found in the provided AnnData.")

    ctrl_data = adata[ctrl_mask].X
    if hasattr(ctrl_data, "toarray"):
        ctrl_data = ctrl_data.toarray()
    return np.asarray(ctrl_data).mean(axis=0)
