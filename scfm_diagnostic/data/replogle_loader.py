"""Loader for the Norman et al. 2019 Perturb-seq dataset.

CRITICAL DESIGN NOTE
--------------------
The Norman dataset contains CRISPRa perturbations (activation, not knockout)
of individual and paired gene combinations in K562 cells.
obs["condition"] contains the perturbation name — single genes are e.g. "CDKN1A"
and combinatorial perturbations are e.g. "CDKN1A+CDKN1B".
Control cells have condition == "ctrl".
"""

import os
from typing import Any, Dict, List, Tuple

import anndata
import numpy as np


NORMAN_FOLDER_ID = "1M0QLP6dKKw3Fsw2rofyxzoBBMrst1fNE"
NORMAN_FILENAME = "norman.h5ad"


def _download_norman(cache_dir: str) -> str:
    """Download Norman dataset from Google Drive if not already cached.

    Parameters
    ----------
    cache_dir:
        Directory to cache the downloaded file.

    Returns
    -------
    str
        Path to the downloaded .h5ad file.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download the Norman dataset.\n"
            "Install with: pip install gdown"
        )

    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, NORMAN_FILENAME)

    if os.path.exists(fpath):
        print(f"Using cached Norman dataset at {fpath}")
        return fpath

    print(f"Downloading Norman dataset to {fpath} ...")
    # List files in the folder and download the .h5ad file
    url = f"https://drive.google.com/drive/folders/{NORMAN_FOLDER_ID}"
    gdown.download_folder(url, output=cache_dir, quiet=False)

    # Find the downloaded .h5ad file in case filename differs
    for fname in os.listdir(cache_dir):
        if fname.endswith(".h5ad"):
            actual_path = os.path.join(cache_dir, fname)
            if actual_path != fpath:
                os.rename(actual_path, fpath)
            break
    else:
        raise FileNotFoundError(
            f"Download completed but no .h5ad file found in {cache_dir}.\n"
            "Please download manually from:\n"
            f"https://drive.google.com/drive/folders/{NORMAN_FOLDER_ID}\n"
            f"and place it at {fpath}"
        )

    print("Download complete.")
    return fpath


def load_norman(cache_dir: str = "data/cache") -> anndata.AnnData:
    """Load the Norman et al. 2019 Perturb-seq dataset.

    Downloads automatically on first call, then uses local cache.

    Parameters
    ----------
    cache_dir:
        Directory to cache the downloaded file.

    Returns
    -------
    anndata.AnnData
        AnnData object with:
        - adata.X: expression matrix (cells x genes)
        - adata.obs["condition"]: perturbation label per cell
        - adata.obs["control"]: True if control cell
    """
    fpath = _download_norman(cache_dir)
    adata = anndata.read_h5ad(fpath)

    # Normalise obs column names — scGPT datasets use varying conventions
    # Try common column names for perturbation and control labels
    if "condition" not in adata.obs.columns:
        for candidate in ["perturbation", "gene_program", "perturbation_name"]:
            if candidate in adata.obs.columns:
                adata.obs["condition"] = adata.obs[candidate]
                break
        else:
            raise KeyError(
                "Could not find perturbation column in adata.obs.\n"
                f"Available columns: {list(adata.obs.columns)}"
            )

    # Add boolean control column
    adata.obs["control"] = adata.obs["condition"].isin(["ctrl", "control", "non-targeting"])

    print(f"Loaded Norman dataset: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Control cells: {adata.obs['control'].sum()}")
    print(f"Unique perturbations: {adata.obs['condition'].nunique()}")

    return adata


def get_control_cells(adata: anndata.AnnData) -> anndata.AnnData:
    """Return subset of control (non-perturbed) cells.

    Parameters
    ----------
    adata:
        Full AnnData object from load_norman().

    Returns
    -------
    anndata.AnnData
        Subset where obs["control"] == True.
    """
    return adata[adata.obs["control"]].copy()


def get_perturbed_cells(
    adata: anndata.AnnData,
    perturbation: str,
) -> anndata.AnnData:
    """Return cells with a specific perturbation.

    Parameters
    ----------
    adata:
        Full AnnData object.
    perturbation:
        Perturbation name, e.g. "CDKN1A" or "CDKN1A+CDKN1B".

    Returns
    -------
    anndata.AnnData
        Subset of cells with the specified perturbation.
    """
    mask = adata.obs["condition"] == perturbation
    if mask.sum() == 0:
        raise ValueError(
            f"Perturbation '{perturbation}' not found.\n"
            f"Available: {sorted(adata.obs['condition'].unique())[:10]} ..."
        )
    return adata[mask].copy()


def list_perturbations(
    adata: anndata.AnnData,
    exclude_control: bool = True,
    single_only: bool = False,
) -> List[str]:
    """Return sorted list of unique perturbation names.

    Parameters
    ----------
    adata:
        Full AnnData object.
    exclude_control:
        If True, exclude control condition. Default True.
    single_only:
        If True, return only single-gene perturbations
        (exclude combinatorial e.g. "CDKN1A+CDKN1B").

    Returns
    -------
    List[str]
        Sorted perturbation names.
    """
    perts = adata.obs["condition"].unique().tolist()
    if exclude_control:
        perts = [p for p in perts if not adata.obs[
            adata.obs["condition"] == p]["control"].all()]
    if single_only:
        perts = [p for p in perts if "+" not in p]
    return sorted(perts)


def get_mean_control_profile(adata: anndata.AnnData) -> np.ndarray:
    """Compute mean expression profile of control cells.

    This is the "mean baseline" that outperforms scGPT.

    Parameters
    ----------
    adata:
        Full AnnData object.

    Returns
    -------
    np.ndarray
        Mean expression vector of shape (n_genes,).
    """
    control = get_control_cells(adata)
    X = control.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X.mean(axis=0)).flatten()


def get_problem_count(adata: anndata.AnnData) -> int:
    """Return total number of unique perturbations excluding control.

    Parameters
    ----------
    adata:
        Full AnnData object.

    Returns
    -------
    int
        Number of unique perturbations.
    """
    return len(list_perturbations(adata))


def generate_synthetic_norman(
    n_cells: int = 500,
    n_genes: int = 200,
    n_perturbations: int = 20,
    seed: int = 42,
) -> anndata.AnnData:
    """Generate synthetic Norman-like data for smoke testing.

    No downloads required. Produces a plausible AnnData object
    with the same structure as the real Norman dataset.

    Parameters
    ----------
    n_cells:
        Total number of cells.
    n_genes:
        Number of genes.
    n_perturbations:
        Number of unique perturbations.
    seed:
        Random seed.

    Returns
    -------
    anndata.AnnData
        Synthetic AnnData with obs["condition"] and obs["control"].
    """
    rng = np.random.default_rng(seed)

    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    pert_names = [f"GENE{i:04d}" for i in range(n_perturbations)]

    conditions = rng.choice(pert_names + ["ctrl"], size=n_cells)
    X = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)

    # Perturbed cells have slightly different expression
    for i, cond in enumerate(conditions):
        if cond != "ctrl" and cond in gene_names:
            gene_idx = gene_names.index(cond)
            X[i, gene_idx] *= rng.uniform(0.1, 3.0)

    import pandas as pd
    obs = pd.DataFrame({"condition": conditions}, index=[f"cell{i}" for i in range(n_cells)])
    obs["control"] = obs["condition"] == "ctrl"
    var = pd.DataFrame(index=gene_names)

    return anndata.AnnData(X=X, obs=obs, var=var)
