"""Load the Replogle et al. 2022 K562 Perturb-seq dataset.

This module handles downloading and caching the Replogle K562 essential
screen dataset from Figshare, and provides helper utilities for accessing
control and perturbed cells.
"""

import os
from pathlib import Path
from typing import List

import anndata
import numpy as np


_FIGSHARE_URL = "https://figshare.com/ndownloader/files/35773219"
_CACHE_FILENAME = "replogle_k562_essential.h5ad"


def load_replogle_k562(cache_dir: str = "data/cache") -> anndata.AnnData:
    """Load the Replogle et al. 2022 K562 essential Perturb-seq dataset.

    Downloads the dataset from Figshare on first call and caches it locally
    so subsequent calls are fast.

    Parameters
    ----------
    cache_dir:
        Directory used to store the cached .h5ad file.

    Returns
    -------
    anndata.AnnData
        AnnData object with:
        - ``adata.X``: raw counts matrix (cells × genes)
        - ``adata.obs["perturbation"]``: perturbation gene name per cell
        - ``adata.obs["control"]``: boolean, True if non-targeting control

    Raises
    ------
    RuntimeError
        If the download fails.  The error message includes manual download
        instructions.
    """
    cache_path = Path(cache_dir) / _CACHE_FILENAME
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        _download_dataset(cache_path)

    adata = anndata.read_h5ad(cache_path)
    adata = _normalise_obs_columns(adata)
    return adata


def _download_dataset(dest: Path) -> None:
    """Download the dataset, preferring *gdown* but falling back to *requests*."""
    try:
        import gdown  # noqa: F401 — only used here
        gdown.download(_FIGSHARE_URL, str(dest), quiet=False, fuzzy=True)
        if not dest.exists():
            raise RuntimeError("gdown reported success but file is missing.")
        return
    except (OSError, RuntimeError) as exc:
        first_error = exc

    # Fallback: plain HTTP download
    try:
        import urllib.request

        urllib.request.urlretrieve(_FIGSHARE_URL, str(dest))
        return
    except OSError as exc2:
        raise RuntimeError(
            f"Could not download the Replogle K562 dataset.\n"
            f"  gdown error : {first_error}\n"
            f"  urllib error: {exc2}\n\n"
            "Manual download instructions:\n"
            f"  1. Open {_FIGSHARE_URL} in a browser.\n"
            f"  2. Save the file as: {dest}\n"
            "  3. Re-run this script."
        ) from exc2


def _normalise_obs_columns(adata: anndata.AnnData) -> anndata.AnnData:
    """Ensure ``perturbation`` and ``control`` obs columns exist.

    Different releases of the Replogle dataset use slightly different column
    names.  This function maps the most common alternatives to the canonical
    names expected by the rest of the codebase.
    """
    obs = adata.obs

    # ------------------------------------------------------------------
    # Perturbation column
    # ------------------------------------------------------------------
    pert_candidates = ["perturbation", "gene", "gene_name", "target_gene"]
    pert_col = next((c for c in pert_candidates if c in obs.columns), None)
    if pert_col is None:
        raise KeyError(
            f"Cannot find a perturbation column.  Available columns: {list(obs.columns)}"
        )
    if pert_col != "perturbation":
        obs["perturbation"] = obs[pert_col]

    # ------------------------------------------------------------------
    # Control column
    # ------------------------------------------------------------------
    ctrl_candidates = ["control", "is_control", "ctrl"]
    ctrl_col = next((c for c in ctrl_candidates if c in obs.columns), None)
    if ctrl_col is not None:
        obs["control"] = obs[ctrl_col].astype(bool)
    else:
        # Infer: cells whose perturbation name looks like a non-targeting
        # control are labelled True.
        ctrl_keywords = ["non-targeting", "nontargeting", "control", "ctrl"]
        obs["control"] = obs["perturbation"].str.lower().apply(
            lambda x: any(k in x for k in ctrl_keywords)
        )

    adata.obs = obs
    return adata


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_control_cells(adata: anndata.AnnData) -> anndata.AnnData:
    """Return the subset of *adata* that are non-targeting control cells.

    Parameters
    ----------
    adata:
        Full AnnData object with ``obs["control"]`` boolean column.

    Returns
    -------
    anndata.AnnData
        Subset where ``obs["control"] == True``.
    """
    return adata[adata.obs["control"] == True].copy()  # noqa: E712


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
        Name of the perturbation gene (must match ``obs["perturbation"]``).

    Returns
    -------
    anndata.AnnData
        Subset of cells with the requested perturbation.
    """
    mask = adata.obs["perturbation"] == perturbation
    return adata[mask].copy()


def list_perturbations(adata: anndata.AnnData) -> List[str]:
    """Return a sorted list of all unique perturbation names, excluding controls.

    Parameters
    ----------
    adata:
        Full AnnData object with ``obs["perturbation"]`` and ``obs["control"]``.

    Returns
    -------
    List[str]
        Sorted list of perturbation gene names.
    """
    non_ctrl = adata[~adata.obs["control"].astype(bool)]
    return sorted(non_ctrl.obs["perturbation"].unique().tolist())
