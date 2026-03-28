"""Failure Mode 2: Distribution shift in gene expression space.

HYPOTHESIS
----------
scGPT was pretrained on a diverse collection of human tissues.  Replogle K562
cells are a cancer cell line with extreme expression patterns that are
underrepresented in the pretraining corpus.  Consequently the model's internal
representation space is not well calibrated for this cell type, leading to
poor generalisation.

This module quantifies that distribution shift.
"""

import warnings
from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def compute_embedding_distribution_shift(
    embedder,
    control_adata,
    model_dir: str,
) -> Dict[str, float]:
    """Characterise how K562 cells sit in the scGPT embedding space.

    Steps
    -----
    1. Embed control cells with scGPT.
    2. Apply PCA (50 components) to the embeddings.
    3. Compute mean pairwise Euclidean distance within the PCA space.
    4. Compute the distance from the embedding centroid to the origin.

    Parameters
    ----------
    embedder:
        :class:`~scfm_diagnostic.models.embedder.FoundationModelEmbedder` instance.
    control_adata:
        AnnData of control cells.
    model_dir:
        Path to scGPT weights directory.

    Returns
    -------
    Dict[str, float]
        Keys: ``mean_pairwise_dist``, ``centroid_distance_to_origin``,
        ``pca_explained_variance_ratio``.
    """
    embeddings = embedder.embed_cells(control_adata, model_dir)

    pca = PCA(n_components=min(50, embeddings.shape[1], embeddings.shape[0] - 1))
    pca_emb = pca.fit_transform(embeddings)

    dists = pairwise_distances(pca_emb, metric="euclidean")
    n = dists.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    mean_pairwise = float(dists[triu_idx].mean()) if len(triu_idx[0]) > 0 else 0.0

    centroid = pca_emb.mean(axis=0)
    centroid_dist = float(np.linalg.norm(centroid))

    return {
        "mean_pairwise_dist": mean_pairwise,
        "centroid_distance_to_origin": centroid_dist,
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }


def compute_mmd(
    embeddings_A: np.ndarray,
    embeddings_B: np.ndarray,
    kernel: str = "rbf",
) -> float:
    """Compute Maximum Mean Discrepancy (MMD) between two embedding sets.

    Uses an RBF (Gaussian) kernel whose bandwidth is set to the median pairwise
    distance across all samples (the "median heuristic").

    Parameters
    ----------
    embeddings_A:
        First set of embeddings, shape ``(n_A, d)``.
    embeddings_B:
        Second set of embeddings, shape ``(n_B, d)``.
    kernel:
        Kernel type.  Currently only ``"rbf"`` is supported.

    Returns
    -------
    float
        MMD² estimate between the two distributions.
    """
    if kernel != "rbf":
        raise ValueError(f"Unsupported kernel: {kernel!r}.  Only 'rbf' is implemented.")

    A = np.asarray(embeddings_A, dtype=float)
    B = np.asarray(embeddings_B, dtype=float)

    # Median bandwidth heuristic
    all_emb = np.vstack([A, B])
    dists = pairwise_distances(all_emb, metric="euclidean")
    n_total = all_emb.shape[0]
    triu = dists[np.triu_indices(n_total, k=1)]
    bandwidth = float(np.median(triu)) if len(triu) > 0 else 1.0
    if bandwidth < 1e-10:
        bandwidth = 1.0
    gamma = 1.0 / (2.0 * bandwidth ** 2)

    def rbf(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        D = pairwise_distances(X, Y, metric="sqeuclidean")
        return np.exp(-gamma * D)

    K_AA = rbf(A, A)
    K_BB = rbf(B, B)
    K_AB = rbf(A, B)

    n_A, n_B = A.shape[0], B.shape[0]
    # Unbiased MMD² estimator
    mmd2 = (
        (K_AA.sum() - np.trace(K_AA)) / (n_A * (n_A - 1) + 1e-12)
        + (K_BB.sum() - np.trace(K_BB)) / (n_B * (n_B - 1) + 1e-12)
        - 2.0 * K_AB.mean()
    )
    return float(mmd2)


def plot_embedding_shift(
    control_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """UMAP (or PCA fallback) plot of control vs perturbed cell embeddings.

    Parameters
    ----------
    control_embeddings:
        Embeddings of control cells, shape ``(n_ctrl, d)``.
    perturbed_embeddings:
        Embeddings of perturbed cells, shape ``(n_pert, d)``.
    save_path:
        If provided, save the figure to this path (PNG/PDF).
    """
    import matplotlib.pyplot as plt

    ctrl = np.asarray(control_embeddings, dtype=float)
    pert = np.asarray(perturbed_embeddings, dtype=float)
    all_emb = np.vstack([ctrl, pert])
    labels = np.array(["control"] * len(ctrl) + ["perturbed"] * len(pert))

    # Attempt UMAP, fall back to PCA
    try:
        from umap import UMAP

        reducer = UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(all_emb)
        method = "UMAP"
    except ImportError:
        warnings.warn(
            "umap-learn not installed; falling back to PCA for embedding plot.",
            stacklevel=2,
        )
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_emb)
        method = "PCA"

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"control": "#0077BB", "perturbed": "#EE7733"}
    for lbl, color in colors.items():
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=lbl,
            alpha=0.5,
            s=10,
            linewidths=0,
        )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title("scGPT Embedding Space: Control vs Perturbed Cells")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
