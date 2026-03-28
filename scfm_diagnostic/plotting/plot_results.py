"""Paper figures for the scGPT diagnostic study.

Usage
-----
python -m scfm_diagnostic.plotting.plot_results \\
    --results-dir results/ \\
    --output-dir paper/figures/

Colour palette (colorblind-safe)
---------------------------------
- Blue   #0077BB
- Orange #EE7733
- Teal   #009988
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0077BB"
ORANGE = "#EE7733"
TEAL = "#009988"


# ---------------------------------------------------------------------------
# Figure 1: Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: dict,
    save_dir: Optional[Path] = None,
) -> None:
    """Bar chart of MSE and delta_pearson_r for all models.

    Highlights that the mean baseline beats scGPT on delta_pearson_r.

    Parameters
    ----------
    results:
        Parsed ``results.json`` dictionary.
    save_dir:
        Output directory.  If ``None``, the figure is shown interactively.
    """
    comparison = results.get("model_comparison", {})
    if not comparison:
        warnings.warn("No model_comparison data in results.json.", stacklevel=2)
        return

    models = list(comparison.get("mse", {}).keys())
    mse_vals = [comparison["mse"].get(m, 0) for m in models]
    dpr_vals = [comparison["delta_pearson_r"].get(m, 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSE
    ax = axes[0]
    bars = ax.bar(x, mse_vals, color=BLUE, width=width)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("Mean Squared Error (lower is better)")

    # delta_pearson_r
    ax = axes[1]
    colors = [ORANGE if "Mean" in m else TEAL for m in models]
    ax.bar(x, dpr_vals, color=colors, width=width)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Delta Pearson r")
    ax.set_title("Delta Pearson r — Perturbation Effect\n(higher is better; Mean baseline ≈ 0 by definition)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    _save_or_show(fig, save_dir, "fig1_model_comparison")


# ---------------------------------------------------------------------------
# Figure 2: scGPT predictions vs true profiles
# ---------------------------------------------------------------------------

def plot_prediction_scatter(
    scgpt_preds: np.ndarray,
    true_profiles: np.ndarray,
    control_mean: np.ndarray,
    save_dir: Optional[Path] = None,
) -> None:
    """Scatter of scGPT predictions vs true profiles, coloured by distance from control.

    Shows regression-to-mean behavior: predictions cluster near the control
    mean regardless of the true perturbed expression.

    Parameters
    ----------
    scgpt_preds:
        scGPT predictions, shape ``(n_perts, n_genes)``.
    true_profiles:
        True perturbed profiles, shape ``(n_perts, n_genes)``.
    control_mean:
        Control mean vector, shape ``(n_genes,)``.
    save_dir:
        Output directory.
    """
    # Flatten: one point per (perturbation, gene)
    pred_flat = scgpt_preds.ravel()
    true_flat = true_profiles.ravel()
    dist_from_ctrl = np.abs(true_profiles - control_mean[np.newaxis, :]).ravel()

    # Subsample if too large
    n = len(pred_flat)
    if n > 20000:
        idx = np.random.choice(n, 20000, replace=False)
        pred_flat = pred_flat[idx]
        true_flat = true_flat[idx]
        dist_from_ctrl = dist_from_ctrl[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        true_flat,
        pred_flat,
        c=dist_from_ctrl,
        cmap="viridis",
        s=5,
        alpha=0.5,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="|true − control|")
    lims = [
        min(true_flat.min(), pred_flat.min()),
        max(true_flat.max(), pred_flat.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("True expression")
    ax.set_ylabel("scGPT predicted expression")
    ax.set_title("scGPT predictions vs true profiles\n(colour = distance from control mean)")
    ax.legend(frameon=False)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "fig2_prediction_scatter")


# ---------------------------------------------------------------------------
# Figure 3: Tokenisation information loss
# ---------------------------------------------------------------------------

def plot_tokenization_loss(
    loss_per_gene: np.ndarray,
    pert_gene_loss: dict,
    save_dir: Optional[Path] = None,
) -> None:
    """Histogram of per-gene information loss due to expression binning.

    Parameters
    ----------
    loss_per_gene:
        Array of information loss ratios, shape ``(n_genes,)``.
    pert_gene_loss:
        Dict mapping perturbation gene names to their information loss.
    save_dir:
        Output directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(loss_per_gene, bins=50, color=BLUE, edgecolor="white", alpha=0.8, label="All genes")
    if pert_gene_loss:
        pert_losses = list(pert_gene_loss.values())
        ax.scatter(
            pert_losses,
            np.ones(len(pert_losses)) * 2,
            color=ORANGE,
            zorder=5,
            s=40,
            label="Perturbation-target genes",
        )

    ax.set_xlabel("Information loss ratio (1 − binned_var / true_var)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"Tokenisation Information Loss (n_bins=51)\n"
                 f"Genes with >50% loss: {(np.asarray(loss_per_gene) > 0.5).sum()}")
    ax.legend(frameon=False)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "fig3_tokenization_loss")


# ---------------------------------------------------------------------------
# Figure 4: UMAP / PCA embedding plot
# ---------------------------------------------------------------------------

def plot_embedding_umap(
    control_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
    save_dir: Optional[Path] = None,
) -> None:
    """UMAP (or PCA fallback) of control vs perturbed cells in scGPT space.

    Parameters
    ----------
    control_embeddings:
        Control cell embeddings, shape ``(n_ctrl, d)``.
    perturbed_embeddings:
        Perturbed cell embeddings, shape ``(n_pert, d)``.
    save_dir:
        Output directory.
    """
    from scfm_diagnostic.diagnostics.failure_mode_2 import plot_embedding_shift

    save_path = str(save_dir / "fig4_embedding_umap.pdf") if save_dir else None
    plot_embedding_shift(control_embeddings, perturbed_embeddings, save_path=save_path)

    if save_dir:
        # Also save PNG
        plot_embedding_shift(
            control_embeddings,
            perturbed_embeddings,
            save_path=str(save_dir / "fig4_embedding_umap.png"),
        )


# ---------------------------------------------------------------------------
# Figure 5: Recalibration improvement
# ---------------------------------------------------------------------------

def plot_recalibration_improvement(
    results: dict,
    save_dir: Optional[Path] = None,
) -> None:
    """Bar chart showing improvement from recalibration.

    Parameters
    ----------
    results:
        Parsed ``results.json`` dictionary.
    save_dir:
        Output directory.
    """
    comparison = results.get("model_comparison", {})
    raw_dpr = comparison.get("delta_pearson_r", {}).get("scGPT (raw)", None)
    recalib_dpr = comparison.get("delta_pearson_r", {}).get("scGPT (recalibrated)", None)
    mean_dpr = comparison.get("delta_pearson_r", {}).get("Mean baseline", None)

    if raw_dpr is None:
        warnings.warn("No scGPT (raw) results found; cannot plot recalibration figure.", stacklevel=2)
        return

    models = ["Mean baseline", "scGPT (raw)", "scGPT (recalibrated)"]
    vals = [mean_dpr or 0, raw_dpr, recalib_dpr or 0]
    colors = [TEAL, BLUE, ORANGE]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(models, vals, color=colors)
    ax.set_ylabel("Delta Pearson r")
    ax.set_title("Recalibration improvement over raw scGPT\n(primary metric: delta Pearson r)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    _save_or_show(fig, save_dir, "fig5_recalibration")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_dir: Optional[Path], stem: str) -> None:
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            path = save_dir / f"{stem}.{ext}"
            fig.savefig(str(path), dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png → {save_dir}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--results-dir", default="results/", help="Directory with results.json.")
    parser.add_argument("--output-dir", default="paper/figures/", help="Directory to save figures.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    results_path = results_dir / "results.json"
    if not results_path.exists():
        print(f"results.json not found at {results_path}.  Run experiments/run_all.py first.")
    else:
        with open(results_path) as f:
            results = json.load(f)

        plot_model_comparison(results, save_dir=output_dir)
        plot_recalibration_improvement(results, save_dir=output_dir)
        print("Figures generated successfully.")
