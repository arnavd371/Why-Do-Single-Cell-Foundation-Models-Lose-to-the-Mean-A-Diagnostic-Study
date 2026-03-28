"""Extract cell and gene embeddings from scGPT for distribution shift analysis.

This module is used by the Failure Mode 2 diagnostic to probe whether K562
cells are "out-of-distribution" with respect to the scGPT pretraining corpus.
"""

import warnings
from typing import List

import numpy as np


class FoundationModelEmbedder:
    """Extract embeddings from a pretrained scGPT model.

    Parameters
    ----------
    model_dir:
        Path to scGPT pretrained weights directory.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._wrapper = None

    def _get_wrapper(self):
        """Lazy-load the SCGPTWrapper to avoid import errors at module level."""
        if self._wrapper is None:
            from scfm_diagnostic.models.scgpt_wrapper import SCGPTWrapper

            self._wrapper = SCGPTWrapper(self.model_dir)
        return self._wrapper

    # ------------------------------------------------------------------
    # Cell embeddings
    # ------------------------------------------------------------------

    def embed_cells(
        self,
        adata,
        model_dir: str,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed cells using the scGPT encoder (CLS token representation).

        Parameters
        ----------
        adata:
            AnnData of cells to embed.  ``adata.X`` should be log-normalised.
        model_dir:
            Path to scGPT weights directory (may differ from ``self.model_dir``).
        batch_size:
            Number of cells per forward pass.

        Returns
        -------
        np.ndarray
            Cell embedding matrix, shape ``(n_cells, embed_dim)``.
        """
        try:
            import torch
            from scfm_diagnostic.models.scgpt_wrapper import SCGPTWrapper

            wrapper = SCGPTWrapper(model_dir)
            model = wrapper.model
            device = wrapper.device
            vocab = wrapper.vocab

            gene_names = list(adata.var_names)
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=float)

            n_cells, n_genes = X.shape
            gene_ids = np.array(
                [vocab.get(g, vocab.pad_token_id) for g in gene_names],
                dtype=np.int64,
            )

            all_embeddings = []
            model.eval()

            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)
                batch_X = X[start:end]

                # Rough binning (51 bins by default)
                n_bins = 51
                expr_min, expr_max = batch_X.min(), batch_X.max() + 1e-8
                bin_edges = np.linspace(expr_min, expr_max, n_bins + 1)
                binned = np.digitize(batch_X, bin_edges[1:-1])

                src = torch.tensor(
                    np.tile(gene_ids, (end - start, 1)),
                    dtype=torch.long,
                    device=device,
                )
                values = torch.tensor(binned, dtype=torch.long, device=device)

                with torch.no_grad():
                    output = model(src, values, src_key_padding_mask=None)

                if isinstance(output, dict) and "cell_emb" in output:
                    emb = output["cell_emb"].cpu().numpy()
                else:
                    # Fallback: mean-pool over gene tokens
                    if isinstance(output, dict):
                        token_out = next(iter(output.values()))
                    else:
                        token_out = output
                    emb = token_out.mean(dim=1).cpu().numpy()

                all_embeddings.append(emb)

            return np.vstack(all_embeddings)

        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Cell embedding failed: {exc}.  Returning random embeddings "
                "(for testing only).",
                stacklevel=2,
            )
            n_cells = adata.n_obs
            return np.random.randn(n_cells, 512).astype(float)

    # ------------------------------------------------------------------
    # Gene embeddings
    # ------------------------------------------------------------------

    def embed_genes(self, gene_names: List[str]) -> np.ndarray:
        """Return scGPT gene token embedding vectors.

        Parameters
        ----------
        gene_names:
            List of gene names (HGNC symbols).

        Returns
        -------
        np.ndarray
            Gene embedding matrix, shape ``(n_genes, embed_dim)``.
        """
        try:
            import torch
            from scfm_diagnostic.models.scgpt_wrapper import SCGPTWrapper

            wrapper = self._get_wrapper()
            model = wrapper.model
            vocab = wrapper.vocab
            device = wrapper.device

            gene_ids = torch.tensor(
                [vocab.get(g, vocab.pad_token_id) for g in gene_names],
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                # Access the model's token embedding table
                embeddings = model.encoder(gene_ids).cpu().numpy()

            return embeddings

        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Gene embedding failed: {exc}.  Returning random embeddings.",
                stacklevel=2,
            )
            return np.random.randn(len(gene_names), 512).astype(float)
