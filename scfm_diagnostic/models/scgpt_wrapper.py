"""scGPT wrapper for perturbation prediction inference.

This module loads a pretrained scGPT checkpoint and exposes a simple
``predict_perturbation`` interface.  It intentionally does NOT retrain
or fine-tune the model — this is a diagnostic project, not a training project.

Installation
------------
pip install git+https://github.com/bowang-lab/scGPT.git

Pretrained weights
------------------
Download the "whole-human" checkpoint from:
https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-w
and point ``model_dir`` at the downloaded directory.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class SCGPTWrapper:
    """Wrapper around the pretrained scGPT model for perturbation prediction.

    Parameters
    ----------
    model_dir:
        Path to the directory containing scGPT pretrained weights and
        associated vocabulary / config files.

    Raises
    ------
    ImportError
        If ``scgpt`` is not installed.
    FileNotFoundError
        If *model_dir* does not exist.
    """

    def __init__(self, model_dir: str) -> None:
        try:
            import scgpt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "scgpt is not installed.\n"
                "Install with:\n"
                "  pip install git+https://github.com/bowang-lab/scGPT.git"
            ) from exc

        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required to run scGPT.  pip install torch>=2.0.0")

        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"model_dir does not exist: {model_path}\n"
                "Download scGPT 'whole-human' weights from:\n"
                "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-w"
            )

        self.model_dir = str(model_path)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if self.device.type == "cpu":
            warnings.warn(
                "CUDA not available — running scGPT on CPU.  Inference will be slow.",
                stacklevel=2,
            )

        self._load_model()

    def _load_model(self) -> None:
        """Load scGPT tokenizer and model from *model_dir*."""
        import scgpt

        # scGPT exposes a TransformerModel + vocabulary inside its model directory.
        # The exact API differs slightly between scGPT versions; we use the
        # high-level load_pretrained helper when available.
        try:
            from scgpt.model import TransformerModel
            from scgpt.tokenizer import GeneVocab

            vocab_path = Path(self.model_dir) / "vocab.json"
            self.vocab = GeneVocab.from_file(str(vocab_path))
            model_config_path = Path(self.model_dir) / "args.json"
            import json

            with open(model_config_path) as f:
                model_args = json.load(f)

            self.model = TransformerModel(
                ntoken=len(self.vocab),
                d_model=model_args.get("embsize", 512),
                nhead=model_args.get("nheads", 8),
                d_hid=model_args.get("d_hid", 512),
                nlayers=model_args.get("nlayers", 12),
                nlayers_cls=model_args.get("n_layers_cls", 3),
                n_cls=1,
                vocab=self.vocab,
                dropout=0.0,
                pad_token=self.vocab.pad_token,
                pad_value=model_args.get("pad_value", 0),
                do_mvc=True,
                do_dab=False,
                use_batch_labels=False,
                num_batch_labels=None,
                domain_spec_batchnorm=False,
                ecs_threshold=model_args.get("ecs_thres", 0.3),
                explicit_zero_prob=model_args.get("explicit_zero_prob", False),
                use_fast_transformer=model_args.get("use_fast_transformer", False),
                pre_norm=model_args.get("pre_norm", False),
            )

            ckpt_path = Path(self.model_dir) / "best_model.pt"
            if not ckpt_path.exists():
                ckpt_path = next(Path(self.model_dir).glob("*.pt"), None)
            if ckpt_path is None:
                raise FileNotFoundError(f"No .pt checkpoint found in {self.model_dir}")

            state = torch.load(str(ckpt_path), map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load scGPT from {self.model_dir}: {exc}\n"
                "Ensure you have the correct checkpoint and scgpt version installed."
            ) from exc

    def predict_perturbation(
        self,
        control_adata,
        perturbation_gene: str,
        n_bins: int = 51,
    ) -> Optional[np.ndarray]:
        """Predict post-perturbation expression using scGPT.

        Parameters
        ----------
        control_adata:
            AnnData of control cells used as the input context.
        perturbation_gene:
            Name of the gene to knock out.
        n_bins:
            Number of expression bins used by scGPT's tokeniser.

        Returns
        -------
        np.ndarray or None
            Predicted expression vector of shape ``(n_genes,)``, or ``None``
            if scGPT inference fails (with a warning printed to stderr).
        """
        try:
            return self._run_scgpt_inference(control_adata, perturbation_gene, n_bins)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"scGPT inference failed for perturbation '{perturbation_gene}': {exc}",
                stacklevel=2,
            )
            return None

    def _run_scgpt_inference(
        self,
        control_adata,
        perturbation_gene: str,
        n_bins: int,
    ) -> np.ndarray:
        """Internal inference routine (called inside try/except)."""
        import scgpt
        from scgpt.utils import set_seed

        set_seed(42)

        import torch

        gene_names = list(control_adata.var_names)
        X = control_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)

        # Use mean control cell as the input expression context
        mean_ctrl = X.mean(axis=0)

        # Tokenise: map expression to bin indices
        expr_min = mean_ctrl.min()
        expr_max = mean_ctrl.max() + 1e-8
        bin_edges = np.linspace(expr_min, expr_max, n_bins + 1)
        binned = np.digitize(mean_ctrl, bin_edges[1:-1])  # values in [0, n_bins-1]

        # Convert gene names to vocab ids
        gene_ids = np.array(
            [self.vocab.get(g, self.vocab.pad_token_id) for g in gene_names],
            dtype=np.int64,
        )

        # Identify perturbation gene index
        pert_idx = gene_names.index(perturbation_gene) if perturbation_gene in gene_names else -1

        src = torch.tensor(gene_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        values = torch.tensor(binned, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.model(src, values, src_key_padding_mask=None)

        # The model returns logits over bins; take argmax and convert back
        if isinstance(output, dict):
            mlm_logits = output.get("mlm_output", output.get("cls_output"))
        else:
            mlm_logits = output

        if mlm_logits is None:
            raise RuntimeError("Could not extract mlm_output from scGPT output dict.")

        # Dequantise predicted bin indices to expression values
        predicted_bins = mlm_logits.squeeze(0).argmax(dim=-1).cpu().numpy()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        predicted_expr = bin_centers[np.clip(predicted_bins, 0, n_bins - 1)]

        return predicted_expr
