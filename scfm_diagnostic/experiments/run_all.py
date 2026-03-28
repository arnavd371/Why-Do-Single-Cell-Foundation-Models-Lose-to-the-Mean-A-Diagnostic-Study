"""Master experiment runner for the scGPT diagnostic study.

Usage
-----
python -m scfm_diagnostic.experiments.run_all \\
    --model-dir path/to/scgpt_weights \\
    --output-dir results/full_run \\
    --seed 42
"""

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the full scGPT diagnostic study pipeline."
    )
    parser.add_argument(
        "--data-dir",
        default="data/cache",
        help="Directory for caching the Replogle dataset (default: data/cache).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to scGPT pretrained weights directory.  "
             "If not provided, scGPT evaluation is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Output directory for results (default: results/).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=5000,
        help="Number of highly variable genes (default: 5000).",
    )
    parser.add_argument(
        "--n-test-perts",
        type=int,
        default=50,
        help="Maximum number of test perturbations to evaluate (default: 50).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the complete diagnostic pipeline."""
    args = _parse_args(argv)
    _set_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and preprocess data
    # ------------------------------------------------------------------
    print("[1/9] Loading Replogle K562 dataset …")
    from scfm_diagnostic.data.replogle_loader import load_replogle_k562
    from scfm_diagnostic.data.preprocessor import preprocess, get_mean_control_profile

    adata = load_replogle_k562(cache_dir=args.data_dir)
    print(f"      Loaded {adata.n_obs} cells × {adata.n_vars} genes.")

    print("[2/9] Preprocessing …")
    adata = preprocess(adata, n_top_genes=args.n_top_genes)
    print(f"      After HVG selection: {adata.n_vars} genes.")

    # ------------------------------------------------------------------
    # 2. Split by perturbation
    # ------------------------------------------------------------------
    print("[3/9] Splitting by perturbation …")
    from scfm_diagnostic.data.replogle_loader import list_perturbations
    from scfm_diagnostic.data.split import train_val_test_split

    all_perts = list_perturbations(adata)
    train_perts, val_perts, test_perts = train_val_test_split(
        all_perts, seed=args.seed
    )
    test_perts = test_perts[: args.n_test_perts]
    print(
        f"      {len(train_perts)} train / {len(val_perts)} val / "
        f"{len(test_perts)} test perturbations."
    )

    # ------------------------------------------------------------------
    # 3. Fit mean baseline
    # ------------------------------------------------------------------
    print("[4/9] Fitting mean baseline …")
    from scfm_diagnostic.data.replogle_loader import get_control_cells
    from scfm_diagnostic.baselines.mean_baseline import MeanBaseline

    control_adata = get_control_cells(adata)
    control_mean = get_mean_control_profile(adata)

    mean_baseline = MeanBaseline()
    mean_baseline.fit(control_adata)

    # ------------------------------------------------------------------
    # 4. Run scGPT on validation perturbations
    # ------------------------------------------------------------------
    scgpt_wrapper = None
    val_scgpt_preds = None
    val_true_profiles = None

    if args.model_dir is not None:
        print("[5/9] Loading scGPT and evaluating on validation set …")
        try:
            from scfm_diagnostic.models.scgpt_wrapper import SCGPTWrapper

            scgpt_wrapper = SCGPTWrapper(args.model_dir)

            val_scgpt_preds_list = []
            val_true_list = []
            for pert in val_perts:
                pred = scgpt_wrapper.predict_perturbation(control_adata, pert)
                if pred is None:
                    continue
                mask = adata.obs["perturbation"] == pert
                cells = adata[mask].X
                if hasattr(cells, "toarray"):
                    cells = cells.toarray()
                true_mean = np.asarray(cells).mean(axis=0)
                val_scgpt_preds_list.append(pred)
                val_true_list.append(true_mean)

            if val_scgpt_preds_list:
                val_scgpt_preds = np.vstack(val_scgpt_preds_list)
                val_true_profiles = np.vstack(val_true_list)
            else:
                warnings.warn("No valid scGPT predictions on validation set.", stacklevel=2)

        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"scGPT loading failed: {exc}", stacklevel=2)
    else:
        print("[5/9] --model-dir not provided, skipping scGPT evaluation.")

    # ------------------------------------------------------------------
    # 5. Fit recalibration on validation predictions
    # ------------------------------------------------------------------
    recalibration_params = None
    if val_scgpt_preds is not None and val_true_profiles is not None:
        print("[6/9] Fitting recalibration on validation predictions …")
        from scfm_diagnostic.diagnostics.recalibration import fit_recalibration

        alpha, beta = fit_recalibration(val_scgpt_preds, val_true_profiles, control_mean)
        recalibration_params = (alpha, beta)
    else:
        print("[6/9] Skipping recalibration (no scGPT validation predictions).")

    # ------------------------------------------------------------------
    # 6. Evaluate all models on test perturbations
    # ------------------------------------------------------------------
    print("[7/9] Evaluating on test perturbations …")
    from scfm_diagnostic.evaluation.evaluate import run_full_evaluation

    results_df = run_full_evaluation(
        adata=adata,
        test_perts=test_perts,
        control_mean=control_mean,
        scgpt_wrapper=scgpt_wrapper,
        mean_baseline=mean_baseline,
        recalibration_params=recalibration_params,
    )

    # ------------------------------------------------------------------
    # 7. Run failure mode diagnostics
    # ------------------------------------------------------------------
    print("[8/9] Running failure mode diagnostics …")
    from scfm_diagnostic.diagnostics.failure_mode_3 import (
        analyze_tokenization_information_loss,
        compute_delta_detectability,
    )

    fm3_results = analyze_tokenization_information_loss(adata)
    print(f"      Failure Mode 3 — mean info loss: {fm3_results['mean_information_loss']:.3f}")

    pert_means = np.vstack(
        [
            np.asarray(
                adata[adata.obs["perturbation"] == p].X.toarray()
                if hasattr(adata[adata.obs["perturbation"] == p].X, "toarray")
                else adata[adata.obs["perturbation"] == p].X
            ).mean(axis=0)
            for p in (all_perts[: min(50, len(all_perts))])
        ]
    )
    fm3_delta = compute_delta_detectability(control_mean, pert_means)
    print(f"      Failure Mode 3 — fraction detectable: {fm3_delta['fraction_detectable']:.3f}")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    print("[9/9] Saving results …")
    results_path = output_dir / "results.json"
    results = {
        "n_train_perts": len(train_perts),
        "n_val_perts": len(val_perts),
        "n_test_perts": len(test_perts),
        "failure_mode_3": {k: v for k, v in fm3_results.items() if not isinstance(v, dict)},
        "delta_detectability": fm3_delta,
    }
    if not results_df.empty:
        summary = results_df.groupby("model")[["mse", "pearson_r", "delta_pearson_r"]].mean()
        results["model_comparison"] = summary.to_dict()

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"      Results saved to {results_path}")

    if not results_df.empty:
        csv_path = output_dir / "per_perturbation_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"      Per-perturbation results saved to {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
