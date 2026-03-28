"""End-to-end smoke test using entirely synthetic data.

This test exercises the complete pipeline — data loading, preprocessing,
splitting, baseline prediction, metrics, failure-mode diagnostics, and
recalibration — WITHOUT downloading any external data or model weights.

Run with:
    python smoke_test.py

Expected output (last line):
    Smoke test passed.

Requirements:
    pip install scanpy anndata numpy scipy scikit-learn

NOTE: scGPT is intentionally NOT imported or instantiated here.
The smoke test must run in under 2 minutes on CPU with zero downloads.
"""

import sys
import time
import warnings

import anndata
import numpy as np
import pandas as pd
import scipy.sparse

# ── Silence noisy 3rd-party warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore")

START = time.time()
MAX_SMOKE_TEST_DURATION_SECONDS = 120

# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_adata(
    n_cells: int = 500,
    n_genes: int = 200,
    n_perts: int = 20,
    n_ctrl: int = 50,
    seed: int = 42,
) -> anndata.AnnData:
    """Create a synthetic AnnData that mimics the Replogle K562 dataset.

    The expression matrix is designed to pass the standard preprocessing
    filters (min_genes=200 per cell, min_cells=3 per gene) by ensuring
    every gene is expressed in every cell (count >= 1), with additional
    Poisson noise.
    """
    rng = np.random.default_rng(seed)

    pert_names = [f"GENE{i:03d}" for i in range(n_perts)]
    n_pert_cells = n_cells - n_ctrl

    obs_perts = ["control"] * n_ctrl
    obs_ctrl = [True] * n_ctrl

    # Assign ~equal number of cells per perturbation
    for idx in range(n_pert_cells):
        obs_perts.append(pert_names[idx % n_perts])
        obs_ctrl.append(False)

    # Counts: floor at 1 so every cell has every gene detected.
    # This guarantees all cells pass filter_cells(min_genes=200) when n_genes=200.
    X_raw = (rng.poisson(4, size=(n_cells, n_genes)) + 1).astype(float)

    obs = pd.DataFrame({
        "perturbation": obs_perts,
        "control": obs_ctrl,
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata = anndata.AnnData(X=X_raw, obs=obs, var=var)
    return adata


def _assert_finite(arr: np.ndarray, name: str) -> None:
    arr = np.asarray(arr, dtype=float)
    assert np.all(np.isfinite(arr)), f"{name} contains non-finite values: {arr}"


def _assert_float_in_range(val: float, lo: float, hi: float, name: str) -> None:
    assert isinstance(val, float), f"{name} should be float, got {type(val)}"
    assert lo <= val <= hi, f"{name} = {val:.4f} outside expected range [{lo}, {hi}]"


# =============================================================================
# Test 1: Preprocessor
# =============================================================================

def test_preprocessor(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocessor runs without error and returns a smaller AnnData."""
    from scfm_diagnostic.data.preprocessor import preprocess, get_mean_control_profile

    processed = preprocess(adata, n_top_genes=100, normalize=True, min_genes=10, min_cells=1)
    assert processed.n_vars <= 100, "HVG selection should reduce gene count."
    assert processed.n_obs > 0, "Should have at least some cells after filtering."

    ctrl_mean = get_mean_control_profile(processed)
    assert ctrl_mean.shape == (processed.n_vars,), "control mean shape mismatch"
    _assert_finite(ctrl_mean, "control_mean")

    print("  ✓ Preprocessor")
    return processed


# =============================================================================
# Test 2: Split by perturbation
# =============================================================================

def test_split(adata: anndata.AnnData):
    """Split is by perturbation, producing non-overlapping sets."""
    from scfm_diagnostic.data.replogle_loader import list_perturbations
    from scfm_diagnostic.data.split import train_val_test_split, get_split_data

    all_perts = list_perturbations(adata)
    assert len(all_perts) > 0, "Should have at least one non-control perturbation."
    assert sorted(all_perts) == all_perts, "list_perturbations should return sorted list."

    train, val, test = train_val_test_split(all_perts, seed=42)

    # Non-overlapping
    assert not set(train) & set(val), "train and val overlap!"
    assert not set(train) & set(test), "train and test overlap!"
    assert not set(val) & set(test), "val and test overlap!"

    # Union covers all
    assert set(train) | set(val) | set(test) == set(all_perts), "Split doesn't cover all perts."

    # get_split_data works
    test_subset = get_split_data(adata, "test", train, val, test)
    assert test_subset.n_obs > 0, "Test split should have cells."

    # Confirm no perturbation leakage: test cells' perturbations ⊆ test_perts
    test_pert_set = set(test_subset.obs["perturbation"].unique())
    assert test_pert_set <= set(test), "Test cells contain perturbations not in test split!"

    print("  ✓ Split (by perturbation, non-overlapping)")
    return train, val, test


# =============================================================================
# Test 3: Mean baseline
# =============================================================================

def test_mean_baseline(adata: anndata.AnnData, test_perts):
    """Mean baseline fits, predicts, and returns expected shapes."""
    from scfm_diagnostic.baselines.mean_baseline import MeanBaseline
    from scfm_diagnostic.data.replogle_loader import get_control_cells

    ctrl = get_control_cells(adata)
    baseline = MeanBaseline()
    baseline.fit(ctrl)

    preds = baseline.predict(test_perts)
    assert preds.shape == (len(test_perts), adata.n_vars), "predict shape wrong"
    _assert_finite(preds, "MeanBaseline.predict")

    # All rows identical (mean baseline has no per-perturbation variation)
    assert np.allclose(preds[0], preds[-1]), "All predictions should be identical."

    deltas = baseline.predict_delta(test_perts)
    assert deltas.shape == (len(test_perts), adata.n_vars)
    assert np.allclose(deltas, 0.0), "predict_delta should be all zeros."

    print("  ✓ MeanBaseline")
    return baseline


# =============================================================================
# Test 4: All metrics
# =============================================================================

def test_metrics(adata: anndata.AnnData):
    """All metrics compute correctly on random predictions."""
    from scfm_diagnostic.evaluation.metrics import mse, pearson_r, delta_pearson_r, compute_all_metrics

    rng = np.random.default_rng(7)
    n_perts, n_genes = 10, adata.n_vars

    pred = rng.random((n_perts, n_genes))
    true = rng.random((n_perts, n_genes))
    ctrl = rng.random(n_genes)

    mse_val = mse(pred, true)
    _assert_float_in_range(mse_val, 0.0, 10.0, "mse")

    pr_val = pearson_r(pred, true)
    _assert_float_in_range(pr_val, -1.0, 1.0, "pearson_r")

    dpr_val = delta_pearson_r(pred, true, ctrl)
    _assert_float_in_range(dpr_val, -1.0, 1.0, "delta_pearson_r")

    # CRITICAL CHECK: mean baseline should get delta_pearson_r ≈ 0.0
    ctrl_preds = np.tile(ctrl, (n_perts, 1))
    mean_dpr = delta_pearson_r(ctrl_preds, true, ctrl)
    assert abs(mean_dpr) < 1e-9, (
        f"Mean baseline delta_pearson_r should be 0.0, got {mean_dpr:.6f}."
        "  Check that control_mean is subtracted from BOTH pred and true."
    )

    # delta_pearson_r subtracts control_mean from both pred and true
    pred_delta = pred - ctrl
    true_delta = true - ctrl
    manual_dpr = float(np.mean([
        np.corrcoef(pred_delta[i], true_delta[i])[0, 1]
        for i in range(n_perts)
    ]))
    assert abs(dpr_val - manual_dpr) < 1e-9, (
        "delta_pearson_r does not correctly subtract control_mean."
    )

    metrics = compute_all_metrics(pred, true, ctrl, "test_model")
    assert metrics["model"] == "test_model"
    assert all(np.isfinite(metrics[k]) for k in ["mse", "pearson_r", "delta_pearson_r"])

    print("  ✓ Metrics (including delta_pearson_r correctness)")


# =============================================================================
# Test 5: Failure mode 3 — tokenisation analysis
# =============================================================================

def test_failure_mode_3(adata: anndata.AnnData):
    """Tokenisation artifact analysis runs on synthetic data."""
    from scfm_diagnostic.diagnostics.failure_mode_3 import (
        analyze_tokenization_information_loss,
        compute_delta_detectability,
        bin_expression,
    )

    # bin_expression
    expr = np.linspace(0.0, 9.9, 100)
    binned = bin_expression(expr, n_bins=51, value_range=(0.0, 10.0))
    assert binned.shape == expr.shape
    _assert_finite(binned, "bin_expression output")

    # analyze_tokenization_information_loss
    fm3 = analyze_tokenization_information_loss(adata, n_bins=51)
    assert "mean_information_loss" in fm3
    assert "genes_with_high_loss" in fm3
    assert "perturbation_genes_information_loss" in fm3
    assert 0.0 <= fm3["mean_information_loss"] <= 1.0, (
        f"mean_information_loss={fm3['mean_information_loss']} out of [0,1]"
    )

    # compute_delta_detectability
    rng = np.random.default_rng(3)
    ctrl_mean = rng.random(adata.n_vars)
    pert_means = rng.random((5, adata.n_vars))
    delta_res = compute_delta_detectability(ctrl_mean, pert_means, n_bins=51)
    assert 0.0 <= delta_res["fraction_detectable"] <= 1.0
    assert np.isfinite(delta_res["mean_delta_magnitude"])

    print("  ✓ Failure Mode 3 (tokenisation artifacts)")


# =============================================================================
# Test 6: Recalibration
# =============================================================================

def test_recalibration(adata: anndata.AnnData):
    """Recalibration fits on validation data and produces finite test outputs."""
    from scfm_diagnostic.diagnostics.recalibration import (
        fit_recalibration,
        apply_recalibration,
        evaluate_recalibration,
    )

    rng = np.random.default_rng(99)
    n_val, n_test, n_genes = 8, 5, adata.n_vars

    # Simulate scGPT over-predicting control mean (regression to mean)
    ctrl = rng.random(n_genes)
    # val preds: heavily biased toward ctrl
    val_preds = 0.1 * rng.random((n_val, n_genes)) + 0.9 * ctrl
    val_true = rng.random((n_val, n_genes))

    alpha, beta = fit_recalibration(val_preds, val_true, ctrl)
    assert alpha.shape == (n_genes,)
    assert beta.shape == (n_genes,)
    _assert_finite(alpha, "alpha")
    _assert_finite(beta, "beta")

    test_preds = 0.1 * rng.random((n_test, n_genes)) + 0.9 * ctrl
    recalib = apply_recalibration(test_preds, ctrl, alpha, beta)
    assert recalib.shape == (n_test, n_genes)
    _assert_finite(recalib, "recalibrated predictions")

    test_true = rng.random((n_test, n_genes))
    eval_res = evaluate_recalibration(test_preds, recalib, test_true, ctrl)
    for key in ["raw_mse", "recalib_mse", "mean_mse",
                "raw_pearson_r", "recalib_pearson_r",
                "raw_delta_pearson_r", "recalib_delta_pearson_r"]:
        assert key in eval_res, f"Missing key {key} in evaluate_recalibration output."
        assert np.isfinite(eval_res[key]), f"{key} = {eval_res[key]} is not finite."

    print("  ✓ Recalibration (fit on val, applied to test)")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("\n=== scFM Diagnostic Smoke Test ===\n")
    print("Building synthetic dataset …")

    # Build two datasets: one raw (for split test), one for preprocessed tests
    raw_adata = _make_synthetic_adata(n_cells=500, n_genes=200, n_perts=20, n_ctrl=50)

    print("\nRunning tests:")

    # Preprocessor (returns processed adata)
    processed = test_preprocessor(raw_adata.copy())

    # Split (use processed adata so perturbation column is present)
    train, val, test = test_split(processed)

    # Mean baseline
    test_perts_sample = test[:5] if test else ["GENE000"]
    baseline = test_mean_baseline(processed, test_perts_sample)

    # Metrics
    test_metrics(processed)

    # Failure mode 3
    test_failure_mode_3(processed)

    # Recalibration
    test_recalibration(processed)

    elapsed = time.time() - START
    assert elapsed < MAX_SMOKE_TEST_DURATION_SECONDS, (
        f"Smoke test took {elapsed:.1f}s > {MAX_SMOKE_TEST_DURATION_SECONDS}s!"
    )

    print(f"\n  ✓ Completed in {elapsed:.1f}s")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
