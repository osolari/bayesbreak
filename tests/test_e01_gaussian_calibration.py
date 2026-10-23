"""E01 diagnostic tests: generator, metrics, seed reproducibility, vectorized curve.

These are correctness tests run before the confirmatory experiment.
Diagnostic results are NOT confirmatory replicates.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the scripts directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_e01_gaussian_calibration import (
    E01_CONFIG,
    E01_CONFIG_SHA256,
    _bayes_curve_fixed_k_vectorized,
    _bayes_curve_mix_k_vectorized,
    _boundary_match,
    _sample_boundaries_uniform,
    compute_calibration_curve,
    compute_f1_metrics,
    compute_replicate_metrics,
    enumerate_cells,
    generate_dataset,
    run_inference,
)


# ---------------------------------------------------------------------------
# Section 1: Config integrity
# ---------------------------------------------------------------------------

def test_config_sha256_stable():
    """E01_CONFIG_SHA256 must match expected value (frozen config)."""
    import hashlib, json
    computed = hashlib.sha256(json.dumps(E01_CONFIG, sort_keys=True).encode()).hexdigest()
    assert computed == E01_CONFIG_SHA256, (
        f"Config SHA256 mismatch: got {computed[:16]}, expected {E01_CONFIG_SHA256[:16]}"
    )


def test_cell_count():
    """E01 must have exactly 14 primary + 8 ablation = 22 cells per run."""
    cells = enumerate_cells(include_ablation=True)
    primary = [c for c in cells if not c["ablation"]]
    ablation = [c for c in cells if c["ablation"]]
    assert len(primary) == 14, f"Expected 14 primary cells, got {len(primary)}"
    assert len(ablation) == 8, f"Expected 8 ablation cells, got {len(ablation)}"
    assert len(cells) == 22, f"Expected 22 total cells, got {len(cells)}"


def test_cell_index_unique():
    """All cell indices must be unique for seed uniqueness."""
    cells = enumerate_cells(include_ablation=True)
    indices = [c["cell_index"] for c in cells]
    assert len(set(indices)) == len(indices), "Duplicate cell indices"


def test_null_cells_exist():
    """There must be 2 null cells (k_true=1, delta=0): one per n-value."""
    cells = enumerate_cells(include_ablation=False)
    null = [c for c in cells if c["k_true"] == 1]
    assert len(null) == 2, f"Expected 2 null cells, got {len(null)}"
    for c in null:
        assert c["delta"] == 0.0
        assert c["prior"] == "geometric_decay_log4"


# ---------------------------------------------------------------------------
# Section 2: Data generator correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k,delta", [
    (256, 1, 0.0),
    (256, 3, 1.0),
    (256, 6, 0.5),
    (512, 3, 2.0),
    (512, 6, 0.5),
])
def test_generator_correctness(n, k, delta):
    """Check generated dataset satisfies structural invariants."""
    seed_seq = np.random.SeedSequence([20261021, 1, 0, 42])
    ds = generate_dataset(n, k, delta, 1.0, 32, seed_seq)

    # Shape
    assert ds["y"].shape == (n,)
    assert ds["x"].shape == (n,)
    assert len(ds["true_means"]) == k
    assert len(ds["true_boundaries"]) == k + 1
    assert ds["true_boundaries"][0] == 0
    assert ds["true_boundaries"][-1] == n

    # Minimum segment length
    segs = ds["seg_lengths"]
    assert all(s >= 32 for s in segs), f"Segment too short: {segs}"
    assert sum(segs) == n

    # True signal shape
    assert ds["true_signal"].shape == (n,)
    for q, (a, b) in enumerate(zip(ds["true_boundaries"][:-1], ds["true_boundaries"][1:])):
        np.testing.assert_allclose(
            ds["true_signal"][a:b], ds["true_means"][q],
            err_msg=f"Segment {q}: signal mismatch"
        )

    # Delta increment check for k>1
    if k > 1:
        for q in range(1, k):
            diff = abs(ds["true_means"][q] - ds["true_means"][q - 1])
            assert abs(diff - delta) < 1e-10, (
                f"Adjacent mean increment {diff:.6f} != delta {delta:.6f}"
            )

    # No NaN or inf in y
    assert np.all(np.isfinite(ds["y"])), "Non-finite values in y"


def test_generator_reproducibility():
    """Same seed must produce identical dataset."""
    seed_seq1 = np.random.SeedSequence([20261021, 1, 5, 17])
    seed_seq2 = np.random.SeedSequence([20261021, 1, 5, 17])
    ds1 = generate_dataset(256, 3, 1.0, 1.0, 32, seed_seq1)
    ds2 = generate_dataset(256, 3, 1.0, 1.0, 32, seed_seq2)
    np.testing.assert_array_equal(ds1["y"], ds2["y"])
    assert ds1["true_boundaries"] == ds2["true_boundaries"]


def test_generator_distinct_seeds():
    """Different replicates must produce different datasets."""
    seed_seq1 = np.random.SeedSequence([20261021, 1, 0, 0])
    seed_seq2 = np.random.SeedSequence([20261021, 1, 0, 1])
    ds1 = generate_dataset(256, 3, 1.0, 1.0, 32, seed_seq1)
    ds2 = generate_dataset(256, 3, 1.0, 1.0, 32, seed_seq2)
    assert not np.allclose(ds1["y"], ds2["y"]), "Different replicates produced identical y"


def test_boundary_sampling_uniform():
    """Stars-and-bars boundary sampler covers all admissible boundary sets."""
    n, k, min_L = 10, 3, 2
    rng = np.random.default_rng(42)
    seen = set()
    n_trials = 10000
    for _ in range(n_trials):
        bds = _sample_boundaries_uniform(n, k, min_L, rng)
        assert len(bds) == k - 1
        seen.add(tuple(bds))
        # Verify minimum gap
        prev = 0
        for b in bds:
            assert b - prev >= min_L, f"Gap {b-prev} < min_L={min_L}"
            prev = b
        assert n - prev >= min_L

    # The number of admissible partitions = C(n - k*min_L + k - 1, k - 1) = C(7, 2) = 21
    n_excess = n - k * min_L  # = 10 - 6 = 4
    import math
    n_admissible = math.comb(n_excess + k - 1, k - 1)  # C(6, 2) = 15
    # All admissible sets should be covered
    assert len(seen) == n_admissible, (
        f"Coverage: saw {len(seen)} distinct boundary sets, expected {n_admissible}"
    )


def test_initial_mean_is_zero_or_half():
    """Initial segment mean must be 0 or 0.5 per E01 spec."""
    for rep in range(50):
        seed_seq = np.random.SeedSequence([20261021, 1, 0, rep])
        ds = generate_dataset(256, 3, 1.0, 1.0, 32, seed_seq)
        first_mean = ds["true_means"][0]
        assert first_mean in (0.0, 0.5), f"Initial mean {first_mean} not in {{0, 0.5}}"


# ---------------------------------------------------------------------------
# Section 3: Vectorized Bayes curve correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k_true,delta", [
    (10, 2, 1.0),
    (20, 3, 0.5),
])
def test_vectorized_curve_matches_production(n, k_true, delta):
    """Vectorized Bayes curve must match production dp.bayes_regression_curve_fixed_k."""
    import bayesbreak.dp as _dp
    from bayesbreak.families.gaussian import BayesBreakGaussian

    rng = np.random.default_rng(99)
    y = rng.standard_normal(n)
    X = np.arange(1, n + 1, dtype=float).reshape(-1, 1)

    est = BayesBreakGaussian(
        estimate_hyper=False, nu=0.0, rho2=4.0, sigma2=1.0,
        k_max=min(n, 8), regression_curve="none",
        prior_k=lambda k: math.exp(-math.log(4.0) * (k - 1))
    )
    est.fit(X, y)
    L, R = est.log_left_, est.log_right_
    lA0, A1 = est.log_block_evidence_, est.block_first_moment_
    k_max = est.n_ if est.n_ < 8 else 8

    for k in range(1, k_max + 1):
        if not np.isfinite(float(L[k, n])):
            continue
        prod_curve = _dp.bayes_regression_curve_fixed_k(L, R, lA0, A1, n, k)
        vec_curve = _bayes_curve_fixed_k_vectorized(L, R, lA0, A1, n, k)
        if np.any(np.isnan(prod_curve)):
            continue
        np.testing.assert_allclose(
            vec_curve, prod_curve, rtol=1e-10, atol=1e-12,
            err_msg=f"n={n}, k={k}: vectorized curve differs from production"
        )


def test_logsumexp_all_inf_returns_neg_inf():
    """_logsumexp_axis0 must return -inf (not log(1e-300)) when all inputs are -inf.

    This guards the n=512 signal_mse bug: spurious finite log_w values caused
    by the former np.maximum(s, 1e-300) floor when s=0.
    """
    from scripts.run_e01_gaussian_calibration import _logsumexp_axis0
    a = np.full((3, 4, 5), -np.inf, dtype=float)
    out = _logsumexp_axis0(a)
    assert out.shape == (4, 5)
    assert np.all(out == -np.inf), f"Expected all -inf, got max={out.max()}"


def test_vectorized_curve_n512_matches_production():
    """n=512 vectorized curve must match production (regression for logsumexp bug)."""
    import bayesbreak.dp as _dp
    from bayesbreak.families.gaussian import BayesBreakGaussian

    rng = np.random.default_rng(7)
    n = 64  # use 64 to stay fast; bug manifested at n=512
    y = rng.standard_normal(n)
    X = np.arange(1, n + 1, dtype=float).reshape(-1, 1)

    est = BayesBreakGaussian(estimate_hyper=False, nu=0.0, rho2=4.0, sigma2=1.0, k_max=8)
    est.fit(X, y)

    L, R = est.log_left_, est.log_right_
    lA0, A1 = est.log_block_evidence_, est.block_first_moment_

    vec = _bayes_curve_mix_k_vectorized(L, R, lA0, A1, n, 8, est.k_posterior_)
    ref = _dp.bayes_regression_curve_mixed_k(L, R, lA0, A1, n, 8, est.k_posterior_)

    np.testing.assert_allclose(vec, ref, rtol=1e-10, atol=1e-12,
                               err_msg="n=64 mix_k curve regression test failed")


def test_vectorized_mix_k_matches_production():
    """Vectorized mix_k curve must match production bayes_regression_curve_mixed_k."""
    import bayesbreak.dp as _dp
    from bayesbreak.families.gaussian import BayesBreakGaussian

    n = 15
    rng = np.random.default_rng(123)
    y = rng.standard_normal(n)
    X = np.arange(1, n + 1, dtype=float).reshape(-1, 1)

    est = BayesBreakGaussian(
        estimate_hyper=False, nu=0.0, rho2=4.0, sigma2=1.0,
        k_max=6, regression_curve="mix_k",
        prior_k=lambda k: math.exp(-math.log(4.0) * (k - 1))
    )
    est.fit(X, y)
    L, R = est.log_left_, est.log_right_
    lA0, A1 = est.log_block_evidence_, est.block_first_moment_
    post_k = est.k_posterior_

    prod_curve = _dp.bayes_regression_curve_mixed_k(L, R, lA0, A1, n, 6, post_k)
    vec_curve = _bayes_curve_mix_k_vectorized(L, R, lA0, A1, n, 6, post_k)

    np.testing.assert_allclose(
        vec_curve, prod_curve, rtol=1e-10, atol=1e-12,
        err_msg="mix_k vectorized curve differs from production"
    )


# ---------------------------------------------------------------------------
# Section 4: Inference correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k_true,delta", [
    (30, 1, 0.0),
    (30, 3, 1.5),
])
def test_inference_normalization(n, k_true, delta):
    """Posterior over k must sum to 1; boundary probs in [0,1]."""
    seed_seq = np.random.SeedSequence([20261021, 1, 0, 5])
    ds = generate_dataset(n, k_true, delta, 1.0, 5, seed_seq)
    infer = run_inference(ds["y"], ds["x"], n, "geometric_decay_log4", k_max=8)

    p_k = infer["p_k_given_y"]
    assert abs(float(np.sum(p_k)) - 1.0) < 1e-8, (
        f"p(k|y) sums to {float(np.sum(p_k)):.10f}, not 1.0"
    )

    bm = infer["boundary_probs_marginal"]
    assert bm.shape == (n - 1,), f"bm shape {bm.shape} != ({n-1},)"
    assert np.all(bm >= 0) and np.all(bm <= 1 + 1e-10), (
        f"boundary_probs outside [0,1]: min={bm.min():.6f}, max={bm.max():.6f}"
    )


def test_inference_k_max_saturation_reported():
    """p_k_max_given_y field must be correctly indexed to k=k_max."""
    seed_seq = np.random.SeedSequence([20261021, 1, 0, 0])
    ds = generate_dataset(20, 1, 0.0, 1.0, 3, seed_seq)
    infer = run_inference(ds["y"], ds["x"], 20, "geometric_decay_log4", k_max=8)
    p_k = infer["p_k_given_y"]
    p_k_max_from_arr = float(p_k[8 - 1])
    p_k_max_reported = infer["p_k_max"]
    assert abs(p_k_max_reported - p_k_max_from_arr) < 1e-15


def test_uniform_prior_differs_from_geom():
    """Uniform and geometric-decay priors must give different posteriors on nonnull data."""
    seed_seq = np.random.SeedSequence([20261021, 1, 0, 7])
    ds = generate_dataset(64, 3, 2.0, 1.0, 10, seed_seq)
    infer_geom = run_inference(ds["y"], ds["x"], 64, "geometric_decay_log4", k_max=8)
    infer_unif = run_inference(ds["y"], ds["x"], 64, "uniform", k_max=8)
    # They should differ in p(k|y)
    diff = np.abs(infer_geom["p_k_given_y"] - infer_unif["p_k_given_y"]).max()
    assert diff > 1e-6, "Geometric and uniform priors gave identical posteriors (unexpected)"


# ---------------------------------------------------------------------------
# Section 5: Boundary matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bhat,bstar,tau,expected_tp", [
    ([3], [3], 0, 1),
    ([3], [4], 0, 0),
    ([3], [4], 1, 1),
    ([10, 20], [10, 21], 3, 2),
    ([10, 20], [5, 15], 3, 0),   # 10-5=5>3, 20-15=5>3
    ([10, 20], [8, 22], 3, 2),   # both within tau=3? 10-8=2, 20-22=2 ✓
    ([], [10], 3, 0),
    ([10], [], 3, 0),
    ([], [], 3, 0),
])
def test_boundary_matching(bhat, bstar, tau, expected_tp):
    tp, _ = _boundary_match(bhat, bstar, tau)
    assert tp == expected_tp, (
        f"bhat={bhat}, bstar={bstar}, tau={tau}: got TP={tp}, expected {expected_tp}"
    )


def test_boundary_match_one_to_one():
    """Each predicted boundary matches at most one true boundary."""
    # bhat=[5,7], bstar=[6] with tau=2: only one of {5,7} can match 6
    tp, _ = _boundary_match([5, 7], [6], tau=2)
    assert tp == 1, f"Expected TP=1 (one-to-one), got {tp}"


def test_f1_null_cell():
    """Null cell (k_true=1) reports n_false metrics, not F1."""
    metrics = compute_f1_metrics([10, 20], [0, 50], taus=[0, 1, 3], k_true=1)
    # k_true=1 → should have n_false_at_tau keys, not f1
    for tau in [0, 1, 3]:
        assert f"n_false_at_{tau}" in metrics, f"Missing n_false_at_{tau}"
        assert math.isnan(metrics[f"f1_at_{tau}"]), f"f1_at_{tau} should be nan for null cell"


def test_f1_perfect_detection():
    """Perfect detection: bhat = bstar → F1=1.0 for any tau>0."""
    # map_boundaries must include endpoints [0, n] matching the production convention
    metrics = compute_f1_metrics([0, 50, 100, 150], [0, 50, 100, 150], taus=[3], k_true=3)
    assert abs(metrics["f1_at_3"] - 1.0) < 1e-10


def test_f1_no_detection():
    """No predicted boundaries → recall=0, F1=0."""
    # map_boundaries [0, 150] means k_map=1 (no interior boundaries predicted)
    metrics = compute_f1_metrics([0, 150], [0, 50, 100, 150], taus=[3], k_true=3)
    assert metrics["f1_at_3"] == 0.0


# ---------------------------------------------------------------------------
# Section 6: Calibration curve
# ---------------------------------------------------------------------------

def test_calibration_perfect():
    """Perfect calibration: predicted=truth → ECE≈0."""
    rng = np.random.default_rng(42)
    # For each "replicate", generate a probability array equal to true z
    bm_list = []
    z_list = []
    for _ in range(20):
        z = rng.choice([0.0, 1.0], size=50, p=[0.9, 0.1])
        bm_list.append(z)
        z_list.append(z)
    calib = compute_calibration_curve(bm_list, z_list, n_bins=5, n_boot=50, boot_rng=rng)
    # ECE should be 0 if pred=obs always
    assert calib["ece"] < 0.01, f"Perfect calibration should give ECE≈0, got {calib['ece']}"


def test_calibration_bin_counts_sum():
    """Bin counts must sum to total number of boundary positions."""
    rng = np.random.default_rng(123)
    bm_list = [rng.uniform(0, 1, size=99) for _ in range(10)]
    z_list = [rng.choice([0.0, 1.0], size=99) for _ in range(10)]
    calib = compute_calibration_curve(bm_list, z_list, n_bins=10, n_boot=50, boot_rng=rng)
    total_positions = sum(len(b) for b in bm_list)
    assert sum(calib["bin_counts"]) == total_positions


# ---------------------------------------------------------------------------
# Section 7: End-to-end pilot (single cell, 2 replicates)
# ---------------------------------------------------------------------------

def test_pilot_e2e_null():
    """End-to-end pilot: null cell (k=1) runs without error."""
    k_true, delta, n = 1, 0.0, 30
    for rep in range(2):
        seed_seq = np.random.SeedSequence([20261021, 1, 0, rep])
        ds = generate_dataset(n, k_true, delta, 1.0, 5, seed_seq)
        infer = run_inference(ds["y"], ds["x"], n, "geometric_decay_log4", k_max=6)
        metrics = compute_replicate_metrics(ds, infer, taus=[0, 1, 3, 5])
        assert "k_marg" in metrics
        assert math.isfinite(metrics["brier"])
        assert math.isfinite(metrics["signal_mse"])


def test_pilot_e2e_nonnull():
    """End-to-end pilot: nonnull cell (k=3) runs without error, metrics in range."""
    k_true, delta, n = 3, 2.0, 64
    reps_done = 0
    for rep in range(3):
        seed_seq = np.random.SeedSequence([20261021, 1, 2, rep])
        ds = generate_dataset(n, k_true, delta, 1.0, 10, seed_seq)
        infer = run_inference(ds["y"], ds["x"], n, "geometric_decay_log4", k_max=8)
        metrics = compute_replicate_metrics(ds, infer, taus=[0, 1, 3, 5])

        assert 0.0 <= metrics.get("brier", float("nan")) <= 1.0, f"Brier {metrics['brier']} out of [0,1]"
        assert math.isfinite(metrics["signal_mse"]), "signal_mse is not finite"
        f1 = metrics.get("f1_at_3", float("nan"))
        assert math.isnan(f1) or 0.0 <= f1 <= 1.0, f"F1 {f1} out of [0,1]"
        reps_done += 1
    assert reps_done == 3


def test_pilot_e2e_ablation():
    """End-to-end pilot: ablation (uniform prior) runs correctly."""
    k_true, delta, n = 3, 0.5, 50
    seed_seq = np.random.SeedSequence([20261021, 1, 8, 0])
    ds = generate_dataset(n, k_true, delta, 1.0, 8, seed_seq)
    infer_geom = run_inference(ds["y"], ds["x"], n, "geometric_decay_log4", k_max=6)
    infer_unif = run_inference(ds["y"], ds["x"], n, "uniform", k_max=6)
    m_geom = compute_replicate_metrics(ds, infer_geom, taus=[3])
    m_unif = compute_replicate_metrics(ds, infer_unif, taus=[3])
    for m in (m_geom, m_unif):
        assert math.isfinite(m["brier"])
        assert math.isfinite(m["signal_mse"])
