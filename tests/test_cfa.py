"""Confirmatory members, posterior summaries, bagging and prediction (ported from gbscfa)."""

import importlib.metadata
from itertools import product
from types import SimpleNamespace
import json
import numpy as np
import pytest
from scipy.special import softmax

import ggbfa
from ggbfa import CFAFit, CFAConfig, BaggedCFAFit, MEMBERS
from ggbfa._bagging import BaggingConfig
from ggbfa._model import CFAResult
from ggbfa._moments import gate_pair_moments, expected_loading_quadratic
from ggbfa._two import _row_sparse_loading_moments
from ggbfa._joint import calibrate_free_covariance


def _dense_reference_row_sparse_loading_moments(
    membership, state_probability, active_mean, active_variance
):
    """Retain the original dense formula as an exact small-matrix oracle."""

    p_count, k_count = state_probability.shape
    loading_mean = np.zeros((p_count, k_count), dtype=float)
    loading_second = np.zeros((p_count, k_count, k_count), dtype=float)
    identity = np.eye(k_count, dtype=float)
    for primary in range(k_count):
        row_ids = np.flatnonzero(membership == primary)
        if row_ids.size == 0:
            continue
        base = identity[primary]
        no_weight = state_probability[row_ids, primary]
        loading_mean[row_ids] += no_weight[:, None] * base
        loading_second[row_ids] += no_weight[:, None, None] * np.outer(base, base)
        for target in range(k_count):
            if target == primary:
                continue
            probability = state_probability[row_ids, target]
            mean = active_mean[row_ids, target]
            variance = active_variance[row_ids, target]
            direction = identity[target] - base
            conditional_mean = base[None, :] + mean[:, None] * direction
            loading_mean[row_ids] += probability[:, None] * conditional_mean
            loading_second[row_ids] += probability[:, None, None] * (
                conditional_mean[:, :, None] * conditional_mean[:, None, :]
                + variance[:, None, None] * np.outer(direction, direction)
            )
    loading_covariance = loading_second - (
        loading_mean[:, :, None] * loading_mean[:, None, :]
    )
    loading_covariance = 0.5 * (
        loading_covariance + np.swapaxes(loading_covariance, 1, 2)
    )
    return loading_mean, loading_second, loading_covariance


def simulated(seed=20260902, block_size=10, n=50):
    rng = np.random.default_rng(seed)
    k = 3
    blocks = [np.arange(i * block_size, (i + 1) * block_size) for i in range(k)]
    loading = np.repeat(np.eye(k), block_size, axis=0)
    for i in range(k):
        loading[i * block_size, i] = 0.75
        loading[i * block_size, (i + 1) % k] = 0.25
    phi = 0.3 * np.ones((k, k)) + 0.7 * np.eye(k)
    f = rng.multivariate_normal(np.zeros(k), phi, size=n)
    x = f @ loading.T + 0.5 * rng.normal(size=(n, len(loading)))
    return x, blocks


def config(option="mf", **kwargs):
    return CFAConfig(
        member=option,
        max_iter=600,
        tolerance=1e-6,
        factor_covariance_interval_draws=100,
        factor_covariance_interval_seed=19,
        **kwargs,
    )


@pytest.fixture(scope="module", params=MEMBERS)
def fitted(request):
    x, blocks = simulated()
    return CFAFit(config(request.param)).fit(x, blocks)


def test_public_api():
    assert MEMBERS == ("1f", "2f+s", "2f", "mf")
    assert ggbfa.__version__ == "0.1.0"
    x, blocks = simulated(n=40)
    fit = ggbfa.cfa(x, blocks, member="2f+s", device="cpu")
    assert isinstance(fit, CFAFit) and fit.member_ == "2f+s"
    assert fit.method_label_ == "gg-BFA (2f+s)"
    assert [list(c) for c in fit.communities_] == [list(b) for b in blocks]
    free = ggbfa.cfa(x, blocks, member="mf", loading_normalization="free", device="cpu")
    assert free.method_label_ == "gg-BFA (mf, free)"
    bagged = ggbfa.cfa(x, blocks, member="2f", device="cpu", bagging=2, bagging_draws=3)
    assert isinstance(bagged, BaggedCFAFit) and bagged.method_label_ == "bagged gg-BFA (2f)"


def test_cfa_accepts_an_efa_like_result():
    x, blocks = simulated(n=40)
    result = SimpleNamespace(communities=[np.asarray(b) for b in blocks])
    a = ggbfa.cfa(x, result, member="2f", device="cpu")
    b = ggbfa.cfa(x, blocks, member="2f", device="cpu")
    np.testing.assert_array_equal(a.covariance(), b.covariance())


def test_sparse_index_loading_moments_match_original_dense_formula():
    rng = np.random.default_rng(260914)
    membership = np.array([0, 1, 2, 3, 4, 0, 2, 4, 1, 3, 3, 0, 4])
    state_probability = rng.dirichlet(np.arange(1.0, 6.0), size=len(membership))
    active_mean = rng.normal(scale=0.35, size=state_probability.shape)
    active_variance = rng.uniform(0.01, 0.4, size=state_probability.shape)
    active_mean[np.arange(len(membership)), membership] = 0.0
    active_variance[np.arange(len(membership)), membership] = 0.0

    expected = _dense_reference_row_sparse_loading_moments(
        membership, state_probability, active_mean, active_variance
    )
    actual = _row_sparse_loading_moments(
        membership, state_probability, active_mean, active_variance
    )
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference, rtol=2e-15, atol=2e-15)


def test_build_metadata_uses_the_runtime_version_and_registers_extras():
    assert importlib.metadata.version("ggbfa") == ggbfa.__version__
    requirements = importlib.metadata.requires("ggbfa")
    assert any("cupy-cuda12x" in r and "gpu" in r for r in requirements)
    assert any("matplotlib" in r and "plot" in r for r in requirements)


@pytest.mark.parametrize(
    "name",
    [
        "loading_prior_mean",
        "loading_prior_variance",
        "enforce_loading_row_sums",
        "estimate_primary_loadings",
        "estimate_cross_loadings",
        "estimate_factor_covariance",
        "likelihood_fraction",
        "discovery_threshold",
    ],
)
def test_unknown_settings_rejected(name):
    with pytest.raises(TypeError):
        CFAConfig(**{name: False})
    x, blocks = simulated(n=10)
    with pytest.raises(TypeError):
        ggbfa.cfa(x, blocks, **{name: False})


def test_member_names():
    assert CFAConfig(member=" 2F+S ").member == "2f+s"
    for name in ("two_sparse", "two+sparse", "multiple", "vb_scfa", "single_sparse"):
        with pytest.raises(ValueError):
            CFAConfig(member=name)


def test_unified_posteriors(fitted):
    assert isinstance(fitted.result_, CFAResult)
    p, k = fitted.loadings_.shape
    expected = {
        "F": (50, k),
        "Lambda": (p, k),
        "gamma": (p, k),
        "psi": (p,),
        "Phi": (k, k),
    }
    for name, shape in expected.items():
        q = fitted.posterior(name)
        assert (
            q.mean.shape
            == q.sd.shape
            == q.lower.shape
            == q.upper.shape
            == q.applicable.shape
            == shape
        )
        assert np.all(np.isfinite(q.mean)) and np.all(q.sd >= 0)
        assert np.all(q.upper >= q.lower)
    primary = fitted.result_.posterior.loading_mask.astype(bool)
    assert not fitted.posterior("gamma").applicable[primary].any()
    assert not fitted.cross_loading_pip_[primary].any()
    np.testing.assert_array_equal(
        fitted.factor_scores_, fitted.result_.posterior.factor_score_mean
    )
    assert fitted.metadata_["converged"]
    assert fitted.timing_["fit_seconds"] > 0 and fitted.timing_["interval_seconds"] > 0


def test_joint_posterior_draws_and_constraints(fitted):
    first = fitted.sample_posterior(300, random_state=15)
    second = fitted.sample_posterior(300, random_state=15)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(
        fitted.sample_loadings(300, random_state=15), first["Lambda"]
    )
    np.testing.assert_allclose(first["Lambda"].sum(axis=2), 1, atol=1e-12)
    np.testing.assert_allclose(
        first["Phi"], first["Phi"].transpose(0, 2, 1), atol=1e-12
    )
    assert np.linalg.eigvalsh(first["Phi"]).min() > 0
    assert np.all(first["psi"] > 0)
    primary = fitted.result_.posterior.loading_mask.astype(bool)
    assert not first["gamma"][:, primary].any()
    if fitted.member_ != "mf":
        np.testing.assert_array_equal(
            first["gamma"], ((first["Lambda"] != 0) & ~primary[None]).astype(int)
        )
        assert first["gamma"].sum(axis=2).max() <= 1
    if fitted.member_ == "2f+s":
        membership = primary.argmax(axis=1)
        for h, budget in enumerate(fitted.overlap_budget_):
            assert first["gamma"][:, membership == h].sum(axis=(1, 2)).max() <= budget


def test_covariance_is_posterior_mean_not_plugin(fitted):
    result = fitted.result_
    post = result.posterior
    actual = fitted.covariance(selected_only=True, standardized=True)
    plugin = (
        post.loading_mean @ post.factor_prior_covariance @ post.loading_mean.T
        + np.diag(post.residual_variance_mean)
    )
    diagonal = np.einsum(
        "ab,pba->p", post.factor_prior_covariance, result.diagnostics.loading_covariance
    )
    np.testing.assert_allclose(np.diag(actual - plugin), diagonal, atol=1e-12)
    if fitted.member_ == "1f":
        np.testing.assert_array_equal(diagonal, 0)
    else:
        assert diagonal.max() > 0
    if fitted.member_ != "2f+s":
        np.testing.assert_allclose(actual - plugin, np.diag(diagonal), atol=1e-12)
    raw = fitted.covariance(selected_only=True)
    np.testing.assert_allclose(
        raw, actual * np.outer(post.data_scale, post.data_scale), atol=1e-12
    )
    assert np.linalg.eigvalsh(actual).min() > 0


@pytest.mark.parametrize(
    "log_odds",
    [
        np.zeros(6),
        np.arange(6.0) - 2,
        np.array([-1000.0, 1000.0, -1000.0, 1000.0, 0.0, 0.0]),
    ],
)
def test_gate_joint_moments_against_exact_enumeration(log_odds):
    for budget in range(7):
        states = np.array([g for g in product([0, 1], repeat=6) if sum(g) <= budget])
        probabilities = softmax(states @ log_odds)
        truth = states.T @ (probabilities[:, None] * states)
        np.testing.assert_allclose(
            gate_pair_moments(log_odds, budget), truth, atol=2e-12
        )
        ids = np.array([5, 0, 2])
        np.testing.assert_allclose(
            gate_pair_moments(log_odds, budget, indices=ids),
            truth[np.ix_(ids, ids)],
            atol=2e-12,
        )


def test_budget_covariance_includes_nonzero_interrow_correction():
    membership = np.array([0, 0, 1, 1])
    p, k = 4, 2
    state = np.full((p, k), 1 / 3)
    state[np.arange(p), membership] = 2 / 3
    target = np.ones((p, k))
    target[np.arange(p), membership] = 0
    amplitude = np.array([[0, 0.7], [0, -0.4], [0.5, 0], [-0.3, 0]])
    mean, second, cov = _row_sparse_loading_moments(
        membership, state, amplitude, np.zeros((p, k))
    )
    post = SimpleNamespace(
        loading_mean=mean,
        member="2f+s",
        loading_mask=np.eye(k)[membership],
        row_overlap_probability=np.full(p, 1 / 3),
    )
    diag = SimpleNamespace(
        loading_covariance=cov,
        conditional_target_probability=target,
        active_state_means=amplitude,
        overlap_budget_by_factor=np.ones(k, int),
        gate_log_odds=np.zeros(p),
    )
    result = SimpleNamespace(posterior=post, diagnostics=diag)
    phi = np.array([[1.0, 0.3], [0.3, 0.8]])
    expected = np.zeros((p, p))
    for left, right in product([-1, 0, 1], repeat=2):
        loading = np.eye(k)[membership].copy()
        for h, chosen in enumerate([left, right]):
            if chosen >= 0:
                row = 2 * h + chosen
                other = 1 - h
                loading[row, other] = amplitude[row, other]
                loading[row, h] = 1 - amplitude[row, other]
        expected += loading @ phi @ loading.T / 9
    actual = expected_loading_quadratic(result, phi)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    diagonal_only = mean @ phi @ mean.T + np.diag(np.einsum("ab,pba->p", phi, cov))
    assert abs(actual[0, 1] - diagonal_only[0, 1]) > 0.01
    np.testing.assert_allclose(
        expected_loading_quadratic(result, phi, indices=[3, 0, 1]),
        expected[np.ix_([3, 0, 1], [3, 0, 1])],
        atol=1e-12,
    )


def test_prediction_no_hidden_truth_and_full_second_moments(fitted):
    x, _ = simulated()
    mask = np.zeros(x.shape[1], bool)
    mask[[0, 2, 12]] = True
    row = x[0].copy()
    row[mask] = np.nan
    first = fitted.predict_masked(row, mask)
    row[mask] = 1e30
    second = fitted.predict_masked(row, mask)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.covariance, second.covariance)
    assert np.linalg.eigvalsh(first.covariance).min() > 0
    local = np.flatnonzero(mask)
    p = fitted.result_.posterior
    standardized = (x[0, ~mask] - p.data_mean[~mask]) / p.data_scale[~mask]
    means, v = fitted._local_factors(standardized[None], ~mask)
    m = means[0]
    mu = p.loading_mean[local] @ m
    expected = expected_loading_quadratic(
        fitted.result_, v + np.outer(m, m), indices=local
    ) - np.outer(mu, mu)
    expected += np.diag(p.residual_variance_mean[local])
    expected *= np.outer(p.data_scale[local], p.data_scale[local])
    np.testing.assert_allclose(first.covariance, expected, atol=1e-11)


def test_export_json_csv(fitted, tmp_path):
    fitted.save_results(tmp_path)
    document = json.loads((tmp_path / "posterior.json").read_text())
    assert set(document["posterior"]) == {"F", "Lambda", "gamma", "psi", "Phi"}
    assert document["metadata"]["interval_seed"] == 19
    assert "numpy" in document["metadata"]["runtime"]
    assert (
        len(document["preprocessing"]["selected_data_scale"])
        == fitted.loadings_.shape[0]
    )
    assert (tmp_path / "posterior.csv").is_file() and (
        tmp_path / "diagnostics.csv"
    ).is_file()


@pytest.fixture(scope="module", params=MEMBERS)
def bagged(request):
    x, blocks = simulated(block_size=4, n=30)
    return BaggedCFAFit(
        BaggingConfig(
            base_config=config(
                request.param,
                max_overlap_fraction=0.5,
                row_overlap_prior_probability=0.1,
            ),
            n_bootstrap=3,
            posterior_draws_per_bootstrap=12,
            random_state=14,
        )
    ).fit(x, blocks)


def test_bagging_defaults_and_lazy_covariance(bagged):
    assert BaggingConfig().n_bootstrap == 20
    assert bagged.metadata_["bootstrap_size"] == 60
    assert bagged.bootstrap_ratio_ == 2
    assert not bagged.metadata_["joint_covariance_materialized"]
    assert not hasattr(bagged, "posterior_covariance_")
    assert all(
        m.result_.posterior.reconstruction_mean.size == 0
        for m in bagged.bootstrap_models_
    )
    for name in ["F", "Lambda", "gamma", "psi", "Phi"]:
        parts = [bagged._component_summary(i, name) for i in range(3)]
        mean = np.mean([p.mean for p in parts], axis=0)
        np.testing.assert_allclose(bagged.posterior(name).mean, mean, atol=1e-12)
        if name != "gamma":
            variance = np.mean([p.sd**2 + (p.mean - mean) ** 2 for p in parts], axis=0)
            np.testing.assert_allclose(
                bagged.posterior(name).sd ** 2, variance, atol=1e-12
            )


def test_bagged_joint_draws_preserve_structures(bagged):
    samples = bagged.sample_posterior(200, random_state=5)
    repeated = bagged.sample_posterior(200, random_state=5)
    for name in samples:
        np.testing.assert_array_equal(samples[name], repeated[name])
    np.testing.assert_allclose(samples["Lambda"].sum(axis=2), 1, atol=1e-12)
    if bagged.member_ != "mf":
        assert samples["gamma"].sum(axis=2).max() <= 1
    if bagged.member_ == "2f+s":
        for h, budget in enumerate(bagged.overlap_budget_):
            assert (
                samples["gamma"][:, h * 4 : (h + 1) * 4].sum(axis=(1, 2)).max()
                <= budget
            )


def test_calibrated_joint_covariance_preserves_affine_row_constraints(bagged):
    result = bagged.joint_covariance(draws=400, random_state=3)
    assert np.linalg.eigvalsh(result.stabilized).min() > -1e-10
    for matrix in [result.raw, result.adjusted, result.stabilized]:
        for row in range(12):
            direction = np.zeros(len(result.labels))
            direction[row * 3 : (row + 1) * 3] = 1
            np.testing.assert_allclose(matrix @ direction, 0, atol=1e-11)
    np.testing.assert_allclose(
        result.stabilized,
        result.affine_map @ result.free_stabilized @ result.affine_map.T,
        atol=1e-12,
    )
    # Phi is one retained variational block, not independent scalar factors.
    phi_ids = [
        i for i, label in enumerate(result.free_labels) if label.startswith("Phi")
    ]
    np.testing.assert_allclose(
        result.free_adjusted[np.ix_(phi_ids, phi_ids)],
        result.free_raw[np.ix_(phi_ids, phi_ids)],
    )
    assert "not a guarantee" in result.qualification


def test_scalar_covariance_rule_is_recovered():
    c = np.array([[1.0, 0.2], [0.2, 2.0]])
    np.testing.assert_allclose(
        calibrate_free_covariance(c, [0, 1], 2), [[1.0, 0.4], [0.4, 2.0]]
    )
    np.testing.assert_allclose(calibrate_free_covariance(c, [0, 0], 2), c)


def test_bagging_observation_covariance_and_prediction(bagged):
    actual = bagged.covariance(selected_only=True, standardized=True)
    expected = np.mean(
        [
            m.covariance(selected_only=True, standardized=True)
            for m in bagged.bootstrap_models_
        ],
        axis=0,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    x, _ = simulated(block_size=4, n=30)
    mask = np.zeros(12, bool)
    mask[[1, 5]] = True
    row = x[0].copy()
    row[mask] = np.nan
    first = bagged.predict_masked(row, mask)
    row[mask] = -1e40
    second = bagged.predict_masked(row, mask)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.covariance, second.covariance)
    assert np.linalg.eigvalsh(first.covariance).min() > 0


def test_joint_allocation_guard_and_selected_parameter_family():
    x, blocks = simulated(block_size=3, n=20)
    m = BaggedCFAFit(
        BaggingConfig(
            base_config=config(),
            n_bootstrap=2,
            posterior_draws_per_bootstrap=3,
            joint_max_parameters=10,
        )
    ).fit(x, blocks)
    with pytest.raises(ValueError, match="joint_max_parameters"):
        m.joint_covariance()
    result = m.joint_covariance(parameters=("Phi",), draws=50)
    assert result.raw.shape == (6, 6)


def test_small_block_budget_is_exactly_zero():
    x, blocks = simulated(block_size=4, n=20)
    m = CFAFit(config("2f+s")).fit(x, blocks)
    np.testing.assert_array_equal(m.overlap_budget_, [0, 0, 0])
    assert not m.sample_posterior(30, parameters=("gamma",))["gamma"].any()


def test_one_factor_and_residual_only_columns(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(25, 5))
    m = CFAFit(config()).fit(x, [np.arange(4)])
    assert m.sample_posterior(1)["Phi"].shape == (1, 1, 1)
    covariance = m.covariance()
    np.testing.assert_allclose(covariance[4, 4], x[:, 4].var())
    np.testing.assert_array_equal(covariance[4, :4], 0)
    m.save_results(tmp_path)


def test_bagging_fit_never_constructs_a_joint_parameter_covariance(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fit must not call np.cov for a joint parameter matrix")

    monkeypatch.setattr(np, "cov", forbidden)
    x, blocks = simulated(block_size=3, n=20)
    model = BaggedCFAFit(
        BaggingConfig(
            base_config=config(), n_bootstrap=2, posterior_draws_per_bootstrap=3
        )
    ).fit(x, blocks)
    assert not model.metadata_["joint_covariance_materialized"]


def test_chunked_phi_intervals_are_symmetric():
    x, blocks = simulated(block_size=3, n=20)
    model = BaggedCFAFit(
        BaggingConfig(
            base_config=config(),
            n_bootstrap=2,
            posterior_draws_per_bootstrap=12,
            summary_chunk_size=1,
        )
    ).fit(x, blocks)
    lower, upper = model.factor_covariance_ci_
    np.testing.assert_allclose(lower, lower.T, atol=1e-12)
    np.testing.assert_allclose(upper, upper.T, atol=1e-12)
