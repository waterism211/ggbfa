"""Analytic tolerances are 1e-9/1e-10; MC checks use pre-specified 6-SE bounds."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm, t

from ggbfa import CFAFit, CFAConfig, BaggedCFAFit
from ggbfa._bagging import BaggingConfig
from ggbfa._prediction import GaussianPredictor
from ggbfa._prediction import checked_spd


def fixture_data():
    rng = np.random.default_rng(391)
    blocks = [np.arange(i * 10, (i + 1) * 10) for i in range(3)]
    loading = np.repeat(np.eye(3), 10, axis=0)
    loading[0] = [0.8, 0.2, 0]
    factors = rng.multivariate_normal(np.zeros(3), 0.7 * np.eye(3) + 0.3, size=55)
    x = factors @ loading.T + 0.5 * rng.normal(size=(55, 30))
    x = np.column_stack((x, rng.normal(size=(55, 2))))
    x = x * np.linspace(0.5, 2, 32) + np.arange(32) / 3
    mask = np.zeros(32, bool)
    mask[[0, 1, 12, 22, 31]] = True
    return x, blocks, mask


@pytest.fixture(
    scope="module",
    params=[(o, b) for o in ("mf", "2f", "2f+s") for b in (False, True)],
)
def model(request):
    option, bagged = request.param
    x, blocks, _ = fixture_data()
    config = CFAConfig(
        member=option, max_iter=600, factor_covariance_interval_draws=100
    )
    estimator = (
        BaggedCFAFit(
            BaggingConfig(
                base_config=config,
                n_bootstrap=3,
                posterior_draws_per_bootstrap=15,
                random_state=7,
            )
        )
        if bagged
        else CFAFit(config)
    )
    return estimator.fit(x[:-1], blocks)


def test_gaussian_direct_reference_permutation_and_copy():
    c = np.array([[3, 0.4, 0.2], [0.4, 2, -0.7], [0.2, -0.7, 1.4]])
    mean = np.array([10.0, 20.0, -3.0])
    row, mask = np.array([1.0, 4.0, 5.0]), np.array([True, False, True])
    predictor = GaussianPredictor(mean, c)
    a = predictor.predict(row, mask)
    expected = mean[mask] + c[mask, 1] / c[1, 1] * (row[1] - mean[1])
    expected_cov = c[np.ix_(mask, mask)] - np.outer(c[mask, 1], c[mask, 1]) / c[1, 1]
    np.testing.assert_allclose(a.mean, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(a.covariance, expected_cov, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(a.lower, a.mean - norm.ppf(0.975) * np.sqrt(a.variance))
    order = np.array([2, 1, 0])
    other = GaussianPredictor(mean[order], c[np.ix_(order, order)]).predict(
        row[order], mask[order]
    )
    np.testing.assert_allclose(other.mean, a.mean[::-1], rtol=1e-9, atol=1e-10)
    c[:] = 100
    mean[:] = 0
    np.testing.assert_array_equal(predictor.predict(row, mask).mean, a.mean)


def test_fixed_global_gaussian_and_factor_moments_agree():
    loading = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    phi, psi = np.array([[1.5, 0.3], [0.3, 0.9]]), np.array([0.2, 0.4, 0.3])
    center, scale = np.array([2.0, -1.0, 3.0]), np.array([2.0, 3.0, 0.5])
    m = CFAFit()
    m.n_features_in_, m.selected_indices_ = 3, np.arange(3)
    m.training_mean_, m.training_variance_ = center, np.ones(3)
    m.result_ = SimpleNamespace(
        posterior=SimpleNamespace(
            loading_mean=loading,
            member="mf",
            residual_variance_mean=psi,
            data_mean=center,
            data_scale=scale,
        ),
        diagnostics=SimpleNamespace(
            loading_covariance=np.zeros((3, 2, 2)),
            residual_shape=np.ones(3),
            residual_scale=psi,
            factor_covariance_posterior_df=1.0,
            factor_covariance_posterior_scale=phi,
        ),
    )
    row, mask = np.array([20.0, 0.5, 4.0]), np.array([True, False, True])
    covariance = (loading @ phi @ loading.T + np.diag(psi)) * np.outer(scale, scale)
    a = GaussianPredictor(center, covariance).predict(row, mask)
    b = m._posterior_moments(row, mask)
    np.testing.assert_allclose(a.mean, b.mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(a.covariance, b.covariance, rtol=1e-9, atol=1e-10)


def test_fitted_gaussian_uses_exact_covariance_and_caches(model, monkeypatch):
    x, _, mask = fixture_data()
    expected = GaussianPredictor(
        model.training_mean_,
        model.covariance(),
        selected_indices=model.selected_indices_,
    ).predict(x[-1], mask)
    actual = model.predict_gaussian(x[-1], mask)
    np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        actual.covariance, expected.covariance, rtol=1e-9, atol=1e-10
    )

    def forbidden(*a, **k):
        raise AssertionError("cached prediction must not refit or recompute covariance")

    monkeypatch.setattr(model, "fit", forbidden)
    monkeypatch.setattr(model, "covariance", forbidden)
    np.testing.assert_array_equal(model.predict_gaussian(x[-1], mask).mean, actual.mean)
    assert model.timing_["gaussian_prepare_seconds"] > 0


def test_quantiles_seed_alias_no_leakage_or_posterior_mutation(model):
    x, _, mask = fixture_data()
    original = {name: p.mean.copy() for name, p in model.posterior_.items()}
    row = x[-1].astype(object)
    row[mask] = "hidden is not even numeric"
    options = dict(draws=1500, random_state=45, batch_size=256, return_draws=True)
    a = model.predict_posterior(row, mask, **options)
    row[mask] = np.nan
    b = model.predict_masked(row, mask, **options)
    for field in ("mean", "covariance", "lower", "upper", "draws"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(
        [a.lower, a.upper],
        np.quantile(
            a.draws, [(1 - a.level) / 2, (1 + a.level) / 2], axis=0, method="linear"
        ),
    )
    c = model.predict_posterior(row, mask, **{**options, "random_state": 46})
    np.testing.assert_array_equal(a.mean, c.mean)
    np.testing.assert_array_equal(a.covariance, c.covariance)
    assert not np.array_equal(a.draws, c.draws)
    row[~mask] = np.asarray(row[~mask], float) + 2
    changed = model.predict_posterior(row, mask, **options)
    assert np.max(np.abs(changed.mean - a.mean)) > 0.01
    for name in original:
        np.testing.assert_array_equal(model.posterior(name).mean, original[name])
    assert (
        a.prediction_type == "variational_posterior" and a.quantile_method == "linear"
    )
    assert a.residual_only_indices.tolist() == [31]


def test_predictive_draw_moments_include_joint_uncertainty_and_noise(model):
    x, _, mask = fixture_data()
    p = model.predict_posterior(
        x[-1], mask, draws=20000, random_state=71, return_draws=True
    )
    centered = p.draws - p.mean
    # Pre-specified six estimated MC standard errors plus numerical tolerance.
    mean_se = np.sqrt(p.variance / len(centered))
    assert np.all(np.abs(p.draws.mean(axis=0) - p.mean) <= 6 * mean_se + 1e-10)
    products = centered[:, :, None] * centered[:, None, :]
    covariance_se = products.std(axis=0, ddof=1) / np.sqrt(len(centered))
    assert np.all(
        np.abs(products.mean(axis=0) - p.covariance) <= 6 * covariance_se + 1e-10
    )
    gaussian = model.predict_gaussian(x[-1], mask)
    assert not np.allclose(gaussian.covariance, p.covariance, rtol=1e-4, atol=1e-5)


def test_bagged_total_variance(model):
    if not isinstance(model, BaggedCFAFit):
        return
    x, _, mask = fixture_data()
    reference, mask, ids, scales, centers, selected = model._prediction_reference(
        x[-1], mask
    )
    parts = [m._posterior_moments(reference, mask) for m in model.bootstrap_models_]
    means = np.array([p.mean * scales + centers for p in parts])
    covs = np.array([p.covariance * np.outer(scales, scales) for p in parts])
    overall = means.mean(axis=0)
    between = (means - overall).T @ (means - overall) / len(parts)
    assert np.diag(between)[selected].max() > 1e-6
    result = model._posterior_moments(x[-1], mask)
    expected = covs.mean(axis=0) + between
    np.testing.assert_allclose(
        result.mean[selected], overall[selected], rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        result.covariance[np.ix_(selected, selected)],
        expected[np.ix_(selected, selected)],
        rtol=1e-9,
        atol=1e-10,
    )


def test_no_observed_selected_and_all_residual_only_prediction(model):
    x, _, _ = fixture_data()
    mask = np.ones(32, bool)
    mask[30:] = False
    p = model.predict_posterior(x[-1], mask, draws=10)
    assert p.selected_masked_indices.size == 30 and np.isfinite(p.mean).all()
    mask = ~mask
    p = model.predict_posterior(x[-1], mask, draws=100)
    np.testing.assert_array_equal(p.mean, model.training_mean_[30:])
    np.testing.assert_array_equal(p.variance, model.training_variance_[30:])
    np.testing.assert_array_equal(p.covariance, np.diag(p.variance))


@pytest.mark.parametrize("value", [0, 1, -1, np.nan, True, "0.95"])
def test_bad_level(value):
    with pytest.raises(ValueError):
        GaussianPredictor([0, 0], np.eye(2)).predict(
            [np.nan, 2], [True, False], level=value
        )


@pytest.mark.parametrize(
    "mask,row",
    [
        ([1, 0], [1, 2]),
        ([False, False], [1, 2]),
        ([True, True], [1, 2]),
        ([True, False], [[1, 2]]),
        ([True, False], [1, np.nan]),
    ],
)
def test_invalid_inputs(mask, row):
    with pytest.raises(ValueError):
        GaussianPredictor([0, 0], np.eye(2)).predict(row, mask)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 2)),
        [[1, 2], [2, 1]],
        [[1, 0.1], [0.2, 1]],
        [[1, np.nan], [np.nan, 1]],
    ],
)
def test_no_silent_regularization(matrix):
    with pytest.raises(ValueError):
        checked_spd(matrix, "test")


def test_bad_sampling_arguments_and_residual_variance(model):
    x, _, mask = fixture_data()
    for key, value in [
        ("draws", 0),
        ("draws", 2.5),
        ("draws", True),
        ("random_state", -1),
        ("random_state", np.nan),
        ("batch_size", 0),
        ("return_draws", 1),
    ]:
        with pytest.raises(ValueError):
            model.predict_posterior(x[-1], mask, **{key: value})
    broken = copy.deepcopy(model)
    broken.training_variance_[31] = 0
    broken._reset_prediction()
    with pytest.raises(ValueError, match="positive definite"):
        broken.predict_gaussian(x[-1], mask)
    with pytest.raises(ValueError, match="positive definite"):
        broken.predict_posterior(x[-1], mask, draws=1)


def test_refit_invalidates_cache(model):
    m = copy.deepcopy(model)
    x, blocks, mask = fixture_data()
    before = m.predict_gaussian(x[-1], mask)
    m.fit(x[:-1] + 7, blocks)
    assert m._gaussian_predictor is None
    assert m.timing_["gaussian_prepare_seconds"] == 0
    after = m.predict_gaussian(x[-1] + 7, mask)
    np.testing.assert_allclose(after.mean, before.mean + 7, rtol=1e-9, atol=1e-10)


def test_sampling_requests_joint_loadings_and_not_training_f(model, monkeypatch):
    import ggbfa._cfa as module

    original = module.draw_fitted
    calls = []

    def audited(result, count, seed, parameters):
        assert parameters == ("Lambda", "psi")
        draws = original(result, count, seed, parameters)
        loading = draws["Lambda"]
        np.testing.assert_allclose(loading.sum(axis=-1), 1, atol=1e-12)
        primary = result.posterior.loading_mask.astype(bool)
        cross = (loading != 0) & ~primary[None]
        if result.posterior.member != "mf":
            assert cross.sum(axis=-1).max() <= 1
        if result.posterior.member == "2f+s":
            for h, budget in enumerate(result.diagnostics.overlap_budget_by_factor):
                assert (
                    cross[:, primary.argmax(axis=1) == h].sum(axis=(1, 2)).max()
                    <= budget
                )
        calls.append(count)
        return draws

    monkeypatch.setattr(module, "draw_fitted", audited)
    x, _, mask = fixture_data()
    model.predict_posterior(x[-1], mask, draws=700, batch_size=128)
    assert calls and max(calls) <= 128 and sum(calls) == 700


def test_bagged_residual_only_is_not_a_component_mixture(model):
    if not isinstance(model, BaggedCFAFit):
        return
    m = copy.deepcopy(model)
    x, _, mask = fixture_data()
    # Excluded bootstrap columns may be constant. They are not part of q and
    # must not replace the explicitly chosen full-training residual convention.
    for component in m.bootstrap_models_:
        component.training_variance_[31] = 0
        component.training_mean_[31] = 1e5
    p = m.predict_posterior(
        x[-1], mask, draws=20000, random_state=61, return_draws=True
    )
    pos = np.flatnonzero(p.masked_indices == 31)[0]
    assert p.mean[pos] == m.training_mean_[31]
    assert p.variance[pos] == m.training_variance_[31]
    assert abs(p.draws[:, pos].mean() - p.mean[pos]) < 6 * np.sqrt(
        p.variance[pos] / 20000
    )


def test_non_gaussian_known_quantiles_include_observation_noise(monkeypatch):
    rng = np.random.default_rng(53)
    x = rng.normal(size=(40, 3))
    m = CFAFit(CFAConfig(member="mf", factor_covariance_interval_draws=100)).fit(
        x, [np.arange(3)]
    )
    # Deterministic one-factor loadings and essentially zero local factor signal;
    # IG(3,2) residual variance implies Student-t_6 noise at scale sqrt(2/3).
    m.result_ = replace(
        m.result_,
        posterior=replace(
            m.result_.posterior,
            data_mean=np.zeros(3),
            data_scale=np.ones(3),
            residual_variance_mean=np.ones(3),
        ),
        diagnostics=replace(
            m.result_.diagnostics,
            residual_shape=np.full(3, 3.0),
            residual_scale=np.full(3, 2.0),
        ),
    )
    monkeypatch.setattr(
        m,
        "_local_factors",
        lambda values, observed: (np.zeros((1, 1)), np.array([[1e-12]])),
    )
    p = m.predict_posterior(
        [np.nan, 0, 0],
        [True, False, False],
        level=0.99,
        draws=40000,
        random_state=231,
        return_draws=True,
    )
    expected = t.ppf([0.005, 0.995], 6) * np.sqrt(2 / 3)
    np.testing.assert_allclose([p.lower[0], p.upper[0]], expected, rtol=0, atol=0.20)
    assert p.upper[0] > norm.ppf(0.995) + 0.2
    assert p.draws.var() > 0.8  # would be ~1e-12 if observation noise were omitted
    smaller = m.predict_posterior(
        [np.nan, 0, 0], [True, False, False], level=0.99, draws=20000, random_state=231
    )
    np.testing.assert_allclose(
        [smaller.lower, smaller.upper], [p.lower, p.upper], rtol=0, atol=0.25
    )
