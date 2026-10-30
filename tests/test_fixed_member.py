"""Member 1f: priors, shared residuals, point-mass loadings, prediction and bagging."""
from dataclasses import replace

import numpy as np
import pytest

import ggbfa
from ggbfa import CFAFit, CFAConfig, BaggedCFAFit
from ggbfa._bagging import BaggingConfig
from ggbfa._model import fit_member
from ggbfa._moments import observation_covariance


def data(seed=453, n=55, sizes=(4, 5, 3)):
    rng = np.random.default_rng(seed)
    k = len(sizes)
    loading = np.repeat(np.eye(k), sizes, axis=0)
    f = rng.multivariate_normal(np.zeros(k), 0.7 * np.eye(k) + 0.3 * np.ones((k, k)), n)
    return (
        f @ loading.T
        + rng.normal(size=(n, sum(sizes))) * 0.6
        + np.arange(sum(sizes)) / 9
    )


def settings(**overrides):
    return CFAConfig(
        **{
            "member": "1f",
            "standardize_data": False,
            "max_iter": 1000,
            "tolerance": 1e-8,
            "elbo_decrease_tolerance": 1e-9,
            "residual_prior_shape": 0.001,
            "residual_prior_scale": 0.001,
            "factor_covariance_interval_draws": 100,
            "factor_covariance_interval_seed": 73,
            **overrides,
        }
    )


def test_explicit_nondefault_priors_and_function_dispatch():
    x = data()
    blocks = [np.arange(4), np.arange(4, 9), np.arange(9, 12)]
    cfg = settings(
        factor_covariance_prior_strength=2.5,
        factor_prior_variance=1.4,
        residual_prior_shape=2.0,
        residual_prior_scale=0.7,
    )
    a = fit_member(x, blocks, cfg)
    b = ggbfa.cfa(x, blocks, **{f: getattr(cfg, f) for f in cfg.__dataclass_fields__})
    np.testing.assert_array_equal(
        a.posterior.factor_score_mean, b.result_.posterior.factor_score_mean
    )
    d = a.diagnostics
    assert d.factor_covariance_prior_df == 6.5  # K + 1 + strength
    np.testing.assert_allclose(d.factor_covariance_prior_scale, 3.5 * np.eye(3))
    assert d.factor_covariance_posterior_df == 6.5 + len(x)
    assert a.posterior.member == "1f"


def test_fixed_draws_psi_blocks_and_point_mass_applicability(tmp_path):
    x = data()
    blocks = [np.array([9, 4, 1]), np.array([8, 3, 0, 7])]
    model = CFAFit(settings(standardize_data=True)).fit(x, blocks)
    group = model.result_.diagnostics.residual_group_index
    samples = model.sample_posterior(400, random_state=19)
    for name in ("Lambda", "gamma"):
        q = model.posterior(name)
        assert not q.applicable.any()
        np.testing.assert_array_equal(q.sd, 0)
        np.testing.assert_array_equal(q.lower, q.upper)
    np.testing.assert_array_equal(
        samples["Lambda"], np.broadcast_to(model.loadings_, (400, 7, 2))
    )
    np.testing.assert_array_equal(samples["gamma"], 0)
    for h in range(2):
        ids = np.flatnonzero(group == h)
        np.testing.assert_array_equal(
            samples["psi"][:, ids],
            np.repeat(samples["psi"][:, ids[:1]], len(ids), axis=1),
        )
    assert np.corrcoef(samples["psi"][:, 0], samples["psi"][:, -1])[0, 1] < 0.2
    p = model.result_.posterior
    expected = p.loading_mask @ model.factor_covariance_ @ p.loading_mask.T + np.diag(
        model.residual_variance_
    )
    np.testing.assert_allclose(observation_covariance(model.result_), expected)
    assert model.method_label_ == "gg-BFA (1f)"
    assert model.metadata_["residual_structure"] == "community_shared"
    model.save_results(tmp_path)
    assert (tmp_path / "posterior.json").is_file()


def test_fixed_gaussian_and_variational_prediction_no_leakage():
    x = data(n=65)
    blocks = [np.arange(4), np.arange(4, 9)]  # explicit residual-only last block
    model = CFAFit(settings()).fit(x[:55], blocks)
    mask = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], bool)
    row = x[55].copy()
    hidden = row.copy()
    hidden[mask] = np.nan
    for name, kwargs in [
        ("predict_gaussian", {}),
        (
            "predict_posterior",
            {"draws": 12000, "random_state": 6, "return_draws": True},
        ),
    ]:
        one = getattr(model, name)(row, mask, **kwargs)
        two = getattr(model, name)(hidden, mask, **kwargs)
        np.testing.assert_array_equal(one.mean, two.mean)
        np.testing.assert_array_equal(one.lower, two.lower)
        assert one.variance.min() > 0
        if one.prediction_type == "variational_posterior":
            alpha = (1 - one.level) / 2
            np.testing.assert_array_equal(
                [one.lower, one.upper],
                np.quantile(one.draws, [alpha, 1 - alpha], axis=0),
            )
            np.testing.assert_allclose(one.draws.mean(axis=0), one.mean, atol=0.025)
            np.testing.assert_allclose(
                np.cov(one.draws, rowvar=False), one.covariance, atol=0.035
            )
        else:
            cov = model.covariance()
            oo, hh = np.ix_(~mask, ~mask), np.ix_(mask, ~mask)
            expected = model.training_mean_[mask] + cov[hh] @ np.linalg.solve(
                cov[oo], row[~mask] - model.training_mean_[~mask]
            )
            np.testing.assert_allclose(one.mean, expected)
    np.testing.assert_allclose(model.transform(x[:55]), model.factor_scores_, atol=2e-4)


def test_fixed_bagging_shared_residuals_joint_affine_and_chunks():
    x = data(n=28)
    blocks = [np.arange(4), np.arange(4, 9), np.arange(9, 12)]
    cfg = BaggingConfig(
        base_config=settings(),
        n_bootstrap=3,
        posterior_draws_per_bootstrap=21,
        summary_chunk_size=2,
        random_state=15,
    )
    model = BaggedCFAFit(cfg).fit(x, blocks)
    assert model.metadata_["bootstrap_size"] == 56
    samples = model.sample_posterior(300, random_state=3)
    for block in blocks:
        np.testing.assert_array_equal(
            samples["psi"][:, block],
            np.repeat(samples["psi"][:, block[:1]], len(block), axis=1),
        )
        np.testing.assert_array_equal(
            model.posterior("psi").lower[block], model.posterior("psi").lower[block[0]]
        )
    for name in ("Lambda", "gamma"):
        assert not model.posterior(name).applicable.any()
        np.testing.assert_array_equal(model.posterior(name).sd, 0)
    joint = model.joint_covariance(draws=300, random_state=11)
    assert len(joint.free_labels) == 6 + 3  # Phi upper triangle and three v_h
    assert joint.raw.shape == (36 + 6 + 12,) * 2
    for matrix in (joint.raw, joint.adjusted, joint.stabilized):
        np.testing.assert_array_equal(matrix[:36], 0)
        for block in blocks:
            np.testing.assert_allclose(
                matrix[42 + block],
                np.repeat(matrix[[42 + block[0]]], len(block), axis=0),
                atol=1e-13,
            )
    empty = model.joint_covariance(parameters=("Lambda",), draws=10)
    assert empty.free_raw.shape == (0, 0)
    np.testing.assert_array_equal(empty.raw, 0)
    old_quantiles = model.posterior("psi").lower.copy()
    model.config = replace(cfg, summary_chunk_size=20)
    np.testing.assert_array_equal(old_quantiles, model._summarize()["psi"].lower)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_prior_shape": 0},
        {"residual_prior_scale": np.nan},
        {"factor_covariance_prior_strength": 0},
        {"factor_prior_variance": -1},
        {"tolerance": np.nan},
        {"credible_interval_level": 1},
        {"elbo_decrease_tolerance": -1},
    ],
)
def test_fixed_invalid_settings_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        CFAFit(settings(**kwargs)).fit(data(), [np.arange(4)])


def test_fixed_rejects_small_or_overlapping_blocks():
    for blocks in [[np.arange(2)], [np.arange(4), np.arange(3, 7)]]:
        with pytest.raises(ValueError):
            CFAFit(settings()).fit(data(), blocks)
