"""loading_normalization="free" for the four confirmatory members."""
import numpy as np
import pytest

from ggbfa import BaggedCFAFit, CFAFit, CFAConfig, MEMBERS
from ggbfa._bagging import BaggingConfig
from ggbfa._joint import coordinate_layout
from ggbfa._model import _draw_loading_posterior
from ggbfa._posterior import PARAMETERS
from _sim import correlation, generate, make_spec, relative_errors

LEARNED = ("2f+s", "2f", "mf")


def standardized(spec, n, seed):
    x = generate(spec, n, seed)
    return (x - x.mean(axis=0)) / x.std(axis=0)


def fit(option, normalization, x, blocks, **settings):
    config = CFAConfig(member=option, loading_normalization=normalization, device="cpu", **settings)
    return CFAFit(config).fit(x, blocks)


@pytest.fixture(scope="module")
def s3_data():
    spec = make_spec("S3")
    return spec, standardized(spec, 2000, 11), [list(block) for block in spec.blocks]


def test_default_and_validation():
    assert CFAConfig().loading_normalization == "row_sum"
    with pytest.raises(ValueError):
        CFAConfig(loading_normalization="unit")
    with pytest.raises(ValueError):
        CFAConfig(loading_normalization="free", primary_loading_prior_variance=0.0)


@pytest.mark.parametrize("option", MEMBERS)
def test_free_fit_is_monotone_and_labelled(option, s3_data):
    spec, x, blocks = s3_data
    est = fit(option, "free", x[:400], blocks, primary_loading_prior_variance=2.0)
    post, diag = est.result_.posterior, est.result_.diagnostics
    assert not post.enforce_loading_row_sums and not diag.enforce_loading_row_sums
    assert diag.primary_loading_prior_mean == 1.0 and diag.primary_loading_prior_variance == 2.0
    history = np.asarray(diag.elbo_history)
    assert diag.converged and np.all(np.diff(history) >= -1e-8 * (1 + np.abs(history[:-1])))
    assert "free" in est.method_label_
    rows = np.arange(len(post.loading_mean))
    primary = np.argmax(post.loading_mask, axis=1)
    assert np.all(post.loading_sd[rows, primary] > 0)  # primary loadings are estimated, not fixed
    covariance = est.covariance()
    assert np.all(np.isfinite(covariance)) and np.linalg.eigvalsh(covariance).min() > 0
    draws = _draw_loading_posterior(est.result_, draws=4000, rng=np.random.default_rng(1))
    assert np.abs(draws.mean(axis=0) - post.loading_mean).max() < 0.05
    samples = est.sample_posterior(draws=20, random_state=3)
    assert samples["Lambda"].shape == (20,) + post.loading_mean.shape


@pytest.mark.parametrize("option", LEARNED)
def test_free_recovers_unequal_primary_loadings(option, s3_data):
    spec, x, blocks = s3_data
    truth = correlation(spec)
    errors = {norm: relative_errors(fit(option, norm, x, blocks).covariance(), truth)["whitened_frobenius"]
              for norm in ("row_sum", "free")}
    assert errors["free"] < 0.5 * errors["row_sum"]


@pytest.mark.parametrize("option", LEARNED)
def test_free_agrees_with_row_sum_when_rows_sum_to_one(option):
    spec = make_spec("S1_fixed")
    x, blocks = standardized(spec, 2000, 12), [list(block) for block in spec.blocks]
    truth = correlation(spec)
    errors = {norm: relative_errors(fit(option, norm, x, blocks).covariance(), truth)["whitened_frobenius"]
              for norm in ("row_sum", "free")}
    assert errors["free"] < errors["row_sum"] + 0.15


def test_joint_coordinates_reject_free(s3_data):
    spec, x, blocks = s3_data
    est = fit("mf", "free", x[:300], blocks)
    with pytest.raises(NotImplementedError):
        coordinate_layout(est.result_.posterior, PARAMETERS)


@pytest.mark.parametrize("option", ("1f", "2f"))
def test_bagging_with_free_loadings(option, s3_data):
    spec, x, blocks = s3_data
    bag = BaggedCFAFit(BaggingConfig(
        base_config=CFAConfig(member=option, loading_normalization="free", device="cpu"),
        n_bootstrap=3, posterior_draws_per_bootstrap=12, random_state=4)).fit(x[:300], blocks)
    summary = bag.posterior_["Lambda"]
    primary = np.argmax(np.eye(len(blocks))[np.concatenate([[h] * len(b) for h, b in enumerate(blocks)])], axis=1)
    assert np.all(summary.sd[np.arange(len(primary)), primary] > 0)
