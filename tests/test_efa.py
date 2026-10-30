"""The exploratory step: communities, the search record, and the hand-off to cfa."""
import numpy as np
import pytest

import ggbfa
from _sim import generate, make_spec


@pytest.fixture(scope="module")
def s1():
    spec = make_spec("S1_fixed")
    return spec, generate(spec, 1000, 11)


@pytest.fixture(scope="module")
def result(s1):
    return ggbfa.efa(s1[1], device="cpu", random_state=3)


def test_two_stage_recovers_the_communities(s1, result):
    spec, x = s1
    assert result.search == "two_stage" and result.status == "first-increase"
    assert result.n_factors == 5 and result.stop_k == 6
    assert sorted(map(sorted, (c.tolist() for c in result.communities))) == sorted(map(list, spec.blocks))
    assert len(result.background) == 200 - sum(len(b) for b in spec.blocks)
    assert 0 < result.alpha < 1
    assert result.stage1["k_ref"] >= 1 and len(result.paths) == 50
    ks = [row["k"] for row in result.trace]
    assert ks == list(range(ks[0], ks[-1] + 1)) and ks[0] == 0
    elbo = [row["elbo"] for row in result.trace]
    assert int(np.argmax(elbo)) == 5
    assert np.array_equal(np.sort(result.order), np.arange(200))
    labels = result.membership()
    assert labels.max() == 5 and (labels == 0).sum() == len(result.background)
    assert result.covariance.shape == (200, 200) and np.linalg.eigvalsh(result.covariance).min() > 0
    assert result.runtime["resolved_device"] == "cpu"


def test_reproducible(s1, result):
    again = ggbfa.efa(s1[1], device="cpu", random_state=3)
    assert [r["elbo"] for r in again.trace] == [r["elbo"] for r in result.trace]


def test_cfa_takes_the_result(s1, result):
    fit = ggbfa.cfa(s1[1], result, member="2f", device="cpu")
    assert len(fit.communities_) == 5
    assert fit.covariance().shape == (200, 200)


def test_full_search_on_a_small_problem():
    spec = make_spec("S1_fixed")
    x = generate(spec, 400, 4)
    res = ggbfa.efa(x, search="full", percentiles=(95, 97), lambdas=(.5, .7), device="cpu", random_state=1)
    assert res.search == "full" and res.stage1 is None
    assert all("candidates" in row for row in res.trace)
    assert 1 <= res.n_factors <= 6


def test_max_k_caps_the_search(s1):
    res = ggbfa.efa(s1[1], device="cpu", random_state=3, max_k=3)
    assert res.status == "cap-reaching" and max(r["k"] for r in res.trace) <= 3


@pytest.mark.parametrize("kwargs", [dict(search="joint"), dict(random_state=-1), dict(max_k=0),
                                    dict(prior={"a0": 2.0})])
def test_invalid_settings(kwargs, s1):
    with pytest.raises(ValueError):
        ggbfa.efa(s1[1][:50], device="cpu", **kwargs)


def test_constant_column_rejected():
    x = np.random.default_rng(1).normal(size=(20, 5))
    x[:, 2] = 1.0
    with pytest.raises(ValueError, match="constant"):
        ggbfa.efa(x, device="cpu")
