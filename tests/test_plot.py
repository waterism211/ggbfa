"""The plot functions draw what the fitted objects contain."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="the plot functions need matplotlib (ggbfa[plot])")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ggbfa  # noqa: E402
from ggbfa import plot  # noqa: E402
from _sim import generate, make_spec  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    x = generate(make_spec("S3"), 1000, 5)
    res = ggbfa.efa(x, device="cpu", random_state=1)
    return x, res, ggbfa.cfa(x, res, member="2f+s", device="cpu")


def test_lazy_import():
    import ggbfa as package
    assert package.plot is plot


def test_search(fitted):
    _, res, _ = fitted
    ax = plot.plot_search(res)
    bars = ax.patches
    assert len(bars) == len(res.trace) - 1
    colors = [tuple(np.round(b.get_facecolor()[:3], 3)) for b in bars]
    assert colors[res.n_factors - 1] == (0.145, 0.145, 0.145)
    assert any(f"K={res.stop_k}" in t.get_text() for t in ax.texts)
    plt.close("all")


def test_correlation(fitted):
    x, res, fit = fitted
    ax = plot.plot_correlation(res, x, fit)
    image = ax.images[0].get_array()
    order = res.order
    z = (x - x.mean(0)) / x.std(0)
    sample = (z.T @ z / len(z))[np.ix_(order, order)]
    np.testing.assert_allclose(np.triu(image, 1), np.triu(sample, 1), atol=1e-12)
    cov = fit.covariance()
    d = 1 / np.sqrt(np.diag(cov))
    fitted_corr = (d[:, None] * cov * d[None, :])[np.ix_(order, order)]
    np.testing.assert_allclose(np.tril(image, -1), np.tril(fitted_corr, -1), atol=1e-12)
    plt.close("all")


def test_factor_covariance_labels(fitted):
    _, _, fit = fitted
    fig = plot.plot_factor_covariance(fit, draws=500, random_state=2)
    k = fit.factor_covariance_.shape[0]
    cells = [ax for ax in fig.axes if ax.texts]
    assert len(cells) == k * k
    phi = fit.sample_posterior(500, random_state=2, parameters=("Phi",))["Phi"]
    lower, upper = np.quantile(phi, [.025, .975], axis=0)
    for index, ax in enumerate(cells):
        a, b = divmod(index, k)
        text = ax.texts[0].get_text()
        assert text.startswith(f"{fit.factor_covariance_[a, b]:.2f}")
        if a != b:
            assert text.endswith("*") == bool(lower[a, b] > 0 or upper[a, b] < 0)
    plt.close("all")


def test_network_edges(fitted):
    _, _, fit = fitted
    threshold = .15
    ax = plot.plot_factor_network(fit, threshold=threshold)
    mean = fit.factor_correlation_
    lower, upper = fit.factor_correlation_ci_
    k = len(mean)
    expected = sum(abs(mean[a, b]) >= threshold and (lower[a, b] > 0 or upper[a, b] < 0)
                   for a in range(k) for b in range(a + 1, k))
    assert len(ax.lines) == expected
    plt.close("all")


def test_loadings(fitted):
    _, _, fit = fitted
    ax = plot.plot_loadings(fit)
    np.testing.assert_array_equal(ax.images[0].get_array(), fit.loadings_)
    plt.close("all")
