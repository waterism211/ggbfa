"""Figures of the gg-BFA paper as functions (needs matplotlib: ``pip install ggbfa[plot]``).

=========================  ==================================================
``plot_search``            early stopping of the exploratory step (Fig. 1c)
``plot_correlation``       sample and fitted correlation matrices (Figs. 1b, 1d, 1g, 3)
``plot_loadings``          posterior mean loadings with the cross-loadings found
``plot_factor_covariance`` marginal posteriors of the factor covariance (Fig. 4c)
``plot_factor_network``    factor correlations whose interval excludes zero (Fig. 4b)
=========================  ==================================================

``plot_search`` takes the result of :func:`ggbfa.efa`; ``plot_correlation`` takes
it and, optionally, a fit of :func:`ggbfa.cfa`; the other functions take a fit.
Every function draws into ``ax`` (or a new figure) and returns the Axes, except
``plot_factor_covariance``, which returns the Figure.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from scipy import stats

GRID = 400                      # matrices are block-averaged to at most GRID x GRID cells for drawing
CORRELATION_NORM = Normalize(-1, 1)
POSITIVE, NEGATIVE = "#d62728", "#1f77b4"


def _axes(ax, size):
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def _labels(labels, k):
    if labels is None:
        return [f"F{h + 1}" for h in range(k)]
    if len(labels) != k:
        raise ValueError(f"labels must have one entry per factor ({k})")
    return list(labels)


def _factors(factors, k):
    ids = np.arange(k) if factors is None else np.asarray(factors, dtype=int)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= k:
        raise ValueError(f"factors must be distinct indices in 0..{k - 1}")
    return ids


def _shrink(matrix, grid=GRID):
    """Block-average a square matrix to at most grid x grid cells (drawing only)."""
    p = len(matrix)
    if p <= grid:
        return matrix
    edges = np.linspace(0, p, grid + 1).astype(int)
    rows = np.add.reduceat(matrix, edges[:-1], axis=0) / np.diff(edges)[:, None]
    return np.add.reduceat(rows, edges[:-1], axis=1) / np.diff(edges)[None, :]


def _ink(rgba):
    """White text on dark backgrounds, black on light ones."""
    red, green, blue = rgba[:3]
    return "white" if .2126 * red + .7152 * green + .0722 * blue < .48 else "black"


def plot_search(result, ax=None, *, title: Optional[str] = "Early stopping"):
    """ELBO gain of each visited K over K - 1 (symlog scale), as in Fig. 1(c).

    The bar of the selected K is black; a decrease, where the search stopped,
    is red; the other visited K are grey.
    """

    ax = _axes(ax, (3.2, 1.6))
    ks = np.array([row["k"] for row in result.trace])
    if len(ks) < 2:
        raise ValueError("the search visited only K = 0")
    gain = np.diff([row["elbo"] for row in result.trace])
    selected = result.n_factors
    colors = [POSITIVE if d < 0 else "#252525" if k == selected else "#bdbdbd" for k, d in zip(ks[1:], gain)]
    ax.bar(ks[1:], gain, color=colors, width=.8, edgecolor="none")
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(0, color="black", lw=.4)
    top = float(max(gain.max(), 1.0))
    ax.set_ylim(min(float(gain.min()) * 2, -2.0), top * 3)
    ax.text(selected, max(gain[ks[1:] == selected][0] if selected > 0 else 0, 0) * 1.5 + 1, rf"$\widehat K={selected}$",
            ha="center", va="bottom", fontsize=8)
    if result.stop_k is not None:
        ax.text(result.stop_k - .7, -.75, f"stop at $K={result.stop_k}$", ha="right", va="center", fontsize=7,
                color=POSITIVE)
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$\Delta\,\mathrm{ELBO}$")
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def _correlation_of_covariance(covariance):
    scale = 1 / np.sqrt(np.diag(covariance))
    return scale[:, None] * covariance * scale[None, :]


def plot_correlation(
    result,
    data,
    fit=None,
    *,
    order: str = "community",
    ax=None,
    grid: int = GRID,
    colorbar: bool = True,
    title: Optional[str] = None,
):
    """Sample correlation of ``data``, as in Figs. 1(b), 1(d) and 3.

    ``order="community"`` sorts the variables by the communities of ``result``
    (background last) and draws their boundaries; ``"original"`` keeps the
    column order.  With ``fit`` (a :func:`ggbfa.cfa` fit), the triangle below
    the diagonal shows the correlation fitted by gg-BFA (CFA) and the one
    above it the sample correlation (EFA), as in the bottom row of Fig. 3.
    Matrices with more than ``grid`` variables are block-averaged for drawing.
    """

    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(result.data_mean):
        raise ValueError("data must have the columns that were passed to efa")
    if order not in ("community", "original"):
        raise ValueError("order must be 'community' or 'original'")
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    sample = z.T @ z / len(z)
    sample = (sample + sample.T) / 2
    ids = result.order if order == "community" else np.arange(x.shape[1])
    shown = _shrink(sample[np.ix_(ids, ids)], grid)
    if fit is not None:
        if fit.n_features_in_ != x.shape[1]:
            raise ValueError("fit and data have different numbers of variables")
        fitted = _shrink(_correlation_of_covariance(fit.covariance())[np.ix_(ids, ids)], grid)
        shown = np.triu(shown, 1) + np.tril(fitted, -1) + np.diag(np.diag(shown))
    ax = _axes(ax, (3.2, 3.0))
    image = ax.imshow(shown, cmap="jet", norm=CORRELATION_NORM, interpolation="nearest", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    if order == "community":
        cuts = np.cumsum([len(c) for c in result.communities]) * len(shown) / len(ids)
        for cut in cuts:
            ax.axhline(cut - .5, color="white", lw=.3)
            ax.axvline(cut - .5, color="white", lw=.3)
    if fit is not None:
        box = dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.85)
        ax.text(.96, .96, "EFA", transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=box)
        ax.text(.04, .04, "CFA", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, bbox=box)
    if colorbar:
        plt.colorbar(image, ax=ax, ticks=[-1, -.5, 0, .5, 1], fraction=.046, pad=.03)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_loadings(fit, ax=None, *, pip_threshold: float = 0.5, labels: Optional[Sequence[str]] = None,
                  title: Optional[str] = None):
    """Posterior mean loadings of the variables in the communities.

    Rows are the variables community by community, columns the factors.
    Cross-loadings with posterior inclusion probability at least
    ``pip_threshold`` are marked with a dot.
    """

    loading = np.asarray(fit.loadings_, dtype=float)
    pip = np.asarray(fit.cross_loading_pip_, dtype=float)
    p, k = loading.shape
    names = _labels(labels, k)
    limit = float(np.max(np.abs(loading)))
    ax = _axes(ax, (1.2 + .25 * k, 3.2))
    image = ax.imshow(loading, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest",
                      rasterized=True)
    rows, columns = np.nonzero(pip >= pip_threshold)
    if len(rows):
        ax.scatter(columns, rows, s=6, color="black", marker="o", linewidths=0,
                   label=f"cross-loading, PIP $\\geq$ {pip_threshold:g}")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.06), fontsize=7, frameon=False)
    for cut in np.cumsum(fit.result_.posterior.block_sizes)[:-1]:
        ax.axhline(cut - .5, color="#555555", lw=.4)
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, fontsize=7, rotation=90 if k > 12 else 0)
    ax.set_yticks([])
    ax.set_ylabel(f"variables in communities ({p})")
    plt.colorbar(image, ax=ax, fraction=.046, pad=.03)
    ax.set_title(title or fit.method_label_, fontsize=9)
    return ax


def _phi_cell(ax, values, mean, level, diagonal, norm):
    face = plt.get_cmap("jet")(norm(mean))
    color = _ink(face)
    ax.set_facecolor(face)
    low_support, high_support = np.quantile(values, [.005, .995])
    grid = np.linspace(low_support, high_support, 120)
    density = stats.gaussian_kde(values)(grid)
    density = .48 * density / density.max() + .2
    lower, upper = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    left, right = min(low_support, lower, 0.), max(high_support, upper, 0.)
    pad = .045 * (right - left)
    ax.fill_between(grid, .2, density, color=color, alpha=.28, lw=0)
    ax.plot(grid, density, color=color, lw=.55)
    ax.hlines(.1, lower, upper, color=color, lw=1.2)
    ax.plot(mean, .1, "o", color=color, ms=1.8)
    ax.vlines(0, 0, .7, color=color, lw=.5, ls=(0, (2, 2)), alpha=.75)
    ax.vlines(mean, 0, .7, color=color, lw=.5, alpha=.8)
    label = f"{mean:.2f}"
    if not diagonal:
        label += "*" if (lower > 0 or upper < 0) else " ns"
    ax.text(.5, .97, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.3, color=color, weight="bold")
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.35)
        spine.set_color("#333333")
    return lower, upper


def plot_factor_covariance(
    fit,
    *,
    factors: Optional[Sequence[int]] = None,
    scale: str = "covariance",
    level: float = 0.95,
    draws: int = 2000,
    random_state: int = 1,
    labels: Optional[Sequence[str]] = None,
    cell_size: float = .75,
    title: Optional[str] = "default",
):
    """Marginal posteriors of the factor covariance, one cell per entry (Fig. 4c).

    Each cell shows a kernel density of ``draws`` posterior draws, the
    posterior mean (number and dot), the equal-tailed ``level`` credible
    interval (bar) and zero (dashed line).  Off-diagonal entries are marked
    ``*`` when the interval excludes zero and ``ns`` otherwise.  The cell
    color follows the posterior mean on the jet scale from -1 to 1.
    ``scale="correlation"`` shows the factor correlations instead.
    ``factors`` selects a subset, for example ``range(8)``.
    """

    if scale not in ("covariance", "correlation"):
        raise ValueError("scale must be 'covariance' or 'correlation'")
    if not 0 < level < 1:
        raise ValueError("level must lie in (0, 1)")
    phi = np.asarray(fit.sample_posterior(draws, random_state=random_state, parameters=("Phi",))["Phi"], dtype=float)
    k = phi.shape[1]
    ids = _factors(factors, k)
    names = _labels(labels, k)
    if scale == "correlation":
        sd = np.sqrt(np.diagonal(phi, axis1=1, axis2=2))
        phi = phi / (sd[:, :, None] * sd[:, None, :])
        mean = phi.mean(axis=0)
    else:
        mean = np.asarray(fit.factor_covariance_, dtype=float)
    m = len(ids)
    fig = plt.figure(figsize=(.5 + cell_size * m, .45 + cell_size * .62 * m))
    cells = fig.add_gridspec(m, m, left=.4 / (.5 + cell_size * m), right=.98, bottom=.02,
                             top=1 - .4 / (.45 + cell_size * .62 * m), hspace=.1, wspace=.06)
    for r, a in enumerate(ids):
        for c, b in enumerate(ids):
            ax = fig.add_subplot(cells[r, c])
            _phi_cell(ax, phi[:, a, b], float(mean[a, b]), level, a == b, CORRELATION_NORM)
            if r == 0:
                ax.set_title(names[b], fontsize=7.5, pad=2)
            if c == 0:
                ax.set_ylabel(names[a], fontsize=7.5, rotation=0, ha="right", va="center", labelpad=3)
    if title == "default":
        title = ("Posterior distributions of the factor " + ("covariance $q(\\Phi)$" if scale == "covariance"
                                                               else "correlations"))
    if title:
        fig.suptitle(title, fontsize=9)
    return fig


def plot_factor_network(
    fit,
    *,
    threshold: float = 0.5,
    factors: Optional[Sequence[int]] = None,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    title: Optional[str] = "default",
):
    """Network of the factor correlations, as in Fig. 4(b).

    Nodes are factors, sized by the number of variables in their community and
    placed on a circle; edges join factors whose posterior mean correlation is
    at least ``threshold`` in absolute value and whose credible interval
    excludes zero, red for positive and blue for negative, wider for stronger.
    """

    mean = np.asarray(fit.factor_correlation_, dtype=float)
    lower, upper = (np.asarray(v, dtype=float) for v in fit.factor_correlation_ci_)
    k = len(mean)
    ids = _factors(factors, k)
    names = _labels(labels, k)
    sizes = np.asarray(fit.result_.posterior.block_sizes, dtype=float) if hasattr(fit, "result_") else \
        np.asarray([len(c) for c in fit.communities_], dtype=float)
    m = len(ids)
    angle = np.pi / 2 - 2 * np.pi * np.arange(m) / m
    xy = np.c_[np.cos(angle), np.sin(angle)]
    ax = _axes(ax, (3.0, 3.0))
    for i in range(m):
        for j in range(i + 1, m):
            a, b = ids[i], ids[j]
            r = mean[a, b]
            if abs(r) >= threshold and (lower[a, b] > 0 or upper[a, b] < 0):
                ax.plot(*xy[[i, j]].T, color=POSITIVE if r > 0 else NEGATIVE, lw=.5 + 3 * abs(r), alpha=.75,
                        zorder=1, solid_capstyle="round")
    shown = sizes[ids]
    ax.scatter(*xy.T, s=40 + 200 * shown / shown.max(), color="#fdd49e", edgecolor="#333333", linewidth=.4, zorder=2)
    for i, a in enumerate(ids):
        ax.text(*(xy[i] * 1.24), names[a], fontsize=7.5, ha="center", va="center")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], color=POSITIVE, lw=1.8, label="positive"),
                       Line2D([], [], color=NEGATIVE, lw=1.8, label="negative")],
              loc="lower right", fontsize=6.5, frameon=False, handlelength=1.4, borderaxespad=0)
    if title == "default":
        title = rf"Factor correlations, $|r|\geq{threshold:g}$"
    if title:
        ax.set_title(title, fontsize=9)
    return ax


__all__ = ["plot_correlation", "plot_factor_covariance", "plot_factor_network", "plot_loadings", "plot_search"]
