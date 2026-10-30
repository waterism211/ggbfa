"""Dense-subgraph search of the exploratory step: a NumPy port of ICONS.

Ported from the R package ICONS 0.2.0 by Yezhi Pan (https://github.com/xavienzo/ICONS,
commit ba204a7, MIT license; see LICENSES/ICONS.txt): ``greedy peeling`` with l0
shrinkage and the adaptive loop of ``icons_detect`` (src/icons.cpp, R/detect.R).

The port reproduces the compiled routine operation by operation, so that it returns
the same communities: weighted degrees are summed in ascending column order, the
peeled node is the one with the smallest degree (ties to the smaller index), degrees
are updated in removal order, and the threshold is the type-7 sample quantile of the
upper-triangular entries computed with R's formula.  The thresholded matrix is held
densely; one peel of m nodes costs O(m^2) vectorized operations.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def r_quantile(values: np.ndarray, prob: float) -> float:
    """Type-7 sample quantile, evaluated with the arithmetic of R's ``quantile``."""

    x = np.sort(np.asarray(values, dtype=float).ravel())
    n = x.size
    if n == 0:
        raise ValueError("no values to take a quantile of")
    index = 1.0 + max(n - 1, 0) * float(prob)
    lo, hi = math.floor(index), math.ceil(index)
    q = float(x[lo - 1])
    if index > lo and x[hi - 1] != q:
        h = index - lo
        q = (1.0 - h) * q + h * float(x[hi - 1])
    return q


def _thresholded(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Keep entries >= threshold (and nonzero); drop the diagonal."""

    kept = np.where((weights >= threshold) & (weights != 0.0), weights, 0.0)
    np.fill_diagonal(kept, 0.0)
    return kept


def _peel(graph: np.ndarray, active: Sequence[int], lam: float) -> Tuple[float, List[int], List[int]]:
    """One greedy peeling pass restricted to ``active``; returns (score, keep, drop).

    ``keep`` holds the best-scoring set in reverse removal order (densest core
    first) and ``drop`` the nodes peeled away before it.  The score is NaN when
    ``active`` has fewer than two nodes.
    """

    m = len(active)
    if m == 0:
        return math.nan, [], []
    if m == 1:
        return math.nan, [int(active[0])], []
    nodes = np.sort(np.asarray(active, dtype=int))
    sub = graph[np.ix_(nodes, nodes)]
    # Sequential sums in ascending column order, as in the compiled loop.
    degree = np.cumsum(sub, axis=1)[:, -1]
    position = np.searchsorted(nodes, np.asarray(active, dtype=int))
    total = 0.0
    for i in position:
        total += float(degree[i])
    total *= 0.5

    current = degree.copy()
    order: List[int] = []
    best, best_cut = -math.inf, -1
    exponent = 2.0 * lam
    for step in range(m):
        i = int(np.argmin(current))  # first minimum: the smaller node index wins ties
        removed = float(current[i])
        order.append(int(nodes[i]))
        total -= removed
        current -= sub[i]  # exact zero for non-neighbours
        current[i] = math.inf
        remaining = m - step - 1
        if remaining >= 1:
            score = total / math.pow(float(remaining), exponent)
            if score > best:  # strict: the first maximum wins
                best, best_cut = score, step + 1
    if best_cut < 0:  # every score was degenerate: keep the whole set
        return math.nan, order[::-1], []
    return best, order[best_cut:][::-1], order[:best_cut]


def detect(weights: np.ndarray, threshold: float, lam: float, min_size: int = 2, max_k: int = 0) -> dict:
    """Adaptive dense-subgraph extraction (``icons_detect`` with a given threshold).

    Returns the communities in extraction order (``blocks``, each in the order
    of the peeling, densest core first), their scores, and the remaining
    variables in ascending order (``background``).
    """

    graph = _thresholded(np.asarray(weights, dtype=float), float(threshold))
    p = graph.shape[0]
    active: List[int] = list(range(p))
    blocks: List[List[int]] = []
    scores: List[float] = []
    while len(active) >= min_size:
        if max_k > 0 and len(blocks) >= max_k:
            break
        score, keep, drop = _peel(graph, active, lam)
        if len(keep) < min_size:
            break
        blocks.append(keep)
        scores.append(score)
        if not drop:
            active = []
            break
        active = drop
    taken = np.zeros(p, dtype=bool)
    for block in blocks:
        taken[block] = True
    return dict(blocks=blocks, scores=scores, background=np.flatnonzero(~taken).tolist())


def icons_paths(
    correlation: np.ndarray,
    percentiles: Sequence[float],
    lambdas: Sequence[float],
    min_size: int = 2,
) -> List[dict]:
    """Nested community paths over the grid of (percentile, lambda) pairs.

    For each percentile ``q`` the edge threshold is the ``q/100`` sample
    quantile of the upper-triangular correlations; each pair gives one path,
    whose first K communities form the candidate partition with K communities.
    Paths are listed with the percentile in the outer loop.
    """

    w = np.asarray(correlation, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("correlation must be a square matrix with at least two rows")
    if not np.all(np.isfinite(w)):
        raise ValueError("correlation must be finite")
    if np.max(np.abs(w - w.T)) > 1e-8 * max(1.0, float(np.max(np.abs(w)))):
        raise ValueError("correlation must be symmetric")
    if isinstance(min_size, bool) or int(min_size) != min_size or min_size < 2:
        raise ValueError("min_size must be an integer >= 2")
    w = (w + w.T) / 2
    upper = w[np.triu_indices(w.shape[0], k=1)]
    paths = []
    for q in percentiles:
        if not 0 <= q <= 100:
            raise ValueError("percentiles must lie in [0, 100]")
        threshold = r_quantile(upper, q / 100)
        for lam in lambdas:
            if not 0 <= lam <= 1:
                raise ValueError("lambdas must lie in [0, 1]")
            found = detect(w, threshold, float(lam), int(min_size))
            paths.append({"q": q, "lambda": float(lam), "threshold": threshold, "blocks": found["blocks"],
                          "K": len(found["blocks"]), "background": found["background"]})
    return paths


__all__ = ["detect", "icons_paths", "r_quantile"]
