"""Selection model of the exploratory step and the early-stopped searches.

The selection model gives each community one factor with unit loadings and a
shared residual variance, and puts the remaining variables in one background
block.  ``fit_marginal`` maximizes the alpha-fractional ELBO over inverse-Wishart
q(Phi) and inverse-gamma q(v) with the factor scores integrated out; the
expectation of the community likelihood is evaluated by Monte Carlo.  Scores are
variational free energies (negative ELBOs): smaller is better.

``search_joint`` is Algorithm 1 of the paper (every distinct prefix of every
path at each K); ``search_path`` is stage 2 of Algorithm 2 (one path).  Both
stop at the first statistically significant increase of the free energy.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, multigammaln

from ._backend import resolve_compute_backend


@dataclass
class BlockStats:
    n: int
    p: int
    blocks: tuple
    sizes: np.ndarray
    mean_cov: np.ndarray
    contrasts: np.ndarray
    background: np.ndarray
    background_ss: float


def block_stats(x, blocks):
    x = np.asarray(x, dtype=float)
    blocks = tuple(np.asarray(b, dtype=int) for b in blocks)
    used = np.concatenate(blocks) if blocks else np.empty(0, dtype=int)
    if len(np.unique(used)) != len(used) or np.any(used < 0) or np.any(used >= x.shape[1]):
        raise ValueError("invalid or overlapping community indices")
    if any(len(b) < 2 for b in blocks) or not np.isfinite(x).all():
        raise ValueError("need finite data and communities of size >=2")
    n, p = x.shape
    sizes = np.array([len(b) for b in blocks], dtype=float)
    y = np.column_stack([x[:, b].sum(axis=1) / np.sqrt(len(b)) for b in blocks]) if blocks else np.empty((n, 0))
    sy = y.T @ y / n
    ss = np.sum(x * x, axis=0) / n
    contrasts = np.array([ss[b].sum() - sy[h, h] for h, b in enumerate(blocks)])
    if np.any(contrasts < -1e-8):
        raise ArithmeticError("negative within-community sum of squares")
    background = np.setdiff1d(np.arange(p), used)
    return BlockStats(n, p, blocks, sizes, sy, np.maximum(contrasts, 0), background, float(ss[background].sum()))


def background_vfe(n, size, ss, alpha, a0=2.0, b0=0.5):
    """Exact -log integral of the powered Gaussian likelihood; ss is /n."""
    a = a0 + alpha * n * size / 2
    b = b0 + alpha * n * ss / 2
    value = (alpha * n * size * np.log(2 * np.pi) / 2
             - a0 * np.log(b0) + gammaln(a0) - gammaln(a) + a * np.log(b))
    return float(value), float(a), float(b)


class MarginalObjective:
    def __init__(self, stats, alpha, *, a0=2.0, b0=0.5, c=0.5, kappa=1.0,
                 draws=256, seed=1, device="cpu", backend=None):
        if not 0 < alpha <= 1 or min(a0 - 1, b0, c, kappa) <= 0:
            raise ValueError("require alpha in (0,1], a0>1 and positive proper prior scales")
        self.stats, self.alpha = stats, float(alpha)
        self.k = len(stats.blocks)
        if self.k < 1:
            raise ValueError("K=0 is evaluated exactly, without optimization")
        self.a0, self.b0, self.c = a0, b0, c
        self.nu0 = self.k + 1 + kappa
        self.nu = self.nu0 + alpha * stats.n
        self.a = a0 + alpha * stats.n * stats.sizes / 2
        self.backend = backend or resolve_compute_backend(device)
        self.xp = self.backend._cupy if self.backend.is_gpu else np
        self.ix = np.tril_indices(self.k)
        self.diag_mask = self.ix[0] == self.ix[1]
        xp = self.xp
        self.sizes = xp.asarray(stats.sizes)
        self.sb = xp.sqrt(self.sizes)
        self.sy = xp.asarray(stats.mean_cov)
        self.contrasts = xp.asarray(stats.contrasts)
        self.a_device = xp.asarray(self.a)
        self.digamma_a = xp.asarray(digamma(self.a))
        self.psi_sum = float(sum(digamma((self.nu - np.arange(self.k)) / 2)))
        self.bg, self.bg_a, self.bg_b = background_vfe(stats.n, len(stats.background), stats.background_ss, alpha, a0, b0)
        self.fixed_constant = self.bg + alpha * stats.n * stats.sizes.sum() * np.log(2 * np.pi) / 2
        self.set_draws(draws, seed)

    def host(self, value):
        return self.xp.asnumpy(value) if self.backend.is_gpu else np.asarray(value)

    def set_draws(self, draws, seed):
        if draws < 2:
            raise ValueError("at least two Monte Carlo draws are needed")
        rng = np.random.default_rng(seed)
        b = np.zeros((draws, self.k, self.k))
        low = np.tril_indices(self.k, -1)
        b[:, low[0], low[1]] = rng.normal(size=(draws, len(low[0])))
        for j in range(self.k):
            b[:, j, j] = np.sqrt(rng.chisquare(self.nu - j, size=draws))
        inv = np.linalg.inv(b)
        self.r = self.xp.asarray(inv.transpose(0, 2, 1) @ inv)
        self.g = self.xp.asarray(rng.gamma(self.a, size=(draws, self.k)))
        self.draws, self.seed = int(draws), int(seed)

    def unpack(self, theta):
        xp = self.xp
        theta = xp.asarray(theta)
        entries = theta[:len(self.ix[0])].copy()
        entries[self.diag_mask] = xp.exp(entries[self.diag_mask])
        l = xp.zeros((self.k, self.k))
        l[self.ix] = entries
        beta = xp.exp(theta[len(self.ix[0]):])
        return l, beta

    def initial(self, alternative=False):
        s = self.stats
        v = np.maximum(s.contrasts / (s.sizes - 1), 0.02)
        phi = (s.mean_cov - np.diag(v)) / np.sqrt(s.sizes[:, None] * s.sizes[None, :])
        eig, vec = np.linalg.eigh(phi)
        phi = (vec * np.maximum(eig, 0.005)) @ vec.T
        if alternative:
            phi = 0.65 * phi + 0.15 * np.eye(self.k)
            v = 1.2 * v
        psi = phi * (self.nu - self.k - 1)
        l = np.linalg.cholesky(psi)
        entries = l[self.ix].copy()
        entries[self.diag_mask] = np.log(entries[self.diag_mask])
        return np.r_[entries, np.log((self.a - 1) * v)]

    def value_gradient(self, theta, *, gradient=True, samples=False):
        xp, k, alpha, n = self.xp, self.k, self.alpha, self.stats.n
        l, beta = self.unpack(theta)
        phi = l[None, :, :] @ self.r @ l.T[None, :, :]
        variances = beta[None, :] / self.g
        cov = self.sb[None, :, None] * phi * self.sb[None, None, :]
        cov[:, np.arange(k), np.arange(k)] += variances
        chol = xp.linalg.cholesky(cov)
        inv = xp.linalg.inv(cov)
        logdet = 2 * xp.log(chol[:, np.arange(k), np.arange(k)]).sum(axis=1)
        random_term = alpha * n / 2 * (logdet + xp.einsum("sij,ji->s", inv, self.sy))
        contrast = alpha * n / 2 * xp.sum(
            -(self.sizes - 1) * (self.digamma_a - xp.log(beta))
            + self.a_device / beta * self.contrasts)
        psi_inv = xp.linalg.inv(l @ l.T)
        logdet_psi = 2 * xp.log(xp.diag(l)).sum()
        expected_logdet = logdet_psi - k * np.log(2) - self.psi_sum
        kl_iw = (self.nu / 2 * logdet_psi - self.nu0 / 2 * k * np.log(self.c)
                 - (self.nu - self.nu0) * k / 2 * np.log(2)
                 - multigammaln(self.nu / 2, k) + multigammaln(self.nu0 / 2, k)
                 - (self.nu - self.nu0) / 2 * expected_logdet
                 - self.nu * k / 2 + self.c * self.nu / 2 * xp.trace(psi_inv))
        kl_gamma = xp.sum(xp.asarray((self.a - self.a0) * digamma(self.a)
                                    - gammaln(self.a) + gammaln(self.a0))
                          + self.a0 * (xp.log(beta) - np.log(self.b0))
                          + self.a_device * (self.b0 / beta - 1))
        value = self.fixed_constant + contrast + kl_iw + kl_gamma + random_term.mean()
        if self.backend.is_gpu:
            counts = self.backend.operation_counts
            counts["marginal_batched_cholesky"] = counts.get("marginal_batched_cholesky", 0) + 1
            counts["marginal_matrix_draws"] = counts.get("marginal_matrix_draws", 0) + self.draws
        if not np.isfinite(float(value)):
            raise FloatingPointError("nonfinite marginal VFE")
        if samples:
            return float(value), self.host(random_term)
        if not gradient:
            return float(value)
        h = alpha * n / 2 * (inv - inv @ self.sy @ inv)
        dphi = self.sb[None, :, None] * h * self.sb[None, None, :]
        dl = 2 * (dphi @ l[None, :, :] @ self.r).mean(axis=0)
        dl += self.nu0 * psi_inv @ l - self.c * self.nu * psi_inv @ psi_inv @ l
        db = (h[:, np.arange(k), np.arange(k)] * variances).mean(axis=0)
        db += alpha * n / 2 * (self.sizes - 1 - self.a_device / beta * self.contrasts)
        db += self.a0 - self.a_device * self.b0 / beta
        grad_l = dl[self.ix]
        grad_l[self.diag_mask] *= xp.diag(l)
        grad = self.host(xp.concatenate([grad_l, db]))
        if not np.isfinite(grad).all():
            raise FloatingPointError("nonfinite marginal gradient")
        return float(value), grad

    def posterior(self, theta):
        l, beta = self.unpack(theta)
        psi = self.host(l @ l.T)
        beta = self.host(beta)
        return {
            "phi_mean": psi / (self.nu - self.k - 1),
            "residual_mean": beta / (self.a - 1),
            "background_mean": self.bg_b / (self.bg_a - 1),
            "iw_df": self.nu, "iw_scale": psi,
            "gamma_shape": self.a.copy(), "gamma_rate": beta,
            "background_shape": self.bg_a, "background_rate": self.bg_b,
        }


def fit_marginal(stats, alpha, *, config, backend, seed):
    started = time.perf_counter()
    objective = MarginalObjective(stats, alpha, **config["prior"],
        draws=config["optimization_draws"], seed=seed, backend=backend)
    attempts = []
    for alternative in (False, True):
        trace = []
        def fun(t):
            v, g = objective.value_gradient(t)
            return v / (alpha * stats.n), g / (alpha * stats.n)
        def callback(t):
            trace.append(float(objective.value_gradient(t, gradient=False)))
        res = minimize(fun, objective.initial(alternative), jac=True, method="L-BFGS-B",
                       callback=callback,
                       options={"maxiter": config["max_iter"], "ftol": config["ftol"],
                                "gtol": config["gtol"], "maxls": 40, "maxcor": 15})
        attempts.append({"theta": res.x, "success": bool(res.success), "message": str(res.message),
                         "iterations": int(res.nit), "training_vfe": float(res.fun * alpha * stats.n),
                         "gradient_max_scaled": float(np.max(np.abs(res.jac))), "trace": trace})
    converged = [attempt for attempt in attempts if attempt["success"]]
    if not converged:
        raise RuntimeError(f"marginal optimizer did not converge: {[a['message'] for a in attempts]}")
    best = min(converged, key=lambda a: a["training_vfe"])
    objective.set_draws(config["evaluation_draws"], seed + 7000001)
    score, random = objective.value_gradient(best["theta"], samples=True)
    se = float(np.std(random, ddof=1) / np.sqrt(len(random)))
    return {"theta": best["theta"], "posterior": objective.posterior(best["theta"]),
            "score": score, "score_se": se, "evaluation_draws": len(random),
            "training_vfe": best["training_vfe"], "attempts": attempts,
            "converged_initializations": len(converged),
            "seconds": time.perf_counter() - started, "seed": int(seed)}


def reconstruct_covariance(stats, posterior):
    membership = np.full(stats.p, -1, dtype=int)
    for h, block in enumerate(stats.blocks):
        membership[block] = h
    used = np.flatnonzero(membership >= 0)
    cov = np.zeros((stats.p, stats.p))
    if len(used):
        ids = membership[used]
        cov[np.ix_(used, used)] = posterior["phi_mean"][np.ix_(ids, ids)]
    residual = np.full(stats.p, posterior["background_mean"])
    for h, block in enumerate(stats.blocks):
        residual[block] = posterior["residual_mean"][h]
    cov[np.diag_indices(stats.p)] += residual
    return cov


def reference_k(paths: Sequence[Mapping[str, Any]]) -> int:
    """Return K_ref, the number of communities on the shortest path."""

    lengths = [int(path["K"]) for path in paths]
    if not lengths:
        raise ValueError("at least one ICONS path is required")
    k_ref = min(lengths)
    if k_ref < 1:
        raise RuntimeError(
            "the shortest ICONS path has no community, so the reference model is undefined"
        )
    return k_ref


def prefix_candidates(paths: Sequence[Mapping[str, Any]], k: int) -> list[dict[str, Any]]:
    """Distinct k-community prefixes in grid order.

    Paths whose first k communities form the same partition share one
    candidate.  ``first_path`` is the grid position of the earliest such path.
    """

    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    distinct: dict[tuple, dict[str, Any]] = {}
    for position, path in enumerate(paths):
        if int(path["K"]) < k:
            continue
        blocks = [[int(index) for index in block] for block in path["blocks"][:k]]
        key = tuple(sorted(tuple(sorted(block)) for block in blocks))
        eta = {name: path[name] for name in ("q", "lambda", "threshold")}
        if key not in distinct:
            distinct[key] = {"blocks": blocks, "etas": [], "first_path": position}
        distinct[key]["etas"].append(eta)
    if not distinct:
        raise RuntimeError(f"no ICONS path has {k} communities")
    return list(distinct.values())


def select_reference(scores: Sequence[float]) -> int:
    """Index of the smallest score; exact ties resolve to the earliest index."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("reference scores must form a nonempty finite vector")
    return int(np.flatnonzero(values == values.min())[0])


def choose_largest_indistinguishable(
    records: Sequence[Mapping[str, Any]], z: float = 2.0, *, exclude_last: bool = False
) -> Mapping[str, Any]:
    """Largest visited K whose score is within z standard errors of the best.

    ``exclude_last`` drops the row that triggered a first-increase stop, which
    is significantly worse than its predecessor and is never selected.
    """

    rows = list(records[:-1] if exclude_last else records)
    if not rows:
        raise ValueError("no visited model is eligible for selection")
    if not np.isfinite(z) or z < 0:
        raise ValueError("z must be finite and nonnegative")
    best = min(rows, key=lambda row: float(row["score"]))
    eligible = [
        row
        for row in rows
        if float(row["score"]) - float(best["score"])
        <= z * float(np.hypot(float(row["score_se"]), float(best["score_se"])))
    ]
    return max(eligible, key=lambda row: int(row["k"]))


def search_path(
    path_blocks: Sequence[Sequence[int]],
    *,
    zero: Callable[[], tuple[float, Any]],
    fit: Callable[[int, list], dict[str, Any]],
    reevaluate: Callable[[int, list, dict[str, Any]], dict[str, Any]],
    z: float = 2.0,
    k_cap: int | None = None,
    prefit: Mapping[int, dict[str, Any]] | None = None,
    on_frozen: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], str]:
    """First-increase search for K along one fixed nested path.

    ``zero()`` returns the closed-form K=0 score and posterior.  ``fit`` fits
    the model with the first k communities.  When a difference is not resolved
    at z standard errors, ``reevaluate`` recomputes only the current score with
    more evaluation draws; earlier frozen scores are never changed.  A
    difference that is still unresolved counts as "not worse" and the search
    continues.
    """

    if not np.isfinite(z) or z < 0:
        raise ValueError("z must be finite and nonnegative")
    blocks_all = [list(block) for block in path_blocks]
    score0, posterior0 = zero()
    records: list[dict[str, Any]] = [
        dict(k=0, score=float(score0), score_se=0.0, delta=None, delta_se=None,
             unresolved=False)
    ]
    fits: dict[int, dict[str, Any]] = {
        0: dict(score=float(score0), score_se=0.0, posterior=posterior0)
    }
    upper = len(blocks_all) if k_cap is None else min(len(blocks_all), int(k_cap))
    status = "path-exhausted" if upper == len(blocks_all) else "cap-reaching"
    for k in range(1, upper + 1):
        blocks = blocks_all[:k]
        result = prefit[k] if prefit is not None and k in prefit else fit(k, blocks)
        previous = records[-1]
        delta = float(result["score"]) - previous["score"]
        delta_se = float(np.hypot(float(result["score_se"]), previous["score_se"]))
        if abs(delta) <= z * delta_se:
            result = reevaluate(k, blocks, result)
            delta = float(result["score"]) - previous["score"]
            delta_se = float(np.hypot(float(result["score_se"]), previous["score_se"]))
        row = dict(
            k=k, score=float(result["score"]), score_se=float(result["score_se"]),
            delta=delta, delta_se=delta_se, unresolved=bool(abs(delta) <= z * delta_se),
        )
        records.append(row)
        fits[k] = result
        if on_frozen is not None:
            on_frozen(row, result)
        if delta > z * delta_se:
            status = "first-increase"
            break
    return records, fits, status



def search_joint(
    paths: Sequence[Mapping[str, Any]],
    *,
    zero: Callable[[], tuple[float, Any]],
    fit: Callable[[int, list], dict[str, Any]],
    reevaluate: Callable[[int, list, dict[str, Any]], dict[str, Any]],
    z: float = 2.0,
    k_cap: int | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], str]:
    """Joint first-increase search (Algorithm 1), the version analyzed by the theory.

    At every K the score is the smallest score among all distinct K-community
    prefixes of the paths; the search stops at the first increase exceeding z
    standard errors.  When the step is unresolved, every candidate at that K is
    re-evaluated with more draws.  The returned fit for each K carries its blocks.
    The real-data analyses of the paper use the two-stage search instead.
    """

    if not np.isfinite(z) or z < 0:
        raise ValueError("z must be finite and nonnegative")
    score0, posterior0 = zero()
    records: list[dict[str, Any]] = [
        dict(k=0, score=float(score0), score_se=0.0, delta=None, delta_se=None,
             unresolved=False, candidate_count=1)
    ]
    fits: dict[int, dict[str, Any]] = {0: dict(score=float(score0), score_se=0.0,
                                               posterior=posterior0, blocks=[])}
    upper = max(int(path["K"]) for path in paths)
    status = "paths-exhausted"
    if k_cap is not None and int(k_cap) < upper:
        upper, status = int(k_cap), "cap-reaching"
    for k in range(1, upper + 1):
        candidates = prefix_candidates(paths, k)
        results = [fit(k, [list(block) for block in item["blocks"]]) for item in candidates]
        best = int(np.argmin([float(result["score"]) for result in results]))
        previous = records[-1]
        delta = float(results[best]["score"]) - previous["score"]
        delta_se = float(np.hypot(float(results[best]["score_se"]), previous["score_se"]))
        if abs(delta) <= z * delta_se:
            results = [reevaluate(k, [list(block) for block in item["blocks"]], result)
                       for item, result in zip(candidates, results)]
            best = int(np.argmin([float(result["score"]) for result in results]))
            delta = float(results[best]["score"]) - previous["score"]
            delta_se = float(np.hypot(float(results[best]["score_se"]), previous["score_se"]))
        chosen = dict(results[best])
        chosen["blocks"] = [list(block) for block in candidates[best]["blocks"]]
        records.append(dict(k=k, score=float(chosen["score"]), score_se=float(chosen["score_se"]),
                            delta=delta, delta_se=delta_se, unresolved=bool(abs(delta) <= z * delta_se),
                            candidate_count=len(candidates), winner_first_path=candidates[best]["first_path"]))
        fits[k] = chosen
        if delta > z * delta_se:
            status = "first-increase"
            break
    return records, fits, status


def null_learning_rate(x) -> float:
    """Learning rate alpha_0 of the null model, in which all variables share one variance.

    The Lyddon--Holmes--Walker rate tr(J L^{-1} J) / tr(J) of the model with no
    community, with J the Fisher information and L the second moment of the
    per-observation score at the maximum-likelihood estimate.  It is roughly the
    effective number of variables divided by p.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    if not np.isfinite(x).all():
        raise ValueError("need finite data")
    background = np.sum(x[:, np.arange(p)] ** 2, axis=1)
    s0 = float(background.mean() / p)
    if s0 <= 0:
        raise ArithmeticError("non-positive residual variance at the null-model MLE")
    scores = np.zeros((n, 1))
    information = np.zeros((1, 1))
    scores[:, -1] = -0.5 * (p / s0 - background / s0 ** 2)
    information[-1, -1] = p / (2 * s0 ** 2)
    if n <= 1:
        raise ValueError("the score covariance needs at least two observations")
    second = scores.T @ scores / n
    alpha = float(np.trace(information @ np.linalg.solve(second, information)) / np.trace(information))
    if not 0 < alpha < 1:
        raise RuntimeError(f"null-model learning rate {alpha} lies outside (0, 1)")
    return alpha


def posterior_zero(stats, alpha, prior):
    """Closed-form free energy and posterior of the model with no community."""
    score, a, b = background_vfe(stats.n, stats.p, stats.background_ss, alpha, prior["a0"], prior["b0"])
    posterior = dict(phi_mean=np.empty((0, 0)), residual_mean=np.empty(0), background_mean=b / (a - 1),
                     background_shape=a, background_rate=b)
    return score, posterior


__all__ = [
    "BlockStats",
    "MarginalObjective",
    "block_stats",
    "choose_largest_indistinguishable",
    "fit_marginal",
    "null_learning_rate",
    "posterior_zero",
    "prefix_candidates",
    "reconstruct_covariance",
    "reference_k",
    "search_joint",
    "search_path",
    "select_reference",
]
