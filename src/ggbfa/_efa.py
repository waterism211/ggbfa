"""Exploratory step of gg-BFA: ``efa`` learns the communities and their number.

1. Standardize the data and run the dense-subgraph search over a grid of
   (percentile, lambda) pairs; each pair gives a nested path of communities.
2. Compute the learning rate alpha_0 of the null model.
3. Search the number of communities K with the alpha_0-fractional variational
   posterior of the selection model, stopping at the first significant decrease
   of the ELBO.  ``search="two_stage"`` (Algorithm 2 of the paper, used for its
   real-data analyses) first keeps one path at a reference size and then moves
   along it; ``search="full"`` (Algorithm 1) compares every distinct prefix of
   every path at each K.
4. Return the largest visited K whose ELBO is statistically indistinguishable
   from the best visited ELBO, with its partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._backend import resolve_compute_backend
from ._icons import icons_paths
from ._selection import (
    MarginalObjective,
    block_stats,
    choose_largest_indistinguishable,
    fit_marginal,
    null_learning_rate,
    posterior_zero,
    prefix_candidates,
    reconstruct_covariance,
    reference_k,
    search_joint,
    search_path,
    select_reference,
)
from ._version import __version__

PERCENTILES = (90, 91, 92, 93, 94, 95, 96, 97, 98, 99)
LAMBDAS = (0.4, 0.5, 0.6, 0.7, 0.8)
PRIOR = {"a0": 2.0, "b0": 0.5, "c": 0.5, "kappa": 1.0}
SEARCHES = ("two_stage", "full")
SEED_STEP = 10007          # fit seed of the model with k communities: random_state + k * SEED_STEP
REEVALUATION_OFFSET = 7000001


@dataclass(frozen=True)
class EFAResult:
    """Communities learned by :func:`ggbfa.efa`; pass it to :func:`ggbfa.cfa`.

    ``communities`` lists the selected communities as 0-based column indices,
    each in the order of the dense-subgraph search (densest core first);
    ``background`` holds the variables in no community.  ``trace`` has one row
    per visited K with its ELBO (larger is better) and Monte Carlo standard
    error.  ``covariance`` is the covariance of the standardized data under the
    selected fractional variational posterior of the selection model.
    """

    communities: Tuple[np.ndarray, ...]
    background: np.ndarray
    n_factors: int
    search: str
    status: str
    trace: List[Dict[str, Any]]
    stop_k: Optional[int]
    alpha: float
    paths: List[Dict[str, Any]]
    covariance: np.ndarray = field(repr=False)
    posterior: Dict[str, Any] = field(repr=False)
    data_mean: np.ndarray = field(repr=False)
    data_scale: np.ndarray = field(repr=False)
    settings: Dict[str, Any] = field(repr=False)
    stage1: Optional[Dict[str, Any]] = field(default=None, repr=False)
    runtime: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def order(self) -> np.ndarray:
        """Variables ordered by community, then the background (for plots)."""

        return np.concatenate([*self.communities, self.background]).astype(int)

    def membership(self) -> np.ndarray:
        """Community label of each variable (1..K), 0 for the background."""

        labels = np.zeros(len(self.data_mean), dtype=int)
        for h, block in enumerate(self.communities):
            labels[block] = h + 1
        return labels


def _standardize(x: np.ndarray):
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("data contain a constant variable")
    return (x - mean) / scale, mean, scale


def _sample_correlation(z: np.ndarray) -> np.ndarray:
    corr = z.T @ z / len(z)
    return (corr + corr.T) / 2


def efa(
    data,
    *,
    search: str = "two_stage",
    percentiles: Sequence[float] = PERCENTILES,
    lambdas: Sequence[float] = LAMBDAS,
    min_size: int = 2,
    prior: Optional[Dict[str, float]] = None,
    optimization_draws: int = 256,
    evaluation_draws: int = 2048,
    max_evaluation_draws: int = 16384,
    max_iter: int = 600,
    ftol: float = 1e-10,
    gtol: float = 1e-5,
    decision_z: float = 2.0,
    max_k: Optional[int] = None,
    random_state: int = 0,
    device: str = "auto",
    gpu_device_id: int = 0,
) -> EFAResult:
    """Learn the communities of gg-BFA (the exploratory step).

    Parameters
    ----------
    data:
        Complete n-by-p matrix; it may have p > n.  Columns are standardized.
    search:
        ``"two_stage"`` (default; Algorithm 2) or ``"full"`` (Algorithm 1).
    percentiles, lambdas, min_size:
        Grid of the dense-subgraph search.  For each percentile (0-100 scale)
        of the sample correlations above the diagonal, the correlations at or
        above it become the edges; lambda is the size penalty of the density.
        Communities have at least ``min_size`` variables.
    prior:
        Hyperparameters ``a0, b0`` of the inverse-gamma residual variances and
        ``c, kappa`` of the inverse-Wishart factor covariance of the selection
        model.
    optimization_draws, evaluation_draws, max_evaluation_draws:
        Monte Carlo draws used to optimize the ELBO, to evaluate it, and to
        re-evaluate an unresolved comparison.
    max_iter, ftol, gtol:
        L-BFGS-B settings of each fit.
    decision_z:
        A change of the ELBO counts only when it exceeds ``decision_z``
        Monte Carlo standard errors.
    max_k:
        Optional cap on the number of communities visited.
    random_state:
        Seed of the Monte Carlo draws; results are reproducible for a fixed
        seed and device.
    device:
        ``"auto"``, ``"cpu"`` or ``"gpu"`` for the ELBO evaluations; the
        dense-subgraph search always runs on the CPU.
    """

    started = perf_counter()
    if search not in SEARCHES:
        raise ValueError(f"search must be one of: {', '.join(SEARCHES)}")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("data must be a complete finite matrix with n >= 3 and p >= 2")
    if isinstance(random_state, bool) or int(random_state) != random_state or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    if max_k is not None and (isinstance(max_k, bool) or int(max_k) != max_k or max_k < 1):
        raise ValueError("max_k must be None or a positive integer")
    prior = dict(PRIOR if prior is None else prior)
    if set(prior) != set(PRIOR):
        raise ValueError(f"prior must have the keys {sorted(PRIOR)}")
    config = dict(prior=prior, optimization_draws=int(optimization_draws),
                  evaluation_draws=int(evaluation_draws), max_evaluation_draws=int(max_evaluation_draws),
                  max_iter=int(max_iter), ftol=float(ftol), gtol=float(gtol))
    z = float(decision_z)
    seed = int(random_state)

    standardized, mean, scale = _standardize(x)
    discovery_started = perf_counter()
    paths = icons_paths(_sample_correlation(standardized), percentiles, lambdas, min_size)
    discovery_seconds = perf_counter() - discovery_started
    backend = resolve_compute_backend(device, gpu_device_id)
    alpha = null_learning_rate(standardized)
    cache: Dict[tuple, Any] = {}

    def stats_for(blocks):
        key = tuple(tuple(int(i) for i in block) for block in blocks)
        if key not in cache:
            cache[key] = block_stats(standardized, blocks)
        return cache[key]

    def zero():
        return posterior_zero(stats_for([]), alpha, prior)

    def fit(k, blocks):
        return fit_marginal(stats_for(blocks), alpha, config=config, backend=backend, seed=seed + k * SEED_STEP)

    def reevaluate(k, blocks, result):
        objective = MarginalObjective(stats_for(blocks), alpha, **prior, backend=backend,
                                      draws=config["max_evaluation_draws"],
                                      seed=result["seed"] + REEVALUATION_OFFSET)
        value, draws = objective.value_gradient(result["theta"], samples=True)
        updated = dict(result)
        updated.update(score=value, score_se=float(draws.std(ddof=1) / np.sqrt(len(draws))),
                       evaluation_draws=len(draws))
        return updated

    stage1 = None
    if search == "full":
        records, fits, status = search_joint(paths, zero=zero, fit=fit, reevaluate=reevaluate, z=z, k_cap=max_k)
        blocks_by_k = {k: fits[k]["blocks"] for k in fits}
    else:
        k_ref = reference_k(paths)
        candidates = prefix_candidates(paths, k_ref)
        fitted = [fit(k_ref, item["blocks"]) for item in candidates]
        chosen = select_reference([item["score"] for item in fitted])
        stage1 = dict(k_ref=k_ref, chosen=chosen, path=candidates[chosen]["first_path"],
                      equivalent=candidates[chosen]["etas"],
                      candidates=[dict(blocks=item["blocks"], etas=item["etas"], score=result["score"],
                                       score_se=result["score_se"])
                                  for item, result in zip(candidates, fitted)])
        path = paths[stage1["path"]]
        records, fits, status = search_path(path["blocks"], zero=zero, fit=fit, reevaluate=reevaluate, z=z,
                                            k_cap=max_k, prefit={k_ref: fitted[chosen]})
        blocks_by_k = {k: [list(b) for b in path["blocks"][:k]] for k in fits}
    selected = choose_largest_indistinguishable(records, z, exclude_last=status == "first-increase")
    k = int(selected["k"])
    blocks = blocks_by_k[k]
    stats = stats_for(blocks)
    posterior = fits[k]["posterior"]
    communities = tuple(np.asarray(block, dtype=int) for block in blocks)
    used = np.concatenate(communities) if communities else np.empty(0, dtype=int)
    background = np.setdiff1d(np.arange(x.shape[1]), used)
    trace = [dict(k=int(row["k"]), elbo=-float(row["score"]), elbo_se=float(row["score_se"]),
                  unresolved=bool(row["unresolved"]),
                  **({"candidates": int(row["candidate_count"])} if "candidate_count" in row else {}))
             for row in records]
    runtime = dict(backend.diagnostics(), package_version=__version__, seconds=perf_counter() - started,
                   discovery_seconds=discovery_seconds, discovery_device="cpu")
    settings = dict(search=search, percentiles=list(percentiles), lambdas=list(lambdas), min_size=int(min_size),
                    decision_z=z, max_k=max_k, random_state=seed, **config)
    return EFAResult(
        communities=communities,
        background=background,
        n_factors=k,
        search=search,
        status=status,
        trace=trace,
        stop_k=int(records[-1]["k"]) if status == "first-increase" else None,
        alpha=alpha,
        paths=paths,
        covariance=reconstruct_covariance(stats, posterior),
        posterior=posterior,
        data_mean=mean,
        data_scale=scale,
        settings=settings,
        stage1=stage1,
        runtime=runtime,
    )


__all__ = ["EFAResult", "efa"]
