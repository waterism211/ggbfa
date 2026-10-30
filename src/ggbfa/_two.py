"""Member 2f: at most one cross-loading per row.

This variational model replaces independent cross-loading
indicators with one categorical state per observed variable.  A row is either
purely primary or has exactly one cross-loading target.  The signed row-sum
constraint is retained exactly::

    lambda_j = e_h + eta_j (e_k - e_h),

where ``h`` is the supplied primary factor and ``k`` is the optional target.
With loading_normalization="free" the row is
``lambda_j = a_j e_h + b_j e_k`` with a Gaussian q(a_j) under a N(m0, v0) prior,
independent of the row state; b_j has the cross-loading prior of eta_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import special, stats

from ._backend import ComputeBackend, resolve_compute_backend
from ._model import (
    Block,
    CFADiagnostics,
    CFAPosterior,
    CFAResult,
    _factor_full_elbo,
    _invert_spd,
    _prepare_data,
    _primary_loading_elbo,
    _residual_ss,
    _solve_spd,
    _update_primary_loadings,
    make_scfa_block_structure,
)
from ._members import normalize_member


@dataclass
class _TwoPosterior(CFAPosterior):
    """Posterior summaries for row overlap and target-factor selection."""

    row_overlap_probability: Optional[np.ndarray] = None
    row_state_probability: Optional[np.ndarray] = None
    target_probability_given_overlap: Optional[np.ndarray] = None
    factor_overlap_probability_mean: Optional[np.ndarray] = None
    target_probability_mean: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _TwoDiagnostics(CFADiagnostics):
    """Variational parameters for the row-sparse loading family."""

    row_state_probability: Optional[np.ndarray] = None
    active_state_means: Optional[np.ndarray] = None
    active_state_variances: Optional[np.ndarray] = None
    overlap_beta_alpha: Optional[np.ndarray] = None
    overlap_beta_beta: Optional[np.ndarray] = None
    target_dirichlet_alpha: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _TwoResult(CFAResult):
    posterior: _TwoPosterior
    diagnostics: _TwoDiagnostics


def _validate_row_sparse_inputs(
    *,
    max_iter: int,
    min_iter: int,
    tolerance: float,
    credible_interval_level: float,
    factor_prior_variance: float,
    row_overlap_prior_alpha: float,
    row_overlap_prior_beta: float,
    target_dirichlet_concentration: float,
    active_cross_loading_variance: float,
    residual_prior_shape: float,
    residual_prior_scale: float,
    elbo_decrease_tolerance: float,
    factor_covariance_prior_strength: float,
    factor_covariance_interval_draws: int,
    factor_covariance_interval_seed: int,
) -> None:
    if max_iter <= 0 or min_iter <= 0 or min_iter > max_iter:
        raise ValueError("require 0 < min_iter <= max_iter")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not 0.0 < credible_interval_level < 1.0:
        raise ValueError("credible_interval_level must be in (0,1)")
    positive = {
        "factor_prior_variance": factor_prior_variance,
        "row_overlap_prior_alpha": row_overlap_prior_alpha,
        "row_overlap_prior_beta": row_overlap_prior_beta,
        "target_dirichlet_concentration": target_dirichlet_concentration,
        "active_cross_loading_variance": active_cross_loading_variance,
        "residual_prior_shape": residual_prior_shape,
        "residual_prior_scale": residual_prior_scale,
        "factor_covariance_prior_strength": factor_covariance_prior_strength,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if elbo_decrease_tolerance < 0.0:
        raise ValueError("elbo_decrease_tolerance must be nonnegative")
    if int(factor_covariance_interval_draws) < 100:
        raise ValueError("at least 100 posterior interval draws are required")
    if int(factor_covariance_interval_seed) != factor_covariance_interval_seed:
        raise ValueError("factor_covariance_interval_seed must be an integer")


def _initialize_states(
    membership: np.ndarray,
    k_count: int,
    prior_alpha: float,
    prior_beta: float,
) -> np.ndarray:
    overlap = prior_alpha / (prior_alpha + prior_beta)
    state = np.full((membership.size, k_count), overlap / (k_count - 1), dtype=float)
    state[np.arange(membership.size), membership] = 1.0 - overlap
    return state


def _free_row_sparse_loading_moments(
    membership: np.ndarray,
    state_probability: np.ndarray,
    active_mean: np.ndarray,
    active_variance: np.ndarray,
    primary_moments: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E[Lambda], E[lambda lambda'], Cov(lambda) for free primary loadings independent of the row state."""

    p_count, k_count = state_probability.shape
    rows = np.arange(p_count)
    factors = np.arange(k_count)
    primary_mean, primary_variance = primary_moments
    cross_mean = state_probability * active_mean
    cross_mean[rows, membership] = 0.0
    cross_second = state_probability * (active_variance + active_mean * active_mean)
    cross_second[rows, membership] = 0.0
    loading_mean = cross_mean.copy()
    loading_mean[rows, membership] = primary_mean
    loading_second = np.zeros((p_count, k_count, k_count), dtype=float)
    loading_second[:, factors, factors] = cross_second
    loading_second[rows, membership, membership] = primary_variance + primary_mean * primary_mean
    loading_second[rows, membership, :] += primary_mean[:, None] * cross_mean
    loading_second[rows, :, membership] += primary_mean[:, None] * cross_mean
    loading_covariance = loading_second - (
        loading_mean[:, :, None] * loading_mean[:, None, :]
    )
    loading_covariance = 0.5 * (
        loading_covariance + np.swapaxes(loading_covariance, 1, 2)
    )
    return loading_mean, loading_second, loading_covariance


def _row_sparse_loading_moments(
    membership: np.ndarray,
    state_probability: np.ndarray,
    active_mean: np.ndarray,
    active_variance: np.ndarray,
    primary_moments: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return E[Lambda], E[lambda lambda'], and Cov(lambda) per row."""

    if primary_moments is not None:
        return _free_row_sparse_loading_moments(
            membership, state_probability, active_mean, active_variance, primary_moments
        )
    p_count, k_count = state_probability.shape
    loading_mean = np.zeros((p_count, k_count), dtype=float)
    loading_second = np.zeros((p_count, k_count, k_count), dtype=float)
    for primary in range(k_count):
        row_ids = np.flatnonzero(membership == primary)
        if row_ids.size == 0:
            continue
        no_weight = state_probability[row_ids, primary]
        loading_mean[row_ids, primary] = no_weight
        loading_second[row_ids, primary, primary] = no_weight
        for target in range(k_count):
            if target == primary:
                continue
            probability = state_probability[row_ids, target]
            mean = active_mean[row_ids, target]
            variance = active_variance[row_ids, target]
            eta_second = variance + mean * mean
            cross_second = mean - eta_second
            loading_mean[row_ids, primary] += probability * (1.0 - mean)
            loading_mean[row_ids, target] += probability * mean
            loading_second[row_ids, primary, primary] += probability * (
                1.0 - 2.0 * mean + eta_second
            )
            loading_second[row_ids, target, target] += probability * eta_second
            loading_second[row_ids, primary, target] = probability * cross_second
            loading_second[row_ids, target, primary] = probability * cross_second
    loading_covariance = loading_second - (
        loading_mean[:, :, None] * loading_mean[:, None, :]
    )
    loading_covariance = 0.5 * (
        loading_covariance + np.swapaxes(loading_covariance, 1, 2)
    )
    return loading_mean, loading_second, loading_covariance


def _update_row_states(
    *,
    membership: np.ndarray,
    factor_second_sum: np.ndarray,
    factor_response_sums: np.ndarray,
    expected_precision: np.ndarray,
    overlap_alpha: np.ndarray,
    overlap_beta: np.ndarray,
    target_alpha: np.ndarray,
    active_cross_loading_variance: float,
    primary_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly update q(state_j) q(eta_j | state_j); with primary_mean, q(b_j | state_j) of the free family."""

    p_count = membership.size
    k_count = factor_second_sum.shape[0]
    state = np.zeros((p_count, k_count), dtype=float)
    active_mean = np.zeros_like(state)
    active_variance = np.zeros_like(state)
    expected_log_overlap = special.digamma(overlap_alpha) - special.digamma(
        overlap_alpha + overlap_beta
    )
    expected_log_no_overlap = special.digamma(overlap_beta) - special.digamma(
        overlap_alpha + overlap_beta
    )
    factor_diagonal = np.diag(factor_second_sum)
    factor_ids = np.arange(k_count)
    prior_precision = 1.0 / active_cross_loading_variance
    for primary in range(k_count):
        row_ids = np.flatnonzero(membership == primary)
        if row_ids.size == 0:
            continue
        targets = factor_ids[factor_ids != primary]
        target_total = np.sum(target_alpha[primary, targets])
        expected_log_target = special.digamma(
            target_alpha[primary, targets]
        ) - special.digamma(target_total)
        if primary_mean is None:
            factor_sum = (
                factor_diagonal[targets]
                + factor_diagonal[primary]
                - 2.0 * factor_second_sum[targets, primary]
            )
            factor_sum = np.maximum(factor_sum, 1e-15)
            response = (
                factor_response_sums[row_ids[:, None], targets[None, :]]
                - factor_response_sums[row_ids, primary][:, None]
                - factor_second_sum[targets, primary][None, :]
                + factor_second_sum[primary, primary]
            )
        else:
            factor_sum = np.maximum(factor_diagonal[targets], 1e-15)
            response = (
                factor_response_sums[row_ids[:, None], targets[None, :]]
                - primary_mean[row_ids, None] * factor_second_sum[primary, targets][None, :]
            )
        tau = expected_precision[row_ids, None]
        variance = 1.0 / (tau * factor_sum[None, :] + prior_precision)
        natural = tau * response
        mean = variance * natural
        log_evidence = (
            0.5 * np.log(variance / active_cross_loading_variance)
            + 0.5 * natural * natural * variance
        )
        log_weight = np.full((row_ids.size, k_count), -np.inf, dtype=float)
        log_weight[:, primary] = expected_log_no_overlap[primary]
        log_weight[:, targets] = (
            expected_log_overlap[primary] + expected_log_target[None, :] + log_evidence
        )
        log_weight -= special.logsumexp(log_weight, axis=1)[:, None]
        state[row_ids] = np.exp(log_weight)
        active_mean[np.ix_(row_ids, targets)] = mean
        active_variance[np.ix_(row_ids, targets)] = variance
    return state, active_mean, active_variance


def _update_global_loading_factors(
    *,
    membership: np.ndarray,
    state_probability: np.ndarray,
    prior_alpha: float,
    prior_beta: float,
    target_concentration: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_count = state_probability.shape[1]
    overlap_alpha = np.full(k_count, prior_alpha, dtype=float)
    overlap_beta = np.full(k_count, prior_beta, dtype=float)
    target_alpha = np.zeros((k_count, k_count), dtype=float)
    for primary in range(k_count):
        rows = membership == primary
        targets = np.arange(k_count) != primary
        overlap_alpha[primary] += np.sum(state_probability[rows][:, targets])
        overlap_beta[primary] += np.sum(state_probability[rows, primary])
        target_alpha[primary, targets] = target_concentration + np.sum(
            state_probability[rows][:, targets], axis=0
        )
    return overlap_alpha, overlap_beta, target_alpha


def _row_sparse_elbo(
    *,
    x: np.ndarray,
    membership: np.ndarray,
    factor_mean: np.ndarray,
    factor_covariance: np.ndarray,
    factor_second_sum: np.ndarray,
    loading_mean: np.ndarray,
    loading_second: np.ndarray,
    loading_covariance: np.ndarray,
    state_probability: np.ndarray,
    active_mean: np.ndarray,
    active_variance: np.ndarray,
    overlap_alpha: np.ndarray,
    overlap_beta: np.ndarray,
    target_alpha: np.ndarray,
    row_overlap_prior_alpha: float,
    row_overlap_prior_beta: float,
    target_dirichlet_concentration: float,
    active_cross_loading_variance: float,
    residual_shape: np.ndarray,
    residual_scale: np.ndarray,
    residual_prior_shape: float,
    residual_prior_scale: float,
    phi_posterior_df: float,
    phi_posterior_scale: np.ndarray,
    phi_prior_df: float,
    phi_prior_scale: np.ndarray,
    primary_term: Optional[float] = None,
) -> float:
    n_count = x.shape[0]
    expected_precision = residual_shape / residual_scale
    expected_log_variance = np.log(residual_scale) - special.digamma(residual_shape)
    rss = _residual_ss(
        x,
        factor_mean,
        factor_covariance,
        loading_mean,
        loading_second,
        loading_covariance,
    )
    value = float(
        np.sum(
            -0.5 * n_count * np.log(2.0 * np.pi)
            - 0.5 * n_count * expected_log_variance
            - 0.5 * expected_precision * rss
        )
    )
    value += _factor_full_elbo(
        n_count,
        factor_covariance,
        factor_second_sum,
        phi_posterior_df=phi_posterior_df,
        phi_posterior_scale=phi_posterior_scale,
        phi_prior_df=phi_prior_df,
        phi_prior_scale=phi_prior_scale,
    )
    expected_log_prior = (
        residual_prior_shape * np.log(residual_prior_scale)
        - special.gammaln(residual_prior_shape)
        - (residual_prior_shape + 1.0) * expected_log_variance
        - residual_prior_scale * expected_precision
    )
    residual_entropy = (
        residual_shape
        + np.log(residual_scale)
        + special.gammaln(residual_shape)
        - (residual_shape + 1.0) * special.digamma(residual_shape)
    )
    value += float(np.sum(expected_log_prior + residual_entropy))

    expected_log_overlap = special.digamma(overlap_alpha) - special.digamma(
        overlap_alpha + overlap_beta
    )
    expected_log_no = special.digamma(overlap_beta) - special.digamma(
        overlap_alpha + overlap_beta
    )
    beta_prior_log_norm = special.betaln(
        row_overlap_prior_alpha, row_overlap_prior_beta
    )
    for primary in range(state_probability.shape[1]):
        e_log_theta = expected_log_overlap[primary]
        e_log_no = expected_log_no[primary]
        value += float(
            (row_overlap_prior_alpha - overlap_alpha[primary]) * e_log_theta
            + (row_overlap_prior_beta - overlap_beta[primary]) * e_log_no
            - beta_prior_log_norm
            + special.betaln(overlap_alpha[primary], overlap_beta[primary])
        )

        rows = np.flatnonzero(membership == primary)
        targets = np.flatnonzero(np.arange(state_probability.shape[1]) != primary)
        q_alpha = target_alpha[primary, targets]
        expected_log_target = special.digamma(q_alpha) - special.digamma(
            np.sum(q_alpha)
        )
        prior_alpha = np.full(targets.size, target_dirichlet_concentration)
        value += float(
            special.gammaln(np.sum(prior_alpha))
            - np.sum(special.gammaln(prior_alpha))
            - special.gammaln(np.sum(q_alpha))
            + np.sum(special.gammaln(q_alpha))
            + np.sum((prior_alpha - q_alpha) * expected_log_target)
        )

        no_probability = state_probability[rows, primary]
        cross_probability = state_probability[np.ix_(rows, targets)]
        value += float(np.sum(no_probability * e_log_no))
        value += float(
            np.sum(cross_probability * (e_log_theta + expected_log_target[None, :]))
        )
        probabilities = state_probability[rows]
        value += float(-np.sum(special.xlogy(probabilities, probabilities)))
        means = active_mean[np.ix_(rows, targets)]
        variances = active_variance[np.ix_(rows, targets)]
        normal_prior_entropy = 0.5 * (
            1.0
            + np.log(variances / active_cross_loading_variance)
            - (variances + means * means) / active_cross_loading_variance
        )
        value += float(np.sum(cross_probability * normal_prior_entropy))
    if primary_term is not None:  # free primary loadings
        value += primary_term
    if not np.isfinite(value):
        raise RuntimeError("row-sparse variational ELBO is not finite")
    return value


def _draw_row_sparse_loadings(
    *,
    membership: np.ndarray,
    state_probability: np.ndarray,
    active_mean: np.ndarray,
    active_variance: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    primary_moments: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    p_count, k_count = state_probability.shape
    output = np.zeros((draws, p_count, k_count), dtype=float)
    uniforms = rng.random((draws, p_count))
    cumulative = np.cumsum(state_probability, axis=1)
    states = np.sum(uniforms[:, :, None] > cumulative[None, :, :], axis=2)
    states = np.minimum(states, k_count - 1)
    for variable_id, primary_value in enumerate(membership):
        primary = int(primary_value)
        if primary_moments is None:
            output[:, variable_id, primary] = 1.0
        else:
            output[:, variable_id, primary] = rng.normal(
                primary_moments[0][variable_id], np.sqrt(primary_moments[1][variable_id]), size=draws
            )
        for target in range(k_count):
            if target == primary:
                continue
            selected = states[:, variable_id] == target
            count = int(np.sum(selected))
            if count == 0:
                continue
            eta = rng.normal(
                active_mean[variable_id, target],
                np.sqrt(active_variance[variable_id, target]),
                size=count,
            )
            output[selected, variable_id, target] = eta
            if primary_moments is None:
                output[selected, variable_id, primary] = 1.0 - eta
    return output


def _fit_two(
    data: np.ndarray,
    blocks: Sequence[Block],
    *,
    member: str = "2f",
    max_iter: int = 1000,
    min_iter: int = 5,
    tolerance: float = 1e-6,
    credible_interval_level: float = 0.95,
    factor_prior_variance: float = 1.0,
    standardize_data: bool = True,
    row_overlap_prior_alpha: float = 1.0,
    row_overlap_prior_beta: float = 9.0,
    target_dirichlet_concentration: float = 1.0,
    active_cross_loading_variance: float = 0.25,
    residual_prior_shape: float = 0.01,
    residual_prior_scale: float = 0.01,
    elbo_decrease_tolerance: float = 1e-8,
    factor_covariance_prior_strength: float = 1.0,
    factor_covariance_interval_draws: int = 2000,
    factor_covariance_interval_seed: int = 1,
    loading_normalization: str = "row_sum",
    primary_loading_prior_mean: float = 1.0,
    primary_loading_prior_variance: float = 1.0,
    compute_backend: ComputeBackend = None,
) -> _TwoResult:
    """Fit member 2f: at most one cross-loading per row."""
    started = perf_counter()
    if loading_normalization not in ("row_sum", "free"):
        raise ValueError("loading_normalization must be 'row_sum' or 'free'")
    free = loading_normalization == "free"

    if normalize_member(member) != "2f":
        raise ValueError("_fit_two requires member='2f'")

    _validate_row_sparse_inputs(
        max_iter=max_iter,
        min_iter=min_iter,
        tolerance=tolerance,
        credible_interval_level=credible_interval_level,
        factor_prior_variance=factor_prior_variance,
        row_overlap_prior_alpha=row_overlap_prior_alpha,
        row_overlap_prior_beta=row_overlap_prior_beta,
        target_dirichlet_concentration=target_dirichlet_concentration,
        active_cross_loading_variance=active_cross_loading_variance,
        residual_prior_shape=residual_prior_shape,
        residual_prior_scale=residual_prior_scale,
        elbo_decrease_tolerance=elbo_decrease_tolerance,
        factor_covariance_prior_strength=factor_covariance_prior_strength,
        factor_covariance_interval_draws=factor_covariance_interval_draws,
        factor_covariance_interval_seed=factor_covariance_interval_seed,
    )
    standardized, data_mean, data_scale = _prepare_data(
        data, standardize=standardize_data
    )
    structure = make_scfa_block_structure(blocks, standardized.shape[1])
    x = standardized[:, structure.selected_indices]
    if compute_backend is None:
        compute_backend = resolve_compute_backend("cpu", 0)
    workspace = compute_backend.workspace(x)
    n_count, p_count = x.shape
    k_count = structure.loading_mask.shape[1]
    if k_count < 2:
        raise ValueError("member 2f requires at least two communities")
    membership = np.asarray(structure.membership, dtype=int)

    state_probability = _initialize_states(
        membership,
        k_count,
        row_overlap_prior_alpha,
        row_overlap_prior_beta,
    )
    active_mean = np.zeros_like(state_probability)
    active_variance = np.zeros_like(state_probability)
    cross_mask = ~np.asarray(structure.loading_mask, dtype=bool)
    active_variance[cross_mask] = active_cross_loading_variance
    overlap_alpha, overlap_beta, target_alpha = _update_global_loading_factors(
        membership=membership,
        state_probability=state_probability,
        prior_alpha=row_overlap_prior_alpha,
        prior_beta=row_overlap_prior_beta,
        target_concentration=target_dirichlet_concentration,
    )
    primary_moments = None
    if free:
        primary_moments = (
            np.full(p_count, float(primary_loading_prior_mean)),
            np.full(p_count, float(primary_loading_prior_variance)),
        )
    loading_mean, loading_second, loading_covariance = _row_sparse_loading_moments(
        membership,
        state_probability,
        active_mean,
        active_variance,
        primary_moments,
    )

    def primary_term():
        if not free:
            return None
        return _primary_loading_elbo(
            primary_moments[0], primary_moments[1], primary_loading_prior_mean, primary_loading_prior_variance
        )

    factor_mean = np.zeros((n_count, k_count), dtype=float)
    base_phi = factor_prior_variance * np.eye(k_count, dtype=float)
    factor_covariance = base_phi.copy()
    factor_second_sum = n_count * factor_covariance
    phi_prior_df = float(k_count + 1) + factor_covariance_prior_strength
    phi_prior_scale = factor_covariance_prior_strength * base_phi
    phi_posterior_df = phi_prior_df + n_count
    phi_posterior_scale = phi_prior_scale + factor_second_sum
    phi_scale_precision, _ = _invert_spd(phi_posterior_scale)
    phi_precision = phi_posterior_df * phi_scale_precision
    phi = phi_posterior_scale / (phi_posterior_df - k_count - 1.0)
    residual_shape = np.full(p_count, residual_prior_shape, dtype=float)
    residual_scale = np.full(p_count, residual_prior_scale, dtype=float)

    elbo_history = [
        _row_sparse_elbo(
            x=x,
            membership=membership,
            factor_mean=factor_mean,
            factor_covariance=factor_covariance,
            factor_second_sum=factor_second_sum,
            loading_mean=loading_mean,
            loading_second=loading_second,
            loading_covariance=loading_covariance,
            state_probability=state_probability,
            active_mean=active_mean,
            active_variance=active_variance,
            overlap_alpha=overlap_alpha,
            overlap_beta=overlap_beta,
            target_alpha=target_alpha,
            row_overlap_prior_alpha=row_overlap_prior_alpha,
            row_overlap_prior_beta=row_overlap_prior_beta,
            target_dirichlet_concentration=target_dirichlet_concentration,
            active_cross_loading_variance=active_cross_loading_variance,
            residual_shape=residual_shape,
            residual_scale=residual_scale,
            residual_prior_shape=residual_prior_shape,
            residual_prior_scale=residual_prior_scale,
            phi_posterior_df=phi_posterior_df,
            phi_posterior_scale=phi_posterior_scale,
            phi_prior_df=phi_prior_df,
            phi_prior_scale=phi_prior_scale,
            primary_term=primary_term(),
        )
    ]
    converged = False
    n_iterations = 0
    max_decrease = 0.0
    for iteration in range(1, max_iter + 1):
        expected_precision = residual_shape / residual_scale
        if workspace is None:
            factor_precision = phi_precision + np.einsum(
                "p,pkl->kl", expected_precision, loading_second, optimize=True
            )
            factor_covariance, _, _ = _solve_spd(factor_precision)
            weighted_loading = loading_mean * expected_precision[:, None]
            factor_mean = (x @ weighted_loading) @ factor_covariance
            factor_second_sum = n_count * factor_covariance + factor_mean.T @ factor_mean
            factor_response_sums = x.T @ factor_mean
        else:
            (
                factor_covariance,
                factor_mean,
                factor_second_sum,
                factor_response_sums,
            ) = workspace.factor_update(
                phi_precision,
                expected_precision,
                loading_second,
                loading_mean,
            )
        phi_posterior_scale = phi_prior_scale + factor_second_sum
        phi_scale_precision, _ = _invert_spd(phi_posterior_scale)
        phi_precision = phi_posterior_df * phi_scale_precision
        phi = phi_posterior_scale / (phi_posterior_df - k_count - 1.0)

        if free:
            cross_mean = loading_mean.copy()
            cross_mean[np.arange(p_count), membership] = 0.0
            primary_moments = _update_primary_loadings(
                membership,
                factor_second_sum,
                factor_response_sums,
                expected_precision,
                cross_mean,
                primary_loading_prior_mean,
                primary_loading_prior_variance,
            )
        state_probability, active_mean, active_variance = _update_row_states(
            membership=membership,
            factor_second_sum=factor_second_sum,
            factor_response_sums=factor_response_sums,
            expected_precision=expected_precision,
            overlap_alpha=overlap_alpha,
            overlap_beta=overlap_beta,
            target_alpha=target_alpha,
            active_cross_loading_variance=active_cross_loading_variance,
            primary_mean=primary_moments[0] if free else None,
        )
        overlap_alpha, overlap_beta, target_alpha = _update_global_loading_factors(
            membership=membership,
            state_probability=state_probability,
            prior_alpha=row_overlap_prior_alpha,
            prior_beta=row_overlap_prior_beta,
            target_concentration=target_dirichlet_concentration,
        )
        loading_mean, loading_second, loading_covariance = _row_sparse_loading_moments(
            membership,
            state_probability,
            active_mean,
            active_variance,
            primary_moments,
        )
        if workspace is None:
            rss = _residual_ss(
                x,
                factor_mean,
                factor_covariance,
                loading_mean,
                loading_second,
                loading_covariance,
            )
        else:
            rss = workspace.residual_ss(
                factor_mean,
                factor_covariance,
                loading_mean,
                loading_second,
                loading_covariance,
            )
        residual_shape = np.full(
            p_count, residual_prior_shape + 0.5 * n_count, dtype=float
        )
        residual_scale = residual_prior_scale + 0.5 * rss
        current = _row_sparse_elbo(
            x=x,
            membership=membership,
            factor_mean=factor_mean,
            factor_covariance=factor_covariance,
            factor_second_sum=factor_second_sum,
            loading_mean=loading_mean,
            loading_second=loading_second,
            loading_covariance=loading_covariance,
            state_probability=state_probability,
            active_mean=active_mean,
            active_variance=active_variance,
            overlap_alpha=overlap_alpha,
            overlap_beta=overlap_beta,
            target_alpha=target_alpha,
            row_overlap_prior_alpha=row_overlap_prior_alpha,
            row_overlap_prior_beta=row_overlap_prior_beta,
            target_dirichlet_concentration=target_dirichlet_concentration,
            active_cross_loading_variance=active_cross_loading_variance,
            residual_shape=residual_shape,
            residual_scale=residual_scale,
            residual_prior_shape=residual_prior_shape,
            residual_prior_scale=residual_prior_scale,
            phi_posterior_df=phi_posterior_df,
            phi_posterior_scale=phi_posterior_scale,
            phi_prior_df=phi_prior_df,
            phi_prior_scale=phi_prior_scale,
            primary_term=primary_term(),
        )
        previous = elbo_history[-1]
        elbo_history.append(current)
        decrease = max(0.0, previous - current)
        max_decrease = max(max_decrease, decrease)
        if decrease > elbo_decrease_tolerance * (1.0 + abs(previous)):
            raise RuntimeError(
                "2f: ELBO decreased beyond tolerance: "
                f"{previous:.12g} -> {current:.12g}"
            )
        relative_change = abs(current - previous) / (1.0 + abs(previous))
        n_iterations = iteration
        if iteration >= min_iter and relative_change <= tolerance:
            converged = True
            break

    posterior_started = perf_counter()
    row_overlap = 1.0 - state_probability[np.arange(p_count), membership]
    target_given_overlap = np.zeros_like(state_probability)
    np.divide(
        state_probability,
        row_overlap[:, None],
        out=target_given_overlap,
        where=row_overlap[:, None] > 0.0,
    )
    target_given_overlap[np.arange(p_count), membership] = 0.0
    factor_overlap_mean = overlap_alpha / (overlap_alpha + overlap_beta)
    target_mean = np.zeros_like(target_alpha)
    for primary in range(k_count):
        targets = np.arange(k_count) != primary
        target_mean[primary, targets] = target_alpha[primary, targets] / np.sum(
            target_alpha[primary, targets]
        )

    interval_rng = np.random.default_rng(
        int(factor_covariance_interval_seed) + 17_000_003
    )
    loading_draws = _draw_row_sparse_loadings(
        membership=membership,
        state_probability=state_probability,
        active_mean=active_mean,
        active_variance=active_variance,
        draws=int(factor_covariance_interval_draws),
        rng=interval_rng,
        primary_moments=primary_moments,
    )
    alpha = 0.5 * (1.0 - credible_interval_level)
    loading_lower = np.quantile(loading_draws, alpha, axis=0)
    loading_upper = np.quantile(loading_draws, 1.0 - alpha, axis=0)
    loading_sd = np.sqrt(
        np.maximum(np.diagonal(loading_covariance, axis1=1, axis2=2), 0.0)
    )
    quantile = float(stats.norm.ppf(1.0 - alpha))
    factor_sd = np.broadcast_to(
        np.sqrt(np.maximum(np.diag(factor_covariance), 0.0))[None, :],
        factor_mean.shape,
    ).copy()
    residual_mean = residual_scale / (residual_shape - 1.0)
    reconstruction = factor_mean @ loading_mean.T

    phi_rng = np.random.default_rng(int(factor_covariance_interval_seed))
    phi_draws = np.asarray(
        stats.invwishart.rvs(
            df=phi_posterior_df,
            scale=phi_posterior_scale,
            size=int(factor_covariance_interval_draws),
            random_state=phi_rng,
        ),
        dtype=float,
    )
    if phi_draws.ndim == 2:
        phi_draws = phi_draws[None, :, :]
    phi_sd = np.std(phi_draws, axis=0, ddof=1)
    phi_lower = np.quantile(phi_draws, alpha, axis=0)
    phi_upper = np.quantile(phi_draws, 1.0 - alpha, axis=0)
    phi_draw_diagonal = np.sqrt(
        np.maximum(np.diagonal(phi_draws, axis1=1, axis2=2), 1e-15)
    )
    correlation_draws = phi_draws / (
        phi_draw_diagonal[:, :, None] * phi_draw_diagonal[:, None, :]
    )
    diagonal = np.arange(k_count)
    correlation_draws[:, diagonal, diagonal] = 1.0

    inclusion_probability = state_probability.copy()
    inclusion_probability[np.arange(p_count), membership] = 1.0
    component_means = np.zeros((p_count, k_count, 2), dtype=float)
    component_variances = np.zeros_like(component_means)
    component_means[:, :, 1] = active_mean
    component_variances[:, :, 1] = active_variance
    component_means[np.arange(p_count), membership, 1] = loading_mean[
        np.arange(p_count), membership
    ]
    component_variances[np.arange(p_count), membership, :] = np.diagonal(
        loading_covariance, axis1=1, axis2=2
    )[np.arange(p_count), membership, None]

    posterior = _TwoPosterior(
        version="_fit_two_single_cross_free" if free else "_fit_two_single_cross",
        selected_indices=structure.selected_indices.copy(),
        block_sizes=np.asarray([block.size for block in structure.blocks], dtype=int),
        loading_mask=structure.loading_mask.copy(),
        loading_mean=loading_mean.copy(),
        loading_sd=loading_sd,
        loading_ci_lower=loading_lower,
        loading_ci_upper=loading_upper,
        loading_inclusion_probability=inclusion_probability,
        factor_score_mean=factor_mean.copy(),
        factor_score_sd=factor_sd,
        factor_score_ci_lower=factor_mean - quantile * factor_sd,
        factor_score_ci_upper=factor_mean + quantile * factor_sd,
        factor_prior_covariance=phi.copy(),
        residual_variance_mean=residual_mean,
        reconstruction_mean=reconstruction,
        reconstruction_rmse=float(np.sqrt(np.mean((x - reconstruction) ** 2))),
        data_mean=data_mean[structure.selected_indices].copy(),
        data_scale=data_scale[structure.selected_indices].copy(),
        credible_interval_level=float(credible_interval_level),
        factor_prior_variance=float(factor_prior_variance),
        cross_loading_inclusion_prior=float(
            row_overlap_prior_alpha / (row_overlap_prior_alpha + row_overlap_prior_beta)
        ),
        spike_variance=0.0,
        slab_variance=float(active_cross_loading_variance),
        estimate_primary_loadings=True,
        estimate_cross_loadings=True,
        enforce_loading_row_sums=not free,
        standardize_data=bool(standardize_data),
        member="2f",
        factor_covariance_has_posterior=True,
        factor_covariance_sd=phi_sd,
        factor_covariance_ci_lower=phi_lower,
        factor_covariance_ci_upper=phi_upper,
        factor_correlation_mean=np.mean(correlation_draws, axis=0),
        factor_correlation_sd=np.std(correlation_draws, axis=0, ddof=1),
        factor_correlation_ci_lower=np.quantile(correlation_draws, alpha, axis=0),
        factor_correlation_ci_upper=np.quantile(correlation_draws, 1.0 - alpha, axis=0),
        row_overlap_probability=row_overlap,
        row_state_probability=state_probability.copy(),
        target_probability_given_overlap=target_given_overlap,
        factor_overlap_probability_mean=factor_overlap_mean,
        target_probability_mean=target_mean,
    )
    diagnostics = _TwoDiagnostics(
        fit_runtime_seconds=posterior_started - started,
        interval_runtime_seconds=perf_counter() - posterior_started,
        elbo_history=np.asarray(elbo_history, dtype=float),
        converged=converged,
        n_iterations=n_iterations,
        max_iterations=int(max_iter),
        tolerance=float(tolerance),
        max_elbo_decrease=float(max_decrease),
        factor_covariance=factor_covariance.copy(),
        loading_covariance=loading_covariance.copy(),
        residual_shape=residual_shape.copy(),
        residual_scale=residual_scale.copy(),
        loading_component_means=component_means,
        loading_component_variances=component_variances,
        loading_slab_probability=inclusion_probability.copy(),
        factor_prior_covariance=phi.copy(),
        estimate_factor_covariance=True,
        estimate_primary_loadings=True,
        estimate_cross_loadings=True,
        enforce_loading_row_sums=not free,
        standardize_data=bool(standardize_data),
        member="2f",
        primary_loading_prior_mean=float(primary_loading_prior_mean) if free else None,
        primary_loading_prior_variance=float(primary_loading_prior_variance) if free else None,
        factor_covariance_posterior_df=float(phi_posterior_df),
        factor_covariance_posterior_scale=phi_posterior_scale.copy(),
        factor_covariance_prior_df=float(phi_prior_df),
        factor_covariance_prior_scale=phi_prior_scale.copy(),
        factor_covariance_interval_draws=int(factor_covariance_interval_draws),
        factor_covariance_interval_seed=int(factor_covariance_interval_seed),
        row_state_probability=state_probability.copy(),
        active_state_means=active_mean.copy(),
        active_state_variances=active_variance.copy(),
        overlap_beta_alpha=overlap_alpha.copy(),
        overlap_beta_beta=overlap_beta.copy(),
        target_dirichlet_alpha=target_alpha.copy(),
        **compute_backend.diagnostics(),
    )
    return _TwoResult(posterior=posterior, diagnostics=diagnostics)
