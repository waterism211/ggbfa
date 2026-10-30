"""Member 2f+s: at most one cross-loading per row and an exact per-community budget.

The loading row is either purely primary or has exactly one cross-loading.
Within every primary block, the joint variational distribution is restricted
to at most ``floor(max_overlap_fraction * block_size)`` active rows.  Exact
row marginal probabilities are obtained by a log-space forward/reverse dynamic
program in ``O(p_h * L_h)`` time, without enumerating subsets.
loading_normalization="free" adds a Gaussian
q(a_j) for the primary loading, independent of the gate and target, as in
``_two``; the cross-loading no longer changes the primary loading.
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
    _factor_full_elbo,
    _invert_spd,
    _prepare_data,
    _primary_loading_elbo,
    _residual_ss,
    _solve_spd,
    _update_primary_loadings,
    make_scfa_block_structure,
)
from ._two import (
    _TwoDiagnostics,
    _TwoPosterior,
    _TwoResult,
    _row_sparse_loading_moments,
)
from ._members import normalize_member


@dataclass
class _TwoSparsePosterior(_TwoPosterior):
    max_overlap_fraction: float = 0.10
    row_overlap_prior_probability: float = 0.05
    overlap_budget_by_factor: Optional[np.ndarray] = None
    expected_overlap_count_by_factor: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _TwoSparseDiagnostics(_TwoDiagnostics):
    overlap_budget_by_factor: Optional[np.ndarray] = None
    gate_log_odds: Optional[np.ndarray] = None
    gate_log_normalizer_by_factor: Optional[np.ndarray] = None
    conditional_target_probability: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _TwoSparseResult(_TwoResult):
    posterior: _TwoSparsePosterior
    diagnostics: _TwoSparseDiagnostics


def _validate_budget_inputs(
    *,
    max_iter: int,
    min_iter: int,
    tolerance: float,
    credible_interval_level: float,
    factor_prior_variance: float,
    row_overlap_prior_probability: float,
    max_overlap_fraction: float,
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
    if not 0.0 < row_overlap_prior_probability < max_overlap_fraction:
        raise ValueError(
            "row_overlap_prior_probability must be positive and smaller "
            "than max_overlap_fraction"
        )
    if not 0.0 < max_overlap_fraction < 1.0:
        raise ValueError("max_overlap_fraction must be in (0,1)")
    positive = {
        "factor_prior_variance": factor_prior_variance,
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


def _truncated_bernoulli_dp(
    log_odds: np.ndarray, budget: int
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Exact marginals for product Bernoulli weights truncated at a budget.

    Returns marginal inclusion probabilities, the log normalizer after removal
    of all no-state constants, and the forward table used for exact sampling.
    Reverse-mode differentiation of the dynamic program provides every
    marginal in the same ``O(p * budget)`` complexity as the normalizer.
    """

    value = np.asarray(log_odds, dtype=float).reshape(-1)
    if not np.all(np.isfinite(value)):
        raise ValueError("log_odds must be finite")
    if int(budget) != budget or budget < 0 or budget > value.size:
        raise ValueError("budget must be an integer in [0, p]")
    budget = int(budget)
    p_count = value.size
    forward = np.full((p_count + 1, budget + 1), -np.inf, dtype=float)
    forward[0, 0] = 0.0
    for variable in range(p_count):
        upper = min(variable + 1, budget)
        forward[variable + 1, 0] = forward[variable, 0]
        if upper:
            counts = np.arange(1, upper + 1)
            forward[variable + 1, counts] = np.logaddexp(
                forward[variable, counts],
                forward[variable, counts - 1] + value[variable],
            )
    log_normalizer = float(special.logsumexp(forward[p_count]))

    adjoint = np.exp(forward[p_count] - log_normalizer)
    marginal = np.zeros(p_count, dtype=float)
    for variable in range(p_count - 1, -1, -1):
        previous_adjoint = np.zeros(budget + 1, dtype=float)
        upper = min(variable + 1, budget)
        counts = np.arange(upper + 1)
        mass = adjoint[counts]
        denominator = forward[variable + 1, counts]
        no_log_weight = forward[variable, counts]
        valid_no = (mass > 0.0) & np.isfinite(denominator) & np.isfinite(no_log_weight)
        previous_adjoint[counts[valid_no]] += mass[valid_no] * np.exp(
            no_log_weight[valid_no] - denominator[valid_no]
        )
        if upper:
            yes_counts = counts[1:]
            yes_mass = mass[1:]
            yes_denominator = denominator[1:]
            yes_log_weight = forward[variable, yes_counts - 1] + value[variable]
            valid_yes = (
                (yes_mass > 0.0)
                & np.isfinite(yes_denominator)
                & np.isfinite(yes_log_weight)
            )
            contribution = yes_mass[valid_yes] * np.exp(
                yes_log_weight[valid_yes] - yes_denominator[valid_yes]
            )
            previous_adjoint[yes_counts[valid_yes] - 1] += contribution
            marginal[variable] = float(np.sum(contribution))
        adjoint = previous_adjoint
    return np.clip(marginal, 0.0, 1.0), log_normalizer, forward


def _sample_truncated_bernoulli(
    *,
    log_odds: np.ndarray,
    budget: int,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw exact joint gate configurations using the DP forward table."""

    _, log_normalizer, forward = _truncated_bernoulli_dp(log_odds, budget)
    count_probability = np.exp(forward[-1] - log_normalizer)
    remaining = rng.choice(np.arange(budget + 1), size=draws, p=count_probability)
    output = np.zeros((draws, len(log_odds)), dtype=bool)
    for variable in range(len(log_odds) - 1, -1, -1):
        eligible = remaining > 0
        if not np.any(eligible):
            continue
        draw_ids = np.flatnonzero(eligible)
        counts = remaining[draw_ids]
        log_probability = (
            forward[variable, counts - 1]
            + log_odds[variable]
            - forward[variable + 1, counts]
        )
        choose = rng.random(draw_ids.size) < np.exp(log_probability)
        chosen_draws = draw_ids[choose]
        output[chosen_draws, variable] = True
        remaining[chosen_draws] -= 1
    if np.any(np.sum(output, axis=1) > budget):
        raise RuntimeError("truncated Bernoulli sampler exceeded its budget")
    return output


def _update_target_alpha(
    *,
    membership: np.ndarray,
    state_probability: np.ndarray,
    concentration: float,
) -> np.ndarray:
    k_count = state_probability.shape[1]
    output = np.zeros((k_count, k_count), dtype=float)
    for primary in range(k_count):
        rows = membership == primary
        targets = np.arange(k_count) != primary
        output[primary, targets] = concentration + np.sum(
            state_probability[rows][:, targets], axis=0
        )
    return output


def _initial_budgeted_states(
    *,
    membership: np.ndarray,
    k_count: int,
    budgets: np.ndarray,
    prior_probability: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.zeros((membership.size, k_count), dtype=float)
    conditional_target = np.zeros_like(state)
    gate_log_odds = np.full(
        membership.size,
        special.logit(prior_probability),
        dtype=float,
    )
    log_normalizer = np.zeros(k_count, dtype=float)
    factors = np.arange(k_count)
    for primary in range(k_count):
        rows = np.flatnonzero(membership == primary)
        targets = factors[factors != primary]
        pip, log_normalizer[primary], _ = _truncated_bernoulli_dp(
            gate_log_odds[rows], int(budgets[primary])
        )
        conditional_target[np.ix_(rows, targets)] = 1.0 / targets.size
        state[rows, primary] = 1.0 - pip
        state[np.ix_(rows, targets)] = pip[:, None] / targets.size
    return state, conditional_target, gate_log_odds, log_normalizer


def _update_budgeted_states(
    *,
    membership: np.ndarray,
    budgets: np.ndarray,
    prior_probability: float,
    factor_second_sum: np.ndarray,
    factor_response_sums: np.ndarray,
    expected_precision: np.ndarray,
    target_alpha: np.ndarray,
    active_cross_loading_variance: float,
    primary_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_count = membership.size
    k_count = factor_second_sum.shape[0]
    state = np.zeros((p_count, k_count), dtype=float)
    conditional_target = np.zeros_like(state)
    active_mean = np.zeros_like(state)
    active_variance = np.zeros_like(state)
    gate_log_odds = np.zeros(p_count, dtype=float)
    log_normalizer = np.zeros(k_count, dtype=float)
    factor_diagonal = np.diag(factor_second_sum)
    factors = np.arange(k_count)
    prior_precision = 1.0 / active_cross_loading_variance
    prior_log_odds = special.logit(prior_probability)
    for primary in range(k_count):
        rows = np.flatnonzero(membership == primary)
        targets = factors[factors != primary]
        q_alpha = target_alpha[primary, targets]
        expected_log_target = special.digamma(q_alpha) - special.digamma(
            np.sum(q_alpha)
        )
        if primary_mean is None:
            factor_sum = (
                factor_diagonal[targets]
                + factor_diagonal[primary]
                - 2.0 * factor_second_sum[targets, primary]
            )
            factor_sum = np.maximum(factor_sum, 1e-15)
            response = (
                factor_response_sums[rows[:, None], targets[None, :]]
                - factor_response_sums[rows, primary][:, None]
                - factor_second_sum[targets, primary][None, :]
                + factor_second_sum[primary, primary]
            )
        else:  # free primary loadings: the cross-loading acts on its target only
            factor_sum = np.maximum(factor_diagonal[targets], 1e-15)
            response = (
                factor_response_sums[rows[:, None], targets[None, :]]
                - primary_mean[rows, None] * factor_second_sum[primary, targets][None, :]
            )
        tau = expected_precision[rows, None]
        variance = 1.0 / (tau * factor_sum[None, :] + prior_precision)
        natural = tau * response
        mean = variance * natural
        log_evidence = (
            0.5 * np.log(variance / active_cross_loading_variance)
            + 0.5 * natural * natural * variance
        )
        target_log_weight = expected_log_target[None, :] + log_evidence
        active_log_evidence = special.logsumexp(target_log_weight, axis=1)
        target_probability = np.exp(target_log_weight - active_log_evidence[:, None])
        local_log_odds = prior_log_odds + active_log_evidence
        pip, log_normalizer[primary], _ = _truncated_bernoulli_dp(
            local_log_odds, int(budgets[primary])
        )
        state[rows, primary] = 1.0 - pip
        state[np.ix_(rows, targets)] = pip[:, None] * target_probability
        conditional_target[np.ix_(rows, targets)] = target_probability
        active_mean[np.ix_(rows, targets)] = mean
        active_variance[np.ix_(rows, targets)] = variance
        gate_log_odds[rows] = local_log_odds
    return (
        state,
        conditional_target,
        active_mean,
        active_variance,
        gate_log_odds,
        log_normalizer,
    )


def _omega_elbo(
    target_alpha: np.ndarray,
    concentration: float,
) -> float:
    k_count = target_alpha.shape[0]
    value = 0.0
    for primary in range(k_count):
        targets = np.arange(k_count) != primary
        q_alpha = target_alpha[primary, targets]
        prior_alpha = np.full(targets.sum(), concentration, dtype=float)
        expected_log = special.digamma(q_alpha) - special.digamma(np.sum(q_alpha))
        value += float(
            special.gammaln(np.sum(prior_alpha))
            - np.sum(special.gammaln(prior_alpha))
            - special.gammaln(np.sum(q_alpha))
            + np.sum(special.gammaln(q_alpha))
            + np.sum((prior_alpha - q_alpha) * expected_log)
        )
    return value


def _budgeted_elbo(
    *,
    x: np.ndarray,
    membership: np.ndarray,
    budgets: np.ndarray,
    prior_probability: float,
    factor_mean: np.ndarray,
    factor_covariance: np.ndarray,
    factor_second_sum: np.ndarray,
    loading_mean: np.ndarray,
    loading_second: np.ndarray,
    loading_covariance: np.ndarray,
    state_probability: np.ndarray,
    conditional_target: np.ndarray,
    active_mean: np.ndarray,
    active_variance: np.ndarray,
    gate_log_odds: np.ndarray,
    gate_log_normalizer: np.ndarray,
    target_alpha: np.ndarray,
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
    value += _omega_elbo(target_alpha, target_dirichlet_concentration)

    log_prior_no = np.log1p(-prior_probability)
    factors = np.arange(state_probability.shape[1])
    for primary in range(state_probability.shape[1]):
        rows = np.flatnonzero(membership == primary)
        targets = factors[factors != primary]
        pip = np.sum(state_probability[np.ix_(rows, targets)], axis=1)
        log_prior_cap = float(
            stats.binom.logcdf(int(budgets[primary]), rows.size, prior_probability)
        )
        value += float(
            np.sum(pip * np.log(prior_probability) + (1.0 - pip) * log_prior_no)
            - log_prior_cap
        )
        value += float(gate_log_normalizer[primary] - np.sum(pip * gate_log_odds[rows]))

        q_alpha = target_alpha[primary, targets]
        expected_log_target = special.digamma(q_alpha) - special.digamma(
            np.sum(q_alpha)
        )
        edge_probability = state_probability[np.ix_(rows, targets)]
        target_probability = conditional_target[np.ix_(rows, targets)]
        value += float(np.sum(edge_probability * expected_log_target[None, :]))
        value += float(-np.sum(special.xlogy(edge_probability, target_probability)))
        means = active_mean[np.ix_(rows, targets)]
        variances = active_variance[np.ix_(rows, targets)]
        normal_prior_entropy = 0.5 * (
            1.0
            + np.log(variances / active_cross_loading_variance)
            - (variances + means * means) / active_cross_loading_variance
        )
        value += float(np.sum(edge_probability * normal_prior_entropy))
    if primary_term is not None:  # free primary loadings
        value += primary_term
    if not np.isfinite(value):
        raise RuntimeError("budgeted row-sparse ELBO is not finite")
    return value


def _draw_budgeted_loadings(
    *,
    membership: np.ndarray,
    budgets: np.ndarray,
    gate_log_odds: np.ndarray,
    conditional_target: np.ndarray,
    active_mean: np.ndarray,
    active_variance: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    primary_moments: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    p_count, k_count = conditional_target.shape
    output = np.zeros((draws, p_count, k_count), dtype=float)
    factors = np.arange(k_count)
    for primary in range(k_count):
        rows = np.flatnonzero(membership == primary)
        targets = factors[factors != primary]
        active = _sample_truncated_bernoulli(
            log_odds=gate_log_odds[rows],
            budget=int(budgets[primary]),
            draws=draws,
            rng=rng,
        )
        if primary_moments is None:
            output[:, rows, primary] = 1.0
        else:
            output[:, rows, primary] = rng.normal(
                primary_moments[0][rows], np.sqrt(primary_moments[1][rows]), size=(draws, rows.size)
            )
        for local_id, variable in enumerate(rows):
            draw_ids = np.flatnonzero(active[:, local_id])
            if draw_ids.size == 0:
                continue
            target_probability = conditional_target[variable, targets]
            chosen_target = rng.choice(
                targets, size=draw_ids.size, p=target_probability
            )
            for target in targets:
                selected_draws = draw_ids[chosen_target == target]
                if selected_draws.size == 0:
                    continue
                eta = rng.normal(
                    active_mean[variable, target],
                    np.sqrt(active_variance[variable, target]),
                    size=selected_draws.size,
                )
                output[selected_draws, variable, target] = eta
                if primary_moments is None:
                    output[selected_draws, variable, primary] = 1.0 - eta
    return output


def _fit_two_sparse(
    data: np.ndarray,
    blocks: Sequence[Block],
    *,
    member: str = "2f+s",
    max_iter: int = 1000,
    min_iter: int = 5,
    tolerance: float = 1e-6,
    credible_interval_level: float = 0.95,
    factor_prior_variance: float = 1.0,
    standardize_data: bool = True,
    row_overlap_prior_probability: float = 0.05,
    max_overlap_fraction: float = 0.10,
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
) -> _TwoSparseResult:
    """Fit the strict 10%-budget row-sparse variational model."""
    started = perf_counter()
    if loading_normalization not in ("row_sum", "free"):
        raise ValueError("loading_normalization must be 'row_sum' or 'free'")
    free = loading_normalization == "free"

    if normalize_member(member) != "2f+s":
        raise ValueError("_fit_two_sparse requires " "member='2f+s'")

    _validate_budget_inputs(
        max_iter=max_iter,
        min_iter=min_iter,
        tolerance=tolerance,
        credible_interval_level=credible_interval_level,
        factor_prior_variance=factor_prior_variance,
        row_overlap_prior_probability=row_overlap_prior_probability,
        max_overlap_fraction=max_overlap_fraction,
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
        raise ValueError("member 2f+s requires at least two communities")
    membership = np.asarray(structure.membership, dtype=int)
    block_sizes = np.bincount(membership, minlength=k_count)
    budgets = np.floor(max_overlap_fraction * block_sizes).astype(int)

    (
        state_probability,
        conditional_target,
        gate_log_odds,
        gate_log_normalizer,
    ) = _initial_budgeted_states(
        membership=membership,
        k_count=k_count,
        budgets=budgets,
        prior_probability=row_overlap_prior_probability,
    )
    active_mean = np.zeros_like(state_probability)
    active_variance = np.zeros_like(state_probability)
    cross_mask = ~np.asarray(structure.loading_mask, dtype=bool)
    active_variance[cross_mask] = active_cross_loading_variance
    target_alpha = _update_target_alpha(
        membership=membership,
        state_probability=state_probability,
        concentration=target_dirichlet_concentration,
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

    def evaluate_elbo() -> float:
        return _budgeted_elbo(
            x=x,
            membership=membership,
            budgets=budgets,
            prior_probability=row_overlap_prior_probability,
            factor_mean=factor_mean,
            factor_covariance=factor_covariance,
            factor_second_sum=factor_second_sum,
            loading_mean=loading_mean,
            loading_second=loading_second,
            loading_covariance=loading_covariance,
            state_probability=state_probability,
            conditional_target=conditional_target,
            active_mean=active_mean,
            active_variance=active_variance,
            gate_log_odds=gate_log_odds,
            gate_log_normalizer=gate_log_normalizer,
            target_alpha=target_alpha,
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
            primary_term=None if not free else _primary_loading_elbo(
                primary_moments[0], primary_moments[1],
                primary_loading_prior_mean, primary_loading_prior_variance),
        )

    elbo_history = [evaluate_elbo()]
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
        (
            state_probability,
            conditional_target,
            active_mean,
            active_variance,
            gate_log_odds,
            gate_log_normalizer,
        ) = _update_budgeted_states(
            membership=membership,
            budgets=budgets,
            prior_probability=row_overlap_prior_probability,
            factor_second_sum=factor_second_sum,
            factor_response_sums=factor_response_sums,
            expected_precision=expected_precision,
            target_alpha=target_alpha,
            active_cross_loading_variance=active_cross_loading_variance,
            primary_mean=primary_moments[0] if free else None,
        )
        target_alpha = _update_target_alpha(
            membership=membership,
            state_probability=state_probability,
            concentration=target_dirichlet_concentration,
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
        current = evaluate_elbo()
        previous = elbo_history[-1]
        elbo_history.append(current)
        decrease = max(0.0, previous - current)
        max_decrease = max(max_decrease, decrease)
        if decrease > elbo_decrease_tolerance * (1.0 + abs(previous)):
            raise RuntimeError(
                "2f+s: ELBO decreased beyond tolerance: "
                f"{previous:.12g} -> {current:.12g}"
            )
        relative_change = abs(current - previous) / (1.0 + abs(previous))
        n_iterations = iteration
        if iteration >= min_iter and relative_change <= tolerance:
            converged = True
            break

    posterior_started = perf_counter()
    row_overlap = 1.0 - state_probability[np.arange(p_count), membership]
    expected_by_factor = np.asarray(
        [np.sum(row_overlap[membership == h]) for h in range(k_count)]
    )
    if np.any(expected_by_factor > budgets + 1e-8):
        raise RuntimeError("posterior expected overlap count exceeded its budget")
    target_mean = np.zeros_like(target_alpha)
    for primary in range(k_count):
        targets = np.arange(k_count) != primary
        target_mean[primary, targets] = target_alpha[primary, targets] / np.sum(
            target_alpha[primary, targets]
        )

    interval_rng = np.random.default_rng(
        int(factor_covariance_interval_seed) + 17_000_003
    )
    loading_draws = _draw_budgeted_loadings(
        membership=membership,
        budgets=budgets,
        gate_log_odds=gate_log_odds,
        conditional_target=conditional_target,
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

    posterior = _TwoSparsePosterior(
        version="_fit_two_sparse_max_fraction_free" if free else "_fit_two_sparse_max_fraction",
        selected_indices=structure.selected_indices.copy(),
        block_sizes=block_sizes.copy(),
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
        cross_loading_inclusion_prior=float(row_overlap_prior_probability),
        spike_variance=0.0,
        slab_variance=float(active_cross_loading_variance),
        estimate_primary_loadings=True,
        estimate_cross_loadings=True,
        enforce_loading_row_sums=not free,
        standardize_data=bool(standardize_data),
        member="2f+s",
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
        target_probability_given_overlap=conditional_target.copy(),
        factor_overlap_probability_mean=expected_by_factor / block_sizes,
        target_probability_mean=target_mean,
        max_overlap_fraction=float(max_overlap_fraction),
        row_overlap_prior_probability=float(row_overlap_prior_probability),
        overlap_budget_by_factor=budgets.copy(),
        expected_overlap_count_by_factor=expected_by_factor,
    )
    diagnostics = _TwoSparseDiagnostics(
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
        member="2f+s",
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
        overlap_beta_alpha=None,
        overlap_beta_beta=None,
        target_dirichlet_alpha=target_alpha.copy(),
        overlap_budget_by_factor=budgets.copy(),
        gate_log_odds=gate_log_odds.copy(),
        gate_log_normalizer_by_factor=gate_log_normalizer.copy(),
        conditional_target_probability=conditional_target.copy(),
        **compute_backend.diagnostics(),
    )
    return _TwoSparseResult(posterior=posterior, diagnostics=diagnostics)
