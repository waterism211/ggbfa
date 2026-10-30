"""Shared variational machinery of the confirmatory step, and member mf.

The coordinate updates implement the confirmatory step of gg-BFA.  Members
2f and 2f+s live in ``_two`` and ``_two_sparse``, member 1f in ``_fixed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, special, stats

from ._backend import ComputeBackend, resolve_compute_backend
from ._members import normalize_member


Block = Sequence[int]


@dataclass(frozen=True)
class ScfaBlockStructure:
    """Nonoverlapping blocks represented as a primary loading mask."""

    blocks: Tuple[np.ndarray, ...]
    selected_indices: np.ndarray
    loading_mask: np.ndarray
    membership: np.ndarray


@dataclass
class CFAPosterior:
    """Approximate posterior summaries of one confirmatory fit."""

    version: str
    selected_indices: np.ndarray
    block_sizes: np.ndarray
    loading_mask: np.ndarray
    loading_mean: np.ndarray
    loading_sd: np.ndarray
    loading_ci_lower: np.ndarray
    loading_ci_upper: np.ndarray
    loading_inclusion_probability: np.ndarray
    factor_score_mean: np.ndarray
    factor_score_sd: np.ndarray
    factor_score_ci_lower: np.ndarray
    factor_score_ci_upper: np.ndarray
    factor_prior_covariance: np.ndarray
    residual_variance_mean: np.ndarray
    reconstruction_mean: np.ndarray
    reconstruction_rmse: float
    data_mean: np.ndarray
    data_scale: np.ndarray
    credible_interval_level: float
    factor_prior_variance: float
    cross_loading_inclusion_prior: float
    spike_variance: float
    slab_variance: float
    estimate_primary_loadings: bool
    estimate_cross_loadings: bool
    enforce_loading_row_sums: bool
    standardize_data: bool
    member: str = "mf"
    factor_covariance_has_posterior: bool = False
    factor_covariance_sd: Optional[np.ndarray] = None
    factor_covariance_ci_lower: Optional[np.ndarray] = None
    factor_covariance_ci_upper: Optional[np.ndarray] = None
    factor_correlation_mean: Optional[np.ndarray] = None
    factor_correlation_sd: Optional[np.ndarray] = None
    factor_correlation_ci_lower: Optional[np.ndarray] = None
    factor_correlation_ci_upper: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CFADiagnostics:
    """ELBO trace and analytic variational-factor parameters."""

    elbo_history: np.ndarray
    converged: bool
    n_iterations: int
    max_iterations: int
    tolerance: float
    max_elbo_decrease: float
    factor_covariance: np.ndarray
    loading_covariance: np.ndarray
    residual_shape: np.ndarray
    residual_scale: np.ndarray
    loading_component_means: np.ndarray
    loading_component_variances: np.ndarray
    loading_slab_probability: np.ndarray
    factor_prior_covariance: np.ndarray
    estimate_factor_covariance: bool
    estimate_primary_loadings: bool
    estimate_cross_loadings: bool
    enforce_loading_row_sums: bool
    standardize_data: bool
    member: str = "mf"
    factor_covariance_posterior_df: Optional[float] = None
    factor_covariance_posterior_scale: Optional[np.ndarray] = None
    factor_covariance_prior_df: Optional[float] = None
    factor_covariance_prior_scale: Optional[np.ndarray] = None
    factor_covariance_interval_draws: int = 0
    factor_covariance_interval_seed: Optional[int] = None
    fit_runtime_seconds: float = 0.0
    interval_runtime_seconds: float = 0.0
    # For fixed loadings, psi_j is the block variable v_{h_j}. None means
    # independent variable-level residual factors.
    residual_group_index: Optional[np.ndarray] = None
    requested_device: str = "auto"
    resolved_device: str = "cpu"
    compute_backend: str = "numpy/scipy"
    actual_gpu_count: int = 0
    device_reason: str = "cpu_requested"
    gpu_device_id: Optional[int] = None
    gpu_name: Optional[str] = None
    cupy_version: Optional[str] = None
    cuda_runtime_version: Optional[int] = None
    gpu_operation_counts: Optional[dict] = None
    # loading_normalization="free" only
    primary_loading_prior_mean: Optional[float] = None
    primary_loading_prior_variance: Optional[float] = None


@dataclass(frozen=True)
class CFAResult:
    """Posterior summaries and optimization diagnostics of one confirmatory fit."""

    posterior: CFAPosterior
    diagnostics: CFADiagnostics


def _draw_loading_components(
    *,
    loading_mask: np.ndarray,
    probabilities: np.ndarray,
    component_means: np.ndarray,
    component_variances: np.ndarray,
    enforce_loading_row_sums: bool,
    draws: int,
    rng: np.random.Generator,
    return_indicators: bool = False,
) -> np.ndarray:
    """Draw loading components under the row-sum constraint or with free primary loadings."""

    components = (
        rng.random((draws,) + probabilities.shape) < probabilities[None, :, :]
    ).astype(int)
    chosen_means = np.take_along_axis(
        component_means[None, :, :, :],
        components[:, :, :, None],
        axis=3,
    )[..., 0]
    chosen_variances = np.take_along_axis(
        component_variances[None, :, :, :],
        components[:, :, :, None],
        axis=3,
    )[..., 0]
    loading = rng.normal(chosen_means, np.sqrt(chosen_variances))
    if enforce_loading_row_sums:
        primary_mask = np.asarray(loading_mask, dtype=bool)
        for variable_id in range(primary_mask.shape[0]):
            primary = int(np.flatnonzero(primary_mask[variable_id])[0])
            cross = ~primary_mask[variable_id]
            loading[:, variable_id, primary] = 1.0 - np.sum(
                loading[:, variable_id, cross], axis=1
            )
    if return_indicators:
        components[:, np.asarray(loading_mask, dtype=bool)] = 0
        return loading, components
    return loading


def free_primary_moments(result, rows=None):
    """(mean, variance) of the free primary loadings, or None under the row-sum normalization."""

    diagnostics = result.diagnostics
    if getattr(diagnostics, "enforce_loading_row_sums", True):
        return None
    mask = np.asarray(result.posterior.loading_mask, dtype=bool)
    ids = np.arange(mask.shape[0]) if rows is None else np.asarray(rows, dtype=int)
    primary = np.argmax(mask[ids], axis=1)
    means = np.asarray(diagnostics.loading_component_means, dtype=float)[ids, primary, 1]
    variances = np.asarray(diagnostics.loading_component_variances, dtype=float)[ids, primary, 1]
    return means, variances


def _draw_loading_posterior(
    result: CFAResult,
    *,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``Lambda`` while preserving any exact row-sum constraint."""

    if draws < 1:
        raise ValueError("draws must be positive")
    posterior = result.posterior
    diagnostics = result.diagnostics
    member = normalize_member(
        getattr(diagnostics, "member", "mf")
    )
    if member == "1f" and getattr(diagnostics, "enforce_loading_row_sums", True):
        return np.broadcast_to(posterior.loading_mask, (draws,) + posterior.loading_mask.shape).copy()
    primary = free_primary_moments(result)
    if member == "2f":
        from ._two import _draw_row_sparse_loadings

        return _draw_row_sparse_loadings(
            membership=np.argmax(posterior.loading_mask, axis=1),
            state_probability=np.asarray(
                diagnostics.row_state_probability, dtype=float
            ),
            active_mean=np.asarray(diagnostics.active_state_means, dtype=float),
            active_variance=np.asarray(diagnostics.active_state_variances, dtype=float),
            draws=draws,
            rng=rng,
            primary_moments=primary,
        )
    if member == "2f+s":
        from ._two_sparse import _draw_budgeted_loadings

        return _draw_budgeted_loadings(
            membership=np.argmax(posterior.loading_mask, axis=1),
            budgets=np.asarray(diagnostics.overlap_budget_by_factor, dtype=int),
            gate_log_odds=np.asarray(diagnostics.gate_log_odds, dtype=float),
            conditional_target=np.asarray(
                diagnostics.conditional_target_probability, dtype=float
            ),
            active_mean=np.asarray(diagnostics.active_state_means, dtype=float),
            active_variance=np.asarray(diagnostics.active_state_variances, dtype=float),
            draws=draws,
            rng=rng,
            primary_moments=primary,
        )
    probabilities = np.asarray(diagnostics.loading_slab_probability, dtype=float)
    component_means = np.asarray(diagnostics.loading_component_means, dtype=float)
    component_variances = np.asarray(
        diagnostics.loading_component_variances, dtype=float
    )
    return _draw_loading_components(
        loading_mask=posterior.loading_mask,
        probabilities=probabilities,
        component_means=component_means,
        component_variances=component_variances,
        enforce_loading_row_sums=diagnostics.enforce_loading_row_sums,
        draws=draws,
        rng=rng,
    )


def standardize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-standardize a complete matrix using ``ddof=0``."""

    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("data must have shape (n>=3, p>=2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must be finite and complete")
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    if np.any(scale <= 0.0):
        raise ValueError("data columns must have positive variance")
    return (x - mean) / scale, mean, scale


def _prepare_data(
    data: np.ndarray, *, standardize: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center columns and optionally rescale them to unit sample SD."""

    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("data must have shape (n>=3, p>=2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must be finite and complete")
    mean = x.mean(axis=0)
    if standardize:
        scale = x.std(axis=0, ddof=0)
        if np.any(scale <= 0.0):
            raise ValueError("data columns must have positive variance")
    else:
        scale = np.ones(x.shape[1], dtype=float)
    return (x - mean) / scale, mean, scale


def make_scfa_block_structure(
    blocks: Sequence[Block], n_variables: int
) -> ScfaBlockStructure:
    """Validate blocks and construct their primary loading mask."""

    parsed = []
    seen = set()
    for block in blocks:
        values = np.asarray(sorted(set(int(node) for node in block)), dtype=int)
        if values.size == 0:
            continue
        if values.min() < 0 or values.max() >= n_variables:
            raise ValueError("a community contains an out-of-range variable index")
        overlap = seen.intersection(int(node) for node in values)
        if overlap:
            raise ValueError("communities must not overlap")
        seen.update(int(node) for node in values)
        parsed.append(values)
    if not parsed:
        raise ValueError("at least one nonempty community is required")

    selected = np.concatenate(parsed).astype(int)
    loading_mask = np.zeros((selected.size, len(parsed)), dtype=float)
    membership = np.empty(selected.size, dtype=int)
    start = 0
    for factor_id, block in enumerate(parsed):
        stop = start + block.size
        loading_mask[start:stop, factor_id] = 1.0
        membership[start:stop] = factor_id
        start = stop
    return ScfaBlockStructure(
        blocks=tuple(block.copy() for block in parsed),
        selected_indices=selected,
        loading_mask=loading_mask,
        membership=membership,
    )


def _validate_inputs(
    *,
    max_iter: int,
    min_iter: int,
    tolerance: float,
    credible_interval_level: float,
    factor_prior_variance: float,
    loading_prior_variance: float,
    cross_loading_inclusion_prior: float,
    spike_variance: float,
    slab_variance: float,
    residual_prior_shape: float,
    residual_prior_scale: float,
    elbo_decrease_tolerance: float,
    mixture_interval_method: str,
    factor_covariance_prior_strength: float,
) -> None:
    if max_iter <= 0 or min_iter <= 0 or min_iter > max_iter:
        raise ValueError("require 0 < min_iter <= max_iter")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not 0.0 < credible_interval_level < 1.0:
        raise ValueError("credible_interval_level must be in (0,1)")
    if factor_prior_variance <= 0.0:
        raise ValueError("factor_prior_variance must be positive")
    if not 0.0 < cross_loading_inclusion_prior < 1.0:
        raise ValueError("cross_loading_inclusion_prior must be in (0,1)")
    if spike_variance <= 0.0 or slab_variance <= 0.0:
        raise ValueError("spike and slab variances must be positive")
    if spike_variance >= slab_variance:
        raise ValueError("spike_variance must be smaller than slab_variance")
    if residual_prior_shape <= 0.0 or residual_prior_scale <= 0.0:
        raise ValueError("residual prior parameters must be positive")
    if elbo_decrease_tolerance < 0.0:
        raise ValueError("elbo_decrease_tolerance must be nonnegative")
    if mixture_interval_method not in {"quantile", "normal_approx"}:
        raise ValueError("mixture_interval_method must be quantile or normal_approx")
    if factor_covariance_prior_strength < 0.0:
        raise ValueError("factor_covariance_prior_strength must be nonnegative")


def _solve_spd(
    precision: np.ndarray, natural_mean: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    symmetric = 0.5 * (precision + precision.T)
    try:
        chol = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("variational precision is not positive definite") from error
    identity = np.eye(symmetric.shape[0], dtype=float)
    covariance = np.linalg.solve(chol.T, np.linalg.solve(chol, identity))
    covariance = 0.5 * (covariance + covariance.T)
    mean = None
    if natural_mean is not None:
        mean = np.linalg.solve(chol.T, np.linalg.solve(chol, natural_mean))
    logdet_covariance = float(-2.0 * np.log(np.diag(chol)).sum())
    return covariance, mean, logdet_covariance


def _invert_spd(covariance: np.ndarray) -> Tuple[np.ndarray, float]:
    precision, _, logdet_precision = _solve_spd(covariance)
    return precision, -float(logdet_precision)


def _initialize_loadings(
    structure: ScfaBlockStructure,
    loading_prior_mean: float,
    loading_prior_variance: float,
    inclusion_prior: float,
    spike_variance: float,
    slab_variance: float,
    estimate_primary_loadings: bool,
    estimate_cross_loadings: bool,
):
    p, k = structure.loading_mask.shape
    mean = np.zeros((p, k), dtype=float)
    second_diagonal = np.zeros((p, k), dtype=float)
    probability = np.zeros((p, k), dtype=float)
    component_mean = np.zeros((p, k, 2), dtype=float)
    component_variance = np.zeros((p, k, 2), dtype=float)
    if estimate_cross_loadings:
        second_diagonal[:, :] = (
            1.0 - inclusion_prior
        ) * spike_variance + inclusion_prior * slab_variance
        probability[:, :] = inclusion_prior
        component_variance[:, :, 0] = spike_variance
        component_variance[:, :, 1] = slab_variance
    for variable_id, primary in enumerate(structure.membership):
        primary = int(primary)
        mean[variable_id, primary] = loading_prior_mean
        second_diagonal[variable_id, primary] = loading_prior_mean**2
        if estimate_primary_loadings:
            second_diagonal[variable_id, primary] += loading_prior_variance
        probability[variable_id, primary] = 1.0
        component_mean[variable_id, primary, 1] = loading_prior_mean
        if estimate_primary_loadings:
            component_variance[variable_id, primary, :] = loading_prior_variance
    return mean, second_diagonal, probability, component_mean, component_variance


def _constrained_loading_moments(
    structure: ScfaBlockStructure,
    mean: np.ndarray,
    second_diagonal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map independent cross-loadings to rows with signed sum one.

    For row ``j`` with primary factor ``h``, the free coordinates are the
    cross-loadings ``eta[j, k]`` and
    ``lambda[j, h] = 1 - sum_{k != h} eta[j, k]``. The affine map induces a
    full, singular within-row covariance whose null direction is the vector of
    ones.
    """

    p, k_count = mean.shape
    row_ids = np.arange(p)
    factor_ids = np.arange(k_count)
    primary_ids = np.asarray(structure.membership, dtype=int)
    row_mean = mean.copy()
    eta_variance = np.maximum(second_diagonal - row_mean**2, 0.0)
    eta_variance[row_ids, primary_ids] = 0.0
    row_mean[row_ids, primary_ids] = 1.0 - (
        np.sum(row_mean, axis=1) - row_mean[row_ids, primary_ids]
    )

    covariance = np.zeros((p, k_count, k_count), dtype=float)
    covariance[:, factor_ids, factor_ids] = eta_variance
    covariance[row_ids, primary_ids, :] = -eta_variance
    covariance[row_ids, :, primary_ids] = -eta_variance
    covariance[row_ids, primary_ids, primary_ids] = np.sum(eta_variance, axis=1)
    second = row_mean[:, :, None] * row_mean[:, None, :] + covariance

    mean[:] = row_mean
    second_diagonal[:] = np.diagonal(second, axis1=1, axis2=2)
    return second, covariance


def _record_constrained_primary_components(
    structure: ScfaBlockStructure,
    loading_mean: np.ndarray,
    loading_covariance: np.ndarray,
    slab_probability: np.ndarray,
    component_means: np.ndarray,
    component_variances: np.ndarray,
) -> None:
    """Record derived-primary marginals while retaining cross components."""

    row_ids = np.arange(loading_mean.shape[0])
    primary_ids = np.asarray(structure.membership, dtype=int)
    variance = np.maximum(loading_covariance[row_ids, primary_ids, primary_ids], 0.0)
    slab_probability[row_ids, primary_ids] = 1.0
    component_means[row_ids, primary_ids, 0] = 0.0
    component_means[row_ids, primary_ids, 1] = loading_mean[row_ids, primary_ids]
    component_variances[row_ids, primary_ids, :] = variance[:, None]


def _primary_loading_elbo(mean, variance, prior_mean, prior_variance) -> float:
    """E[log N(lambda_jh; m0, v0)] + entropy of the Gaussian q(lambda_jh) (loading_normalization="free")."""

    return float(
        np.sum(
            -0.5 * (np.log(2.0 * np.pi * prior_variance) + (variance + (mean - prior_mean) ** 2) / prior_variance)
            + 0.5 * (1.0 + np.log(2.0 * np.pi * variance))
        )
    )


def _update_primary_loadings(
    membership: np.ndarray,
    factor_second_sum: np.ndarray,
    factor_response_sums: np.ndarray,
    expected_precision: np.ndarray,
    cross_mean: np.ndarray,
    prior_mean: float,
    prior_variance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian q(lambda_jh) of free primary loadings given E[lambda_jk], k != h (zero at h)."""

    rows = np.arange(membership.size)
    other = (cross_mean @ factor_second_sum)[rows, membership]
    response = factor_response_sums[rows, membership] - other
    precision = (
        expected_precision * factor_second_sum[membership, membership]
        + 1.0 / prior_variance
    )
    variance = 1.0 / precision
    mean = variance * (
        expected_precision * response + prior_mean / prior_variance
    )
    return mean, variance


def _free_loading_moments(
    mean: np.ndarray, second_diagonal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """E[lambda lambda'] and Cov(lambda) per row for independent coordinates (free normalization)."""

    p, k_count = mean.shape
    variance = np.maximum(second_diagonal - mean * mean, 0.0)
    covariance = np.zeros((p, k_count, k_count), dtype=float)
    covariance[:, np.arange(k_count), np.arange(k_count)] = variance
    second = mean[:, :, None] * mean[:, None, :] + covariance
    return second, covariance


def _update_loadings_free(
    structure: ScfaBlockStructure,
    factor_second_sum: np.ndarray,
    factor_response_sums: np.ndarray,
    expected_precision: np.ndarray,
    loading_mean: np.ndarray,
    loading_second_diagonal: np.ndarray,
    slab_probability: np.ndarray,
    component_means: np.ndarray,
    component_variances: np.ndarray,
    inclusion_prior: float,
    spike_variance: float,
    slab_variance: float,
    prior_mean: float,
    prior_variance: float,
) -> None:
    """Update free primary loadings, then spike-and-slab cross-loadings, in original coordinates."""

    membership = np.asarray(structure.membership, dtype=int)
    rows = np.arange(membership.size)
    primary_mask = np.asarray(structure.loading_mask, dtype=bool)
    mean, variance = _update_primary_loadings(
        membership,
        factor_second_sum,
        factor_response_sums,
        expected_precision,
        np.where(primary_mask, 0.0, loading_mean),
        prior_mean,
        prior_variance,
    )
    loading_mean[rows, membership] = mean
    loading_second_diagonal[rows, membership] = variance + mean * mean
    slab_probability[rows, membership] = 1.0
    component_means[rows, membership, 0] = 0.0
    component_means[rows, membership, 1] = mean
    component_variances[rows, membership, :] = variance[:, None]

    spike_prior_precision = 1.0 / spike_variance
    slab_prior_precision = 1.0 / slab_variance
    log_spike_prior = np.log1p(-inclusion_prior)
    log_slab_prior = np.log(inclusion_prior)
    for factor_id in range(loading_mean.shape[1]):
        row_ids = np.flatnonzero(membership != factor_id)
        if row_ids.size == 0:
            continue
        tau = expected_precision[row_ids]
        factor_sum = max(float(factor_second_sum[factor_id, factor_id]), 1e-15)
        response = (
            factor_response_sums[row_ids, factor_id]
            - loading_mean[row_ids] @ factor_second_sum[:, factor_id]
            + loading_mean[row_ids, factor_id] * factor_sum
        )
        natural = tau * response
        spike_posterior_variance = 1.0 / (tau * factor_sum + spike_prior_precision)
        slab_posterior_variance = 1.0 / (tau * factor_sum + slab_prior_precision)
        spike_posterior_mean = spike_posterior_variance * natural
        slab_posterior_mean = slab_posterior_variance * natural
        log_spike_weight = (
            log_spike_prior
            - 0.5 * np.log(spike_variance)
            + 0.5 * np.log(spike_posterior_variance)
            + 0.5 * natural * natural * spike_posterior_variance
        )
        log_slab_weight = (
            log_slab_prior
            - 0.5 * np.log(slab_variance)
            + 0.5 * np.log(slab_posterior_variance)
            + 0.5 * natural * natural * slab_posterior_variance
        )
        slab = special.expit(log_slab_weight - log_spike_weight)
        spike = 1.0 - slab
        loading_mean[row_ids, factor_id] = (
            spike * spike_posterior_mean + slab * slab_posterior_mean
        )
        loading_second_diagonal[row_ids, factor_id] = spike * (
            spike_posterior_variance + spike_posterior_mean**2
        ) + slab * (slab_posterior_variance + slab_posterior_mean**2)
        slab_probability[row_ids, factor_id] = slab
        component_means[row_ids, factor_id, 0] = spike_posterior_mean
        component_means[row_ids, factor_id, 1] = slab_posterior_mean
        component_variances[row_ids, factor_id, 0] = spike_posterior_variance
        component_variances[row_ids, factor_id, 1] = slab_posterior_variance


def _update_loadings_unit_row_sum(
    structure: ScfaBlockStructure,
    factor_second_sum: np.ndarray,
    factor_response_sums: np.ndarray,
    expected_precision: np.ndarray,
    loading_mean: np.ndarray,
    loading_second_diagonal: np.ndarray,
    slab_probability: np.ndarray,
    component_means: np.ndarray,
    component_variances: np.ndarray,
    inclusion_prior: float,
    spike_variance: float,
    slab_variance: float,
    estimate_cross_loadings: bool,
) -> None:
    """Update free cross-loadings under the exact row-sum constraint."""

    k_count = loading_mean.shape[1]
    primary_mask = np.asarray(structure.loading_mask, dtype=bool)
    if not estimate_cross_loadings:
        cross_mask = ~primary_mask
        loading_mean[cross_mask] = 0.0
        loading_second_diagonal[cross_mask] = 0.0
        slab_probability[cross_mask] = 0.0
        component_means[cross_mask] = 0.0
        component_variances[cross_mask] = 0.0
        return

    spike_prior_precision = 1.0 / spike_variance
    slab_prior_precision = 1.0 / slab_variance
    log_spike_prior = np.log1p(-inclusion_prior)
    log_slab_prior = np.log(inclusion_prior)
    factor_diagonal = np.diag(factor_second_sum)
    factor_ids = np.arange(k_count)
    membership = np.asarray(structure.membership, dtype=int)
    for primary in range(k_count):
        row_ids = np.flatnonzero(membership == primary)
        if row_ids.size == 0:
            continue
        tau = expected_precision[row_ids]
        cross_ids = factor_ids[factor_ids != primary]
        transformed_second = (
            factor_second_sum[np.ix_(cross_ids, cross_ids)]
            - factor_second_sum[cross_ids, primary][:, None]
            - factor_second_sum[primary, cross_ids][None, :]
            + factor_second_sum[primary, primary]
        )
        base_response = (
            factor_response_sums[row_ids[:, None], cross_ids[None, :]]
            - factor_response_sums[row_ids, primary][:, None]
            - factor_second_sum[primary, cross_ids][None, :]
            + factor_second_sum[primary, primary]
        )
        for cross_position, factor_id in enumerate(cross_ids):
            factor_sum = float(transformed_second[cross_position, cross_position])
            factor_sum = max(factor_sum, 1e-15)
            response = (
                base_response[:, cross_position]
                - loading_mean[row_ids[:, None], cross_ids[None, :]]
                @ transformed_second[:, cross_position]
                + loading_mean[row_ids, factor_id] * factor_sum
            )

            natural = tau * response
            spike_posterior_variance = 1.0 / (tau * factor_sum + spike_prior_precision)
            slab_posterior_variance = 1.0 / (tau * factor_sum + slab_prior_precision)
            spike_posterior_mean = spike_posterior_variance * natural
            slab_posterior_mean = slab_posterior_variance * natural
            log_spike_weight = (
                log_spike_prior
                - 0.5 * np.log(spike_variance)
                + 0.5 * np.log(spike_posterior_variance)
                + 0.5 * natural * natural * spike_posterior_variance
            )
            log_slab_weight = (
                log_slab_prior
                - 0.5 * np.log(slab_variance)
                + 0.5 * np.log(slab_posterior_variance)
                + 0.5 * natural * natural * slab_posterior_variance
            )
            slab = special.expit(log_slab_weight - log_spike_weight)
            spike = 1.0 - slab
            mean = spike * spike_posterior_mean + slab * slab_posterior_mean
            second = spike * (
                spike_posterior_variance + spike_posterior_mean**2
            ) + slab * (slab_posterior_variance + slab_posterior_mean**2)
            loading_mean[row_ids, factor_id] = mean
            loading_second_diagonal[row_ids, factor_id] = second
            slab_probability[row_ids, factor_id] = slab
            component_means[row_ids, factor_id, 0] = spike_posterior_mean
            component_means[row_ids, factor_id, 1] = slab_posterior_mean
            component_variances[row_ids, factor_id, 0] = spike_posterior_variance
            component_variances[row_ids, factor_id, 1] = slab_posterior_variance


def _residual_ss(
    x: np.ndarray,
    factor_mean: np.ndarray,
    factor_covariance: np.ndarray,
    loading_mean: np.ndarray,
    loading_second: np.ndarray,
    loading_covariance: np.ndarray,
) -> np.ndarray:
    mean_residual = x - factor_mean @ loading_mean.T
    loading_uncertainty = np.einsum(
        "kl,pkl->p", factor_mean.T @ factor_mean, loading_covariance, optimize=True
    )
    factor_uncertainty = x.shape[0] * np.einsum(
        "kl,pkl->p", factor_covariance, loading_second, optimize=True
    )
    return (
        np.sum(mean_residual * mean_residual, axis=0)
        + loading_uncertainty
        + factor_uncertainty
    )


def _factor_elbo(
    n: int,
    factor_covariance: np.ndarray,
    factor_second_sum: np.ndarray,
    factor_prior_covariance: np.ndarray,
) -> float:
    sign, logdet_factor_covariance = np.linalg.slogdet(factor_covariance)
    if sign <= 0:
        raise RuntimeError("factor variational covariance is not positive definite")
    prior_precision, logdet_prior = _invert_spd(factor_prior_covariance)
    k = factor_covariance.shape[0]
    entropy = 0.5 * n * (k * (1.0 + np.log(2.0 * np.pi)) + logdet_factor_covariance)
    prior = -0.5 * n * (k * np.log(2.0 * np.pi) + logdet_prior) - 0.5 * np.trace(
        prior_precision @ factor_second_sum
    )
    return float(prior + entropy)


def _inverse_wishart_expected_log_precision(df: float, scale: np.ndarray) -> float:
    """Return E[log |Phi^-1|] for an inverse-Wishart variate."""

    matrix = np.asarray(scale, dtype=float)
    k = matrix.shape[0]
    sign, logdet_scale = np.linalg.slogdet(matrix)
    if sign <= 0 or df <= k - 1:
        raise RuntimeError("inverse-Wishart parameters are not valid")
    arguments = 0.5 * (df + 1.0 - np.arange(1, k + 1))
    return float(np.sum(special.digamma(arguments)) + k * np.log(2.0) - logdet_scale)


def _inverse_wishart_expected_log_density(
    *,
    density_df: float,
    density_scale: np.ndarray,
    expected_logdet_covariance: float,
    expected_precision: np.ndarray,
) -> float:
    """Expected log inverse-Wishart density under supplied moments."""

    matrix = np.asarray(density_scale, dtype=float)
    k = matrix.shape[0]
    sign, logdet_scale = np.linalg.slogdet(matrix)
    if sign <= 0 or density_df <= k - 1:
        raise RuntimeError("inverse-Wishart density parameters are not valid")
    normalizer = (
        0.5 * density_df * logdet_scale
        - 0.5 * density_df * k * np.log(2.0)
        - special.multigammaln(0.5 * density_df, k)
    )
    return float(
        normalizer
        - 0.5 * (density_df + k + 1.0) * expected_logdet_covariance
        - 0.5 * np.trace(matrix @ expected_precision)
    )


def _factor_full_elbo(
    n: int,
    factor_covariance: np.ndarray,
    factor_second_sum: np.ndarray,
    *,
    phi_posterior_df: float,
    phi_posterior_scale: np.ndarray,
    phi_prior_df: float,
    phi_prior_scale: np.ndarray,
) -> float:
    """Factor-score and inverse-Wishart q(Phi) ELBO terms."""

    sign, logdet_factor_covariance = np.linalg.slogdet(factor_covariance)
    if sign <= 0:
        raise RuntimeError("factor variational covariance is not positive definite")
    k = factor_covariance.shape[0]
    posterior_scale_precision, _ = _invert_spd(phi_posterior_scale)
    expected_phi_precision = phi_posterior_df * posterior_scale_precision
    expected_logdet_precision = _inverse_wishart_expected_log_precision(
        phi_posterior_df, phi_posterior_scale
    )
    expected_logdet_phi = -expected_logdet_precision

    expected_factor_prior = 0.5 * n * (
        expected_logdet_precision - k * np.log(2.0 * np.pi)
    ) - 0.5 * np.trace(expected_phi_precision @ factor_second_sum)
    factor_entropy = (
        0.5 * n * (k * (1.0 + np.log(2.0 * np.pi)) + logdet_factor_covariance)
    )
    expected_phi_prior = _inverse_wishart_expected_log_density(
        density_df=phi_prior_df,
        density_scale=phi_prior_scale,
        expected_logdet_covariance=expected_logdet_phi,
        expected_precision=expected_phi_precision,
    )
    expected_log_q_phi = _inverse_wishart_expected_log_density(
        density_df=phi_posterior_df,
        density_scale=phi_posterior_scale,
        expected_logdet_covariance=expected_logdet_phi,
        expected_precision=expected_phi_precision,
    )
    return float(
        expected_factor_prior + factor_entropy + expected_phi_prior - expected_log_q_phi
    )


def _elbo(
    x: np.ndarray,
    structure: ScfaBlockStructure,
    factor_mean: np.ndarray,
    factor_covariance: np.ndarray,
    factor_second_sum: np.ndarray,
    loading_mean: np.ndarray,
    loading_second: np.ndarray,
    loading_covariance: np.ndarray,
    component_means: np.ndarray,
    component_variances: np.ndarray,
    slab_probability: np.ndarray,
    residual_shape: np.ndarray,
    residual_scale: np.ndarray,
    factor_prior_covariance: np.ndarray,
    loading_prior_mean: float,
    loading_prior_variance: float,
    inclusion_prior: float,
    spike_variance: float,
    slab_variance: float,
    residual_prior_shape: float,
    residual_prior_scale: float,
    estimate_primary_loadings: bool,
    estimate_cross_loadings: bool,
    enforce_loading_row_sums: bool,
    phi_posterior_df: Optional[float] = None,
    phi_posterior_scale: Optional[np.ndarray] = None,
    phi_prior_df: Optional[float] = None,
    phi_prior_scale: Optional[np.ndarray] = None,
) -> float:
    n = x.shape[0]
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
            -0.5 * n * np.log(2.0 * np.pi)
            - 0.5 * n * expected_log_variance
            - 0.5 * expected_precision * rss
        )
    )
    if phi_posterior_df is None:
        value += _factor_elbo(
            n, factor_covariance, factor_second_sum, factor_prior_covariance
        )
    else:
        if (
            phi_posterior_scale is None
            or phi_prior_df is None
            or phi_prior_scale is None
        ):
            raise RuntimeError(
                "variational factor-covariance ELBO parameters are incomplete"
            )
        value += _factor_full_elbo(
            n,
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

    primary = structure.loading_mask > 0.5
    if estimate_primary_loadings and not enforce_loading_row_sums:
        primary_means = component_means[:, :, 1][primary]
        primary_variances = component_variances[:, :, 1][primary]
        primary_quadratic = (
            primary_variances + (primary_means - loading_prior_mean) ** 2
        ) / loading_prior_variance
        value += float(
            np.sum(
                -0.5
                * (np.log(2.0 * np.pi * loading_prior_variance) + primary_quadratic)
                + 0.5 * (1.0 + np.log(2.0 * np.pi * primary_variances))
            )
        )

    cross = ~primary
    if estimate_cross_loadings and np.any(cross):
        spike_means = component_means[:, :, 0][cross]
        slab_means = component_means[:, :, 1][cross]
        spike_variances = component_variances[:, :, 0][cross]
        slab_variances = component_variances[:, :, 1][cross]
        slab = np.clip(slab_probability[cross], 0.0, 1.0)
        spike = 1.0 - slab
        log_gamma_prior = special.xlogy(spike, 1.0 - inclusion_prior) + special.xlogy(
            slab, inclusion_prior
        )
        spike_log_prior = -0.5 * (
            np.log(2.0 * np.pi * spike_variance)
            + (spike_variances + spike_means**2) / spike_variance
        )
        slab_log_prior = -0.5 * (
            np.log(2.0 * np.pi * slab_variance)
            + (slab_variances + slab_means**2) / slab_variance
        )
        gamma_entropy = -special.xlogy(spike, spike) - special.xlogy(slab, slab)
        normal_entropy = spike * 0.5 * (
            1.0 + np.log(2.0 * np.pi * spike_variances)
        ) + slab * 0.5 * (1.0 + np.log(2.0 * np.pi * slab_variances))
        value += float(
            np.sum(
                log_gamma_prior
                + spike * spike_log_prior
                + slab * slab_log_prior
                + gamma_entropy
                + normal_entropy
            )
        )
    if not np.isfinite(value):
        raise RuntimeError("mean-field variational ELBO is not finite")
    return value


def _normal_mixture_quantile(
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    probability: float,
) -> float:
    standard_deviations = np.sqrt(variances)

    def centered_cdf(value: float) -> float:
        return float(
            weights @ stats.norm.cdf((value - means) / standard_deviations)
            - probability
        )

    tail_probability = min(probability, 1.0 - probability)
    width_z = max(8.0, abs(float(stats.norm.ppf(tail_probability))) + 4.0)
    lower = float(np.min(means - width_z * standard_deviations))
    upper = float(np.max(means + width_z * standard_deviations))
    width = max(1.0, upper - lower)
    while centered_cdf(lower) > 0.0:
        lower -= width
        width *= 2.0
    width = max(1.0, upper - lower)
    while centered_cdf(upper) < 0.0:
        upper += width
        width *= 2.0
    return float(optimize.brentq(centered_cdf, lower, upper))


def _fit_multiple(
    data: np.ndarray,
    blocks: Sequence[Block],
    *,
    max_iter: int = 1000,
    min_iter: int = 5,
    tolerance: float = 1e-06,
    credible_interval_level: float = 0.95,
    factor_prior_variance: float = 1.0,
    standardize_data: bool = True,
    cross_loading_inclusion_prior: float = 0.1,
    spike_variance: float = 0.0001,
    slab_variance: float = 0.25,
    residual_prior_shape: float = 0.01,
    residual_prior_scale: float = 0.01,
    elbo_decrease_tolerance: float = 1e-08,
    mixture_interval_method: str = "normal_approx",
    factor_covariance_prior_strength: float = 1.0,
    _factor_covariance_interval_draws: int = 2000,
    _factor_covariance_interval_seed: int = 1,
    loading_normalization: str = "row_sum",
    primary_loading_prior_mean: float = 1.0,
    primary_loading_prior_variance: float = 1.0,
    compute_backend: ComputeBackend = None,
) -> CFAResult:
    """Fit the multiple family with variational q(Phi): signed row sums one, or free primary loadings."""
    started = perf_counter()
    if loading_normalization not in ("row_sum", "free"):
        raise ValueError("loading_normalization must be 'row_sum' or 'free'")
    free = loading_normalization == "free"
    # primary-loading prior as seen by the ELBO; unused (and unchanged) under the row-sum constraint
    elbo_prior_mean = primary_loading_prior_mean if free else 1.0
    elbo_prior_variance = primary_loading_prior_variance if free else 0.0
    _validate_inputs(
        max_iter=max_iter,
        min_iter=min_iter,
        tolerance=tolerance,
        credible_interval_level=credible_interval_level,
        factor_prior_variance=factor_prior_variance,
        loading_prior_variance=0.0,
        cross_loading_inclusion_prior=cross_loading_inclusion_prior,
        spike_variance=spike_variance,
        slab_variance=slab_variance,
        residual_prior_shape=residual_prior_shape,
        residual_prior_scale=residual_prior_scale,
        elbo_decrease_tolerance=elbo_decrease_tolerance,
        mixture_interval_method=mixture_interval_method,
        factor_covariance_prior_strength=factor_covariance_prior_strength,
    )
    if factor_covariance_prior_strength <= 0.0:
        raise ValueError("member mf requires factor_covariance_prior_strength > 0")
    if _factor_covariance_interval_draws < 100:
        raise ValueError(
            "member mf requires at least 100 factor-covariance interval draws"
        )
    if int(_factor_covariance_interval_seed) != _factor_covariance_interval_seed:
        raise ValueError("factor-covariance interval seed must be an integer")
    (standardized, data_mean, data_scale) = _prepare_data(
        data, standardize=standardize_data
    )
    structure = make_scfa_block_structure(blocks, standardized.shape[1])
    x = standardized[:, structure.selected_indices]
    if compute_backend is None:
        compute_backend = resolve_compute_backend("cpu", 0)
    workspace = compute_backend.workspace(x)
    (n, p) = x.shape
    k = structure.loading_mask.shape[1]
    (
        loading_mean,
        loading_second_diagonal,
        inclusion_probability,
        component_means,
        component_variances,
    ) = _initialize_loadings(
        structure,
        elbo_prior_mean,
        elbo_prior_variance,
        cross_loading_inclusion_prior,
        spike_variance,
        slab_variance,
        True,
        True,
    )
    if free:
        (loading_second, loading_covariance) = _free_loading_moments(
            loading_mean, loading_second_diagonal
        )
    else:
        (loading_second, loading_covariance) = _constrained_loading_moments(
            structure, loading_mean, loading_second_diagonal
        )
        _record_constrained_primary_components(
            structure,
            loading_mean,
            loading_covariance,
            inclusion_probability,
            component_means,
            component_variances,
        )
    factor_mean = np.zeros((n, k), dtype=float)
    base_phi = factor_prior_variance * np.eye(k, dtype=float)
    factor_covariance = base_phi.copy()
    factor_second_sum = n * factor_covariance
    phi_prior_df: Optional[float] = None
    phi_prior_scale: Optional[np.ndarray] = None
    phi_posterior_df: Optional[float] = None
    phi_posterior_scale: Optional[np.ndarray] = None
    phi_prior_df = float(k + 1) + factor_covariance_prior_strength
    phi_prior_scale = factor_covariance_prior_strength * base_phi
    phi_posterior_df = phi_prior_df + n
    phi_posterior_scale = phi_prior_scale + factor_second_sum
    (phi_scale_precision, _) = _invert_spd(phi_posterior_scale)
    phi_precision = phi_posterior_df * phi_scale_precision
    phi = phi_posterior_scale / (phi_posterior_df - k - 1.0)
    residual_shape = np.full(p, residual_prior_shape, dtype=float)
    residual_scale = np.full(p, residual_prior_scale, dtype=float)
    elbo_history = [
        _elbo(
            x,
            structure,
            factor_mean,
            factor_covariance,
            factor_second_sum,
            loading_mean,
            loading_second,
            loading_covariance,
            component_means,
            component_variances,
            inclusion_probability,
            residual_shape,
            residual_scale,
            phi,
            elbo_prior_mean,
            elbo_prior_variance,
            cross_loading_inclusion_prior,
            spike_variance,
            slab_variance,
            residual_prior_shape,
            residual_prior_scale,
            True,
            True,
            not free,
            phi_posterior_df=phi_posterior_df,
            phi_posterior_scale=phi_posterior_scale,
            phi_prior_df=phi_prior_df,
            phi_prior_scale=phi_prior_scale,
        )
    ]
    converged = False
    max_decrease = 0.0
    n_iterations = 0
    for iteration in range(1, max_iter + 1):
        expected_precision = residual_shape / residual_scale
        if workspace is None:
            factor_precision = phi_precision + np.einsum(
                "p,pkl->kl", expected_precision, loading_second, optimize=True
            )
            (factor_covariance, _, _) = _solve_spd(factor_precision)
            weighted_loading = loading_mean * expected_precision[:, None]
            factor_mean = x @ weighted_loading @ factor_covariance
            factor_second_sum = n * factor_covariance + factor_mean.T @ factor_mean
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
        if phi_prior_scale is None or phi_posterior_df is None:
            raise RuntimeError("inverse-Wishart state is incomplete")
        phi_posterior_scale = phi_prior_scale + factor_second_sum
        (phi_scale_precision, _) = _invert_spd(phi_posterior_scale)
        phi_precision = phi_posterior_df * phi_scale_precision
        phi = phi_posterior_scale / (phi_posterior_df - k - 1.0)
        if free:
            _update_loadings_free(
                structure,
                factor_second_sum,
                factor_response_sums,
                expected_precision,
                loading_mean,
                loading_second_diagonal,
                inclusion_probability,
                component_means,
                component_variances,
                cross_loading_inclusion_prior,
                spike_variance,
                slab_variance,
                primary_loading_prior_mean,
                primary_loading_prior_variance,
            )
            (loading_second, loading_covariance) = _free_loading_moments(
                loading_mean, loading_second_diagonal
            )
        else:
            _update_loadings_unit_row_sum(
                structure,
                factor_second_sum,
                factor_response_sums,
                expected_precision,
                loading_mean,
                loading_second_diagonal,
                inclusion_probability,
                component_means,
                component_variances,
                cross_loading_inclusion_prior,
                spike_variance,
                slab_variance,
                True,
            )
            (loading_second, loading_covariance) = _constrained_loading_moments(
                structure, loading_mean, loading_second_diagonal
            )
            _record_constrained_primary_components(
                structure,
                loading_mean,
                loading_covariance,
                inclusion_probability,
                component_means,
                component_variances,
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
        residual_shape = np.full(p, residual_prior_shape + 0.5 * n, dtype=float)
        residual_scale = residual_prior_scale + 0.5 * rss
        current = _elbo(
            x,
            structure,
            factor_mean,
            factor_covariance,
            factor_second_sum,
            loading_mean,
            loading_second,
            loading_covariance,
            component_means,
            component_variances,
            inclusion_probability,
            residual_shape,
            residual_scale,
            phi,
            elbo_prior_mean,
            elbo_prior_variance,
            cross_loading_inclusion_prior,
            spike_variance,
            slab_variance,
            residual_prior_shape,
            residual_prior_scale,
            True,
            True,
            not free,
            phi_posterior_df=phi_posterior_df,
            phi_posterior_scale=phi_posterior_scale,
            phi_prior_df=phi_prior_df,
            phi_prior_scale=phi_prior_scale,
        )
        previous = elbo_history[-1]
        elbo_history.append(current)
        decrease = max(0.0, previous - current)
        max_decrease = max(max_decrease, decrease)
        if decrease > elbo_decrease_tolerance * (1.0 + abs(previous)):
            raise RuntimeError(
                f"mf: ELBO decreased beyond numerical tolerance: {previous:.12g} -> {current:.12g}"
            )
        relative_change = abs(current - previous) / (1.0 + abs(previous))
        n_iterations = iteration
        if iteration >= min_iter and relative_change <= tolerance:
            converged = True
            break
    posterior_started = perf_counter()
    inclusion_probability[structure.loading_mask > 0.5] = 1.0
    loading_variance = np.maximum(
        np.diagonal(loading_covariance, axis1=1, axis2=2), 0.0
    )
    loading_sd = np.sqrt(loading_variance)
    alpha = 0.5 * (1.0 - credible_interval_level)
    quantile = float(stats.norm.ppf(1.0 - alpha))
    loading_lower = loading_mean - quantile * loading_sd
    loading_upper = loading_mean + quantile * loading_sd
    if mixture_interval_method == "quantile":
        for variable_id in range(p):
            for factor_id in range(k):
                if structure.loading_mask[variable_id, factor_id] > 0.5:
                    continue
                slab = inclusion_probability[variable_id, factor_id]
                weights = np.asarray([1.0 - slab, slab])
                means = component_means[variable_id, factor_id]
                variances = component_variances[variable_id, factor_id]
                loading_lower[variable_id, factor_id] = _normal_mixture_quantile(
                    weights, means, variances, alpha
                )
                loading_upper[variable_id, factor_id] = _normal_mixture_quantile(
                    weights, means, variances, 1.0 - alpha
                )
        loading_rng = np.random.default_rng(
            int(_factor_covariance_interval_seed) + 17000003
        )
        loading_draws = _draw_loading_components(
            loading_mask=structure.loading_mask,
            probabilities=inclusion_probability,
            component_means=component_means,
            component_variances=component_variances,
            enforce_loading_row_sums=not free,
            draws=int(_factor_covariance_interval_draws),
            rng=loading_rng,
        )
        primary = structure.loading_mask > 0.5
        primary_lower = np.quantile(loading_draws, alpha, axis=0)
        primary_upper = np.quantile(loading_draws, 1.0 - alpha, axis=0)
        loading_lower[primary] = primary_lower[primary]
        loading_upper[primary] = primary_upper[primary]
    factor_sd = np.broadcast_to(
        np.sqrt(np.maximum(np.diag(factor_covariance), 0.0))[None, :], factor_mean.shape
    ).copy()
    residual_mean = residual_scale / (residual_shape - 1.0)
    reconstruction = factor_mean @ loading_mean.T
    phi_sd = None
    phi_lower = None
    phi_upper = None
    correlation_mean = None
    correlation_sd = None
    correlation_lower = None
    correlation_upper = None
    if phi_posterior_df is None or phi_posterior_scale is None:
        raise RuntimeError("inverse-Wishart posterior is unavailable")
    rng = np.random.default_rng(int(_factor_covariance_interval_seed))
    phi_draws = stats.invwishart.rvs(
        df=phi_posterior_df,
        scale=phi_posterior_scale,
        size=int(_factor_covariance_interval_draws),
        random_state=rng,
    )
    phi_draws = np.asarray(phi_draws, dtype=float)
    if k == 1:
        phi_draws = phi_draws.reshape(int(_factor_covariance_interval_draws), 1, 1)
    elif phi_draws.ndim == 2:
        phi_draws = phi_draws[None, :, :]
    alpha = 0.5 * (1.0 - credible_interval_level)
    phi_sd = np.std(phi_draws, axis=0, ddof=1)
    phi_lower = np.quantile(phi_draws, alpha, axis=0)
    phi_upper = np.quantile(phi_draws, 1.0 - alpha, axis=0)
    phi_draw_diagonal = np.sqrt(
        np.maximum(np.diagonal(phi_draws, axis1=1, axis2=2), 1e-15)
    )
    correlation_draws = phi_draws / (
        phi_draw_diagonal[:, :, None] * phi_draw_diagonal[:, None, :]
    )
    diagonal = np.arange(k)
    correlation_draws[:, diagonal, diagonal] = 1.0
    correlation_mean = np.mean(correlation_draws, axis=0)
    correlation_sd = np.std(correlation_draws, axis=0, ddof=1)
    correlation_lower = np.quantile(correlation_draws, alpha, axis=0)
    correlation_upper = np.quantile(correlation_draws, 1.0 - alpha, axis=0)
    posterior = CFAPosterior(
        version="mf_free" if free else "mf_row_sum",
        selected_indices=structure.selected_indices.copy(),
        block_sizes=np.asarray([block.size for block in structure.blocks], dtype=int),
        loading_mask=structure.loading_mask.copy(),
        loading_mean=loading_mean.copy(),
        loading_sd=loading_sd,
        loading_ci_lower=loading_lower,
        loading_ci_upper=loading_upper,
        loading_inclusion_probability=inclusion_probability.copy(),
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
        cross_loading_inclusion_prior=float(cross_loading_inclusion_prior),
        spike_variance=float(spike_variance),
        slab_variance=float(slab_variance),
        estimate_primary_loadings=bool(True),
        estimate_cross_loadings=bool(True),
        enforce_loading_row_sums=bool(not free),
        standardize_data=bool(standardize_data),
        member="mf",
        factor_covariance_has_posterior=bool(True),
        factor_covariance_sd=phi_sd,
        factor_covariance_ci_lower=phi_lower,
        factor_covariance_ci_upper=phi_upper,
        factor_correlation_mean=correlation_mean,
        factor_correlation_sd=correlation_sd,
        factor_correlation_ci_lower=correlation_lower,
        factor_correlation_ci_upper=correlation_upper,
    )
    diagnostics = CFADiagnostics(
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
        loading_component_means=component_means.copy(),
        loading_component_variances=component_variances.copy(),
        loading_slab_probability=inclusion_probability.copy(),
        factor_prior_covariance=phi.copy(),
        estimate_factor_covariance=bool(True),
        estimate_primary_loadings=bool(True),
        estimate_cross_loadings=bool(True),
        enforce_loading_row_sums=bool(not free),
        standardize_data=bool(standardize_data),
        member="mf",
        primary_loading_prior_mean=float(primary_loading_prior_mean) if free else None,
        primary_loading_prior_variance=float(primary_loading_prior_variance) if free else None,
        factor_covariance_posterior_df=phi_posterior_df,
        factor_covariance_posterior_scale=(
            None if phi_posterior_scale is None else phi_posterior_scale.copy()
        ),
        factor_covariance_prior_df=phi_prior_df,
        factor_covariance_prior_scale=(
            None if phi_prior_scale is None else phi_prior_scale.copy()
        ),
        factor_covariance_interval_draws=int(_factor_covariance_interval_draws),
        factor_covariance_interval_seed=int(_factor_covariance_interval_seed),
        **compute_backend.diagnostics(),
    )
    return CFAResult(posterior=posterior, diagnostics=diagnostics)


def fit_member(data: np.ndarray, blocks: Sequence[Block], config) -> CFAResult:
    """Fit one confirmatory member with the settings of a ``CFAConfig``."""

    settings = config.fit_kwargs()
    member = settings.pop("member")
    compute_backend = resolve_compute_backend(
        settings.pop("device"), settings.pop("gpu_device_id")
    )
    settings["compute_backend"] = compute_backend
    if member == "1f":
        from ._fixed import _fit_fixed

        return _fit_fixed(data, blocks, **settings)
    if member == "2f":
        from ._two import _fit_two

        return _fit_two(data, blocks, **settings)
    if member == "2f+s":
        from ._two_sparse import _fit_two_sparse

        return _fit_two_sparse(data, blocks, **settings)
    settings["_factor_covariance_interval_draws"] = settings.pop(
        "factor_covariance_interval_draws"
    )
    settings["_factor_covariance_interval_seed"] = settings.pop(
        "factor_covariance_interval_seed"
    )
    return _fit_multiple(data, blocks, **settings)


__all__ = ["CFADiagnostics", "CFAPosterior", "CFAResult", "fit_member"]
