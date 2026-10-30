"""Member 1f: fixed loadings Lambda=A, residual variances shared within a community, full q(Phi).

With loading_normalization="free" the primary loading of each variable has a
Gaussian q under a N(m0, v0) prior; residual variances stay shared within
communities.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import special, stats

from ._backend import ComputeBackend, resolve_compute_backend
from ._model import (
    CFADiagnostics,
    CFAPosterior,
    CFAResult,
    make_scfa_block_structure,
    _prepare_data,
    _primary_loading_elbo,
    _residual_ss,
    _update_primary_loadings,
)


@dataclass(frozen=True)
class FixedDiagnostics(CFADiagnostics):
    initialization_factor_eigenvalue_floor: float = 0.0
    initialization_residual_floor: float = 1e-6


def _initial_moments(x, sizes):
    """Block covariance averages without allocating a p-by-p covariance."""
    n, _ = x.shape
    k = len(sizes)
    centered = x - x.mean(axis=0)
    sums = np.empty((n, k))
    psi, phi = np.empty(k), np.zeros((k, k))
    starts = np.r_[0, np.cumsum(sizes)]
    for h in range(k):
        block = centered[:, starts[h] : starts[h + 1]]
        sums[:, h] = block.sum(axis=1)
        ss = float(np.einsum("ij,ij->", block, block))
        phi[h, h] = (sums[:, h] @ sums[:, h] - ss) / (
            (n - 1) * sizes[h] * (sizes[h] - 1)
        )
        psi[h] = ss / ((n - 1) * sizes[h]) - phi[h, h]
        for other in range(h):
            phi[h, other] = phi[other, h] = (sums[:, h] @ sums[:, other]) / (
                (n - 1) * sizes[h] * sizes[other]
            )
    values, vectors = np.linalg.eigh(phi)
    floor = max(1e-6, 1e-6 * float(np.max(np.abs(values))))
    # Initialization only; never modify a reported covariance or predictive solve.
    return (
        (vectors * np.maximum(values, floor)) @ vectors.T,
        np.maximum(psi, 1e-6),
        floor,
    )


def _rss(x, sizes, mean, covariance):
    starts = np.r_[0, np.cumsum(sizes)]
    return np.array(
        [
            np.sum((x[:, starts[h] : starts[h + 1]] - mean[:, [h]]) ** 2)
            + len(x) * sizes[h] * covariance[h, h]
            for h in range(len(sizes))
        ]
    )


def _elbo(
    x,
    sizes,
    mean,
    covariance,
    precision_scale,
    df,
    shape,
    rate,
    prior_scale,
    prior_df,
    a0,
    b0,
    block_rss=None,
    primary_term=None,
):
    n, p = x.shape
    k = len(sizes)
    sign, logdet_scale = np.linalg.slogdet(precision_scale)
    sign_f, logdet_f = np.linalg.slogdet(covariance)
    if sign <= 0 or sign_f <= 0:
        raise RuntimeError("1f: variational scale lost positive definiteness")
    elogdet = (
        np.sum(special.digamma((df - np.arange(k)) / 2)) + k * np.log(2) + logdet_scale
    )
    precision = df * precision_scale
    elogtau = special.digamma(shape) - np.log(rate)
    likelihood = (
        0.5 * n * np.sum(sizes * elogtau)
        - 0.5 * n * p * np.log(2 * np.pi)
        - 0.5 * np.sum(shape / rate * (_rss(x, sizes, mean, covariance) if block_rss is None else block_rss))
    )
    scores = 0.5 * n * (elogdet - k * np.log(2 * np.pi))
    scores -= 0.5 * np.trace(precision @ (mean.T @ mean + n * covariance))
    prior = 0.5 * (prior_df - k - 1) * elogdet - 0.5 * np.trace(prior_scale @ precision)
    prior += 0.5 * prior_df * np.linalg.slogdet(prior_scale)[1]
    prior -= 0.5 * prior_df * k * np.log(2) + special.multigammaln(prior_df / 2, k)
    residual = np.sum(
        a0 * np.log(b0) - special.gammaln(a0) + (a0 - 1) * elogtau - b0 * shape / rate
    )
    entropy = 0.5 * n * (k * (1 + np.log(2 * np.pi)) + logdet_f)
    entropy += stats.wishart.entropy(df=df, scale=precision_scale)
    entropy += np.sum(stats.gamma.entropy(shape, scale=1 / rate))
    value = float(likelihood + scores + prior + residual + entropy)
    if primary_term is not None:  # free primary loadings: E[log prior] + entropy
        value += primary_term
    if not np.isfinite(value):
        raise RuntimeError("1f: ELBO is not finite")
    return value


def _fit_fixed(
    data,
    blocks,
    *,
    max_iter,
    min_iter,
    tolerance,
    credible_interval_level,
    factor_prior_variance,
    standardize_data,
    residual_prior_shape,
    residual_prior_scale,
    elbo_decrease_tolerance,
    factor_covariance_prior_strength,
    factor_covariance_interval_draws,
    factor_covariance_interval_seed,
    loading_normalization="row_sum",
    primary_loading_prior_mean=1.0,
    primary_loading_prior_variance=1.0,
    compute_backend: ComputeBackend = None,
):
    started = perf_counter()
    if loading_normalization not in ("row_sum", "free"):
        raise ValueError("loading_normalization must be 'row_sum' or 'free'")
    free = loading_normalization == "free"
    for name, value in {
        "tolerance": tolerance,
        "factor_prior_variance": factor_prior_variance,
        "residual_prior_shape": residual_prior_shape,
        "residual_prior_scale": residual_prior_scale,
        "factor_covariance_prior_strength": factor_covariance_prior_strength,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < credible_interval_level < 1:
        raise ValueError("credible_interval_level must lie in (0,1)")
    if not np.isfinite(elbo_decrease_tolerance) or elbo_decrease_tolerance < 0:
        raise ValueError("elbo_decrease_tolerance must be finite and nonnegative")
    raw = np.asarray(data, dtype=float)
    if raw.ndim != 2:
        raise ValueError("data must be a row-by-feature matrix")
    structure = make_scfa_block_structure(blocks, raw.shape[1])
    sizes = np.array([len(b) for b in structure.blocks])
    if np.any(sizes < 3):
        raise ValueError("member 1f requires at least three variables per community")
    x, center, scale = _prepare_data(
        raw[:, structure.selected_indices], standardize=standardize_data
    )
    if compute_backend is None:
        compute_backend = resolve_compute_backend("cpu", 0)
    workspace = compute_backend.workspace(x)
    n, p = x.shape
    k = len(sizes)
    mask = structure.loading_mask
    group = structure.membership
    prior_df = k + 1 + factor_covariance_prior_strength
    prior_scale = factor_covariance_prior_strength * factor_prior_variance * np.eye(k)
    df = prior_df + n
    initial_phi, initial_psi, floor = _initial_moments(x, sizes)
    precision_scale = np.linalg.inv(initial_phi) / df
    shape = residual_prior_shape + 0.5 * n * sizes
    if np.any(shape <= 1):
        raise ValueError(
            "residual shapes must exceed one so that reported residual-variance means exist"
        )
    rate = shape * initial_psi
    block_sums = x @ mask
    loading_second = np.einsum("pi,pj->pij", mask, mask)
    loading_covariance = np.zeros_like(loading_second)
    loading_mean = mask
    rows = np.arange(p)
    if free:
        primary_mean = np.full(p, float(primary_loading_prior_mean))
        primary_variance = np.full(p, float(primary_loading_prior_variance))

        def set_loading_moments():
            nonlocal loading_mean, loading_second, loading_covariance
            loading_mean = mask * primary_mean[:, None]
            loading_covariance = np.zeros((p, k, k))
            loading_covariance[rows, group, group] = primary_variance
            loading_second = loading_mean[:, :, None] * loading_mean[:, None, :] + loading_covariance

        set_loading_moments()
    response_sums = None

    def update_factors():
        nonlocal response_sums
        tau = shape / rate
        if free:
            if workspace is not None:
                covariance, mean_value, second_sum, response_sums = workspace.factor_update(
                    df * precision_scale,
                    tau[group],
                    loading_second,
                    loading_mean,
                )
                return mean_value, covariance, second_sum
            precision = df * precision_scale + np.einsum(
                "p,pkl->kl", tau[group], loading_second, optimize=True)
            cov = np.linalg.inv(precision)
            mean_value = (x @ (loading_mean * tau[group][:, None])) @ cov
            response_sums = x.T @ mean_value
            return mean_value, cov, mean_value.T @ mean_value + n * cov
        if workspace is not None:
            covariance, mean_value, second_sum, _ = workspace.factor_update(
                df * precision_scale,
                tau[group],
                loading_second,
                mask,
            )
            return mean_value, covariance, second_sum
        cov = np.linalg.inv(
            df * precision_scale + np.diag(sizes * tau)
        )
        mean_value = (block_sums * tau[None]) @ cov
        return mean_value, cov, mean_value.T @ mean_value + n * cov

    def residual_sums(mean_value, covariance):
        if free:
            if workspace is None:
                variable_rss = _residual_ss(
                    x, mean_value, covariance, loading_mean, loading_second, loading_covariance)
            else:
                variable_rss = workspace.residual_ss(
                    mean_value, covariance, loading_mean, loading_second, loading_covariance)
            return np.bincount(group, weights=variable_rss, minlength=k)
        if workspace is None:
            return _rss(x, sizes, mean_value, covariance)
        variable_rss = workspace.residual_ss(
            mean_value,
            covariance,
            mask,
            loading_second,
            loading_covariance,
        )
        return np.bincount(group, weights=variable_rss, minlength=k)

    def objective():
        if free:
            return _elbo(
                x, sizes, mean, cov, precision_scale, df, shape, rate, prior_scale, prior_df,
                residual_prior_shape, residual_prior_scale,
                block_rss=residual_sums(mean, cov),
                primary_term=_primary_loading_elbo(
                    primary_mean, primary_variance, primary_loading_prior_mean, primary_loading_prior_variance),
            )
        return _elbo(
            x,
            sizes,
            mean,
            cov,
            precision_scale,
            df,
            shape,
            rate,
            prior_scale,
            prior_df,
            residual_prior_shape,
            residual_prior_scale,
        )

    mean, cov, factor_second_sum = update_factors()
    history = [objective()]
    converged, max_decrease = False, 0.0
    for iteration in range(1, max_iter + 1):
        mean, cov, factor_second_sum = update_factors()
        precision_scale = np.linalg.inv(prior_scale + factor_second_sum)
        if free:
            primary_mean, primary_variance = _update_primary_loadings(
                group, factor_second_sum, response_sums, (shape / rate)[group], np.zeros((p, k)),
                primary_loading_prior_mean, primary_loading_prior_variance)
            set_loading_moments()
        rate = residual_prior_scale + 0.5 * residual_sums(mean, cov)
        value = objective()
        previous = history[-1]
        history.append(value)
        decrease = max(0.0, previous - value)
        max_decrease = max(max_decrease, decrease)
        if decrease > elbo_decrease_tolerance * (1 + abs(previous)):
            raise RuntimeError(f"1f: ELBO decreased: {previous} -> {value}")
        if (
            iteration >= min_iter
            and abs(value - previous) / (1 + abs(previous)) <= tolerance
        ):
            converged = True
            break
    # Preserve the exact state scored by the last ELBO, even at max_iter.
    fit_done = perf_counter()
    phi_scale = np.linalg.inv(precision_scale)
    phi_mean = phi_scale / (df - k - 1)
    alpha = (1 - credible_interval_level) / 2
    z = stats.norm.ppf(1 - alpha)
    f_sd = np.broadcast_to(np.sqrt(np.diag(cov)), mean.shape).copy()
    draws = np.asarray(
        stats.invwishart.rvs(
            df=df,
            scale=phi_scale,
            size=factor_covariance_interval_draws,
            random_state=np.random.default_rng(factor_covariance_interval_seed),
        )
    ).reshape(factor_covariance_interval_draws, k, k)
    sds = np.sqrt(np.diagonal(draws, axis1=1, axis2=2))
    correlations = draws / (sds[:, :, None] * sds[:, None, :])
    reconstruction = mean @ loading_mean.T
    if free:
        loading_sd = mask * np.sqrt(primary_variance)[:, None]
        component_means = np.repeat(loading_mean[..., None], 2, axis=2)
        component_variances = np.repeat((mask * primary_variance[:, None])[..., None], 2, axis=2)
    else:
        loading_sd = np.zeros_like(mask)
        component_means = np.repeat(mask[..., None], 2, axis=2)
        component_variances = np.zeros((p, k, 2))
    post = CFAPosterior(
        version="1f_free" if free else "1f_row_sum",
        selected_indices=structure.selected_indices.copy(),
        block_sizes=sizes,
        loading_mask=mask,
        loading_mean=loading_mean.copy(),
        loading_sd=loading_sd,
        loading_ci_lower=loading_mean - z * loading_sd,
        loading_ci_upper=loading_mean + z * loading_sd,
        loading_inclusion_probability=mask.copy(),
        factor_score_mean=mean,
        factor_score_sd=f_sd,
        factor_score_ci_lower=mean - z * f_sd,
        factor_score_ci_upper=mean + z * f_sd,
        factor_prior_covariance=phi_mean,
        residual_variance_mean=(rate / (shape - 1))[group],
        reconstruction_mean=reconstruction,
        reconstruction_rmse=float(np.sqrt(np.mean((x - reconstruction) ** 2))),
        data_mean=center,
        data_scale=scale,
        credible_interval_level=credible_interval_level,
        factor_prior_variance=factor_prior_variance,
        cross_loading_inclusion_prior=0.0,
        spike_variance=0.0,
        slab_variance=0.0,
        estimate_primary_loadings=free,
        estimate_cross_loadings=False,
        enforce_loading_row_sums=not free,
        standardize_data=standardize_data,
        member="1f",
        factor_covariance_has_posterior=True,
        factor_covariance_sd=draws.std(axis=0, ddof=1),
        factor_covariance_ci_lower=np.quantile(draws, alpha, axis=0),
        factor_covariance_ci_upper=np.quantile(draws, 1 - alpha, axis=0),
        factor_correlation_mean=correlations.mean(axis=0),
        factor_correlation_sd=correlations.std(axis=0, ddof=1),
        factor_correlation_ci_lower=np.quantile(correlations, alpha, axis=0),
        factor_correlation_ci_upper=np.quantile(correlations, 1 - alpha, axis=0),
    )
    diagnostics = FixedDiagnostics(
        elbo_history=np.asarray(history),
        converged=converged,
        n_iterations=iteration,
        max_iterations=max_iter,
        tolerance=tolerance,
        max_elbo_decrease=max_decrease,
        factor_covariance=cov,
        loading_covariance=loading_covariance.copy() if free else np.zeros((p, k, k)),
        residual_shape=shape[group],
        residual_scale=rate[group],
        loading_component_means=component_means,
        loading_component_variances=component_variances,
        loading_slab_probability=mask.copy(),
        factor_prior_covariance=phi_mean,
        estimate_factor_covariance=True,
        estimate_primary_loadings=free,
        estimate_cross_loadings=False,
        enforce_loading_row_sums=not free,
        primary_loading_prior_mean=float(primary_loading_prior_mean) if free else None,
        primary_loading_prior_variance=float(primary_loading_prior_variance) if free else None,
        standardize_data=standardize_data,
        member="1f",
        factor_covariance_posterior_df=df,
        factor_covariance_posterior_scale=phi_scale,
        factor_covariance_prior_df=prior_df,
        factor_covariance_prior_scale=prior_scale,
        factor_covariance_interval_draws=factor_covariance_interval_draws,
        factor_covariance_interval_seed=factor_covariance_interval_seed,
        fit_runtime_seconds=fit_done - started,
        interval_runtime_seconds=perf_counter() - fit_done,
        residual_group_index=group.copy(),
        initialization_factor_eigenvalue_floor=floor,
        **compute_backend.diagnostics(),
    )
    return CFAResult(post, diagnostics)
