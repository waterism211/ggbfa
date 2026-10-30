"""Deterministic posterior moments, including budget-induced row dependence."""

import numpy as np
from scipy.special import logsumexp


def _conditional_gate_marginals(log_odds, excluded, budget):
    """Exact gate marginals given each excluded gate is one, in batches.

    The caller already subtracts one from the budget. Vectorization is over
    conditioning rows; no subsets or posterior Monte Carlo are enumerated.
    """
    values = np.broadcast_to(log_odds, (len(excluded), len(log_odds))).copy()
    values[np.arange(len(excluded)), excluded] = -np.inf
    batch, p = values.shape
    forward = np.full((p + 1, batch, budget + 1), -np.inf)
    forward[0, :, 0] = 0.0
    for j in range(p):
        forward[j + 1, :, 0] = forward[j, :, 0]
        forward[j + 1, :, 1:] = np.logaddexp(
            forward[j, :, 1:], forward[j, :, :-1] + values[:, j, None]
        )
    mass = np.exp(forward[-1] - logsumexp(forward[-1], axis=1)[:, None])
    marginal = np.zeros((batch, p))
    for j in range(p - 1, -1, -1):
        denominator = forward[j + 1]
        no = np.full_like(denominator, -np.inf)
        np.subtract(
            forward[j],
            denominator,
            out=no,
            where=np.isfinite(forward[j]) & np.isfinite(denominator),
        )
        previous = mass * np.exp(no)
        numerator = forward[j, :, :-1] + values[:, j, None]
        yes = np.full_like(numerator, -np.inf)
        np.subtract(
            numerator,
            denominator[:, 1:],
            out=yes,
            where=np.isfinite(numerator) & np.isfinite(denominator[:, 1:]),
        )
        contribution = mass[:, 1:] * np.exp(yes)
        previous[:, :-1] += contribution
        marginal[:, j] = contribution.sum(axis=1)
        mass = previous
    return np.clip(marginal, 0.0, 1.0)


def gate_pair_moments(log_odds, budget, *, indices=None, batch_size=16):
    """Return exact E[g_i g_j] under a truncated-Bernoulli block factor.

    Cost O(r p L), memory O(batch_size p L + r^2), where r is the number
    of requested rows. This cost is incurred for covariance reporting only,
    not for each VB iteration. Equal and extreme log odds are supported.
    """
    from ._two_sparse import _truncated_bernoulli_dp

    values = np.asarray(log_odds, dtype=float)
    probabilities, _, _ = _truncated_bernoulli_dp(values, budget)
    ids = np.arange(values.size) if indices is None else np.asarray(indices, dtype=int)
    result = np.zeros((ids.size, ids.size))
    if budget >= values.size:
        result = np.outer(probabilities[ids], probabilities[ids])
    elif budget > 1:
        for start in range(0, ids.size, batch_size):
            chosen = ids[start : start + batch_size]
            conditional = _conditional_gate_marginals(values, chosen, budget - 1)
            result[start : start + chosen.size] = (
                probabilities[chosen, None] * conditional[:, ids]
            )
        result = 0.5 * (result + result.T)
    np.fill_diagonal(result, probabilities[ids])
    return result


def expected_loading_quadratic(result, matrix, *, indices=None):
    """Return E_q[Lambda A Lambda.T] for a fixed symmetric K-by-K A."""
    posterior, diagnostics = result.posterior, result.diagnostics
    p = len(posterior.loading_mean)
    ids = np.arange(p) if indices is None else np.asarray(indices, dtype=int)
    loading = np.asarray(posterior.loading_mean)[ids]
    a = np.asarray(matrix, dtype=float)
    value = loading @ a @ loading.T
    diagonal = np.einsum(
        "ab,pba->p", a, diagnostics.loading_covariance[ids], optimize=True
    )
    value[np.diag_indices_from(value)] += diagonal
    if posterior.member == "2f+s":
        membership = np.argmax(posterior.loading_mask, axis=1)
        target = diagnostics.conditional_target_probability
        direction = target * diagnostics.active_state_means
        direction = direction.copy()
        direction[np.arange(p), membership] = 0.0
        if getattr(diagnostics, "enforce_loading_row_sums", True):  # the primary loading moves with the cross-loading
            direction[np.arange(p), membership] = -direction.sum(axis=1)
        for h, budget in enumerate(diagnostics.overlap_budget_by_factor):
            block = np.flatnonzero(membership == h)
            positions = np.flatnonzero(membership[ids] == h)
            if positions.size < 2 or budget == 0:
                continue
            local = np.searchsorted(block, ids[positions])
            joint = gate_pair_moments(
                diagnostics.gate_log_odds[block], int(budget), indices=local
            )
            gate = posterior.row_overlap_probability[ids[positions]]
            gate_covariance = joint - np.outer(gate, gate)
            np.fill_diagonal(gate_covariance, 0.0)  # row variance is already included
            u = direction[ids[positions]]
            value[np.ix_(positions, positions)] += gate_covariance * (u @ a @ u.T)
    return 0.5 * (value + value.T)


def observation_covariance(result):
    """E_q[Lambda Phi Lambda.T + diag(psi)] on the fitted model scale."""
    p = result.posterior
    value = expected_loading_quadratic(result, p.factor_prior_covariance)
    value[np.diag_indices_from(value)] += p.residual_variance_mean
    return value
