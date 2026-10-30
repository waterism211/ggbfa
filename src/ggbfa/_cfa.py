"""Confirmatory step of gg-BFA: ``cfa`` and the fitted object it returns."""

from __future__ import annotations

from time import perf_counter
from typing import Sequence, Tuple

import numpy as np

from ._config import CFAConfig
from ._version import __version__
from ._model import CFAResult, fit_member
from ._members import member_label, normalize_member
from ._moments import observation_covariance, expected_loading_quadratic
from ._prediction import MaskedPrediction, PredictionAccess, checked_spd, visible_row
from ._posterior import (
    PosteriorAccess,
    fitted_summaries,
    draw_fitted,
    loading_complexity,
    PARAMETERS,
    runtime_provenance,
)


def _as_communities(communities) -> Tuple[np.ndarray, ...]:
    """Accept an ``efa`` result or a sequence of 0-based column-index arrays."""

    values = getattr(communities, "communities", communities)
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise TypeError("communities must be an efa result or a sequence of index arrays")
    blocks = tuple(np.asarray(block, dtype=int) for block in values)
    if not blocks:
        raise ValueError("at least one community is required")
    return blocks


def cfa(
    data,
    communities,
    member: str = "2f",
    *,
    bagging: int | None = None,
    bootstrap_size: int | None = None,
    bagging_draws: int = 200,
    random_state: int = 1,
    **settings,
):
    """Fit one confirmatory member of gg-BFA to the given communities.

    Parameters
    ----------
    data:
        Complete n-by-p matrix.  Each column is standardized internally.
    communities:
        The result of :func:`ggbfa.efa`, or a sequence of 0-based column-index
        arrays.  Variables outside every community keep only their own variance.
    member:
        ``"1f"``, ``"2f+s"``, ``"2f"`` or ``"mf"``.
    bagging:
        ``None`` for a single variational fit, or the number of bootstrap fits
        ``B >= 2`` of variational bagging (each of ``bootstrap_size`` rows,
        2n by default; ``bagging_draws`` posterior draws per fit for the
        marginal intervals; ``random_state`` seeds the bootstrap).
    **settings:
        Any field of :class:`ggbfa.CFAConfig`, for example
        ``loading_normalization="free"``, ``device="cpu"`` or prior settings.

    Returns
    -------
    CFAFit, or BaggedCFAFit when ``bagging`` is given.
    """

    config = CFAConfig(member=member, **settings)
    blocks = _as_communities(communities)
    if bagging is None:
        return CFAFit(config).fit(data, blocks)
    from ._bagging import BaggedCFAFit, BaggingConfig

    return BaggedCFAFit(
        BaggingConfig(
            base_config=config,
            n_bootstrap=bagging,
            bootstrap_size=bootstrap_size,
            posterior_draws_per_bootstrap=bagging_draws,
            random_state=random_state,
        )
    ).fit(data, blocks)


class CFAFit(PredictionAccess, PosteriorAccess):
    """One confirmatory member fitted by mean-field variational Bayes.

    Created by :func:`ggbfa.cfa`.  Phi has an inverse-Wishart variational
    factor, and element-wise covariance and correlation intervals are
    reported.
    """

    def __init__(self, config: CFAConfig | None = None):
        if config is not None and not isinstance(config, CFAConfig):
            raise TypeError("config must be a CFAConfig")
        self.config = CFAConfig() if config is None else config

    def fit(self, data: np.ndarray, communities: Sequence[Sequence[int]]) -> "CFAFit":
        """Fit the member to a complete row-by-variable matrix."""

        started = perf_counter()
        x = np.asarray(data, dtype=float)
        if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
            raise ValueError("data must have shape (n>=3, p>=2)")
        if not np.all(np.isfinite(x)):
            raise ValueError("cfa requires a finite, complete matrix")
        blocks = _as_communities(communities)
        result = fit_member(x, blocks, self.config)
        fit_done = perf_counter()
        self.result_: CFAResult = result
        self.n_features_in_ = int(x.shape[1])
        self.training_mean_ = x.mean(axis=0)
        self.training_variance_ = x.var(axis=0, ddof=0)
        self.selected_indices_ = result.posterior.selected_indices.copy()
        self.communities_ = tuple(np.unique(block) for block in blocks if len(block))
        self.posterior_ = fitted_summaries(result)
        self.complexity_ = loading_complexity(
            self.posterior_,
            getattr(result.posterior, "row_overlap_probability", None),
            getattr(result.posterior, "overlap_budget_by_factor", None),
        )
        if self.member_ == "1f":
            self.complexity_.update(expected_overlapping_rows=0., expected_overlap_fraction=0.)
        self.timing_ = {
            "fit_and_intervals_seconds": fit_done - started,
            "fit_seconds": result.diagnostics.fit_runtime_seconds,
            "interval_seconds": result.diagnostics.interval_runtime_seconds,
            "summary_seconds": perf_counter() - fit_done,
            "covariance_seconds": 0.0,
            "prediction_seconds": 0.0,
        }
        self.metadata_ = {
            "runtime": runtime_provenance(result.diagnostics),
            "package_version": __version__,
            "member": self.member_,
            "loading_normalization": self.config.loading_normalization,
            "covariance_definition": "E_q[Lambda Phi Lambda.T + diag(psi)]",
            "converged": bool(result.diagnostics.converged),
            "n_iterations": int(result.diagnostics.n_iterations),
            "n_samples": int(x.shape[0]),
            "model_scale": (
                "training z-score ddof=0"
                if self.config.standardize_data
                else "centered input"
            ),
            "interval_seed": self.config.factor_covariance_interval_seed,
            "factor_intervals": "variational approximate, not guaranteed calibrated",
            "selected_fraction": len(self.selected_indices_) / self.n_features_in_,
        }
        self.metadata_["residual_structure"] = (
            "community_shared" if self.member_ == "1f" else "variable_specific")
        if self.member_ == "1f":
            self.metadata_["initialization_only"] = {
                "factor_eigenvalue_floor": result.diagnostics.initialization_factor_eigenvalue_floor,
                "residual_floor": result.diagnostics.initialization_residual_floor,
                "reported_covariance_regularized": False,
            }
        self._reset_prediction()
        return self

    def sample_posterior(self, draws=1000, *, random_state=1, parameters=PARAMETERS):
        """Draw coupled loading/indicator states and all remaining q factors."""
        return draw_fitted(self._require_fit(), draws, random_state, parameters)

    def _require_fit(self) -> CFAResult:
        if not hasattr(self, "result_"):
            raise RuntimeError("fit must be called before using this method")
        return self.result_

    @property
    def loadings_(self) -> np.ndarray:
        return self._require_fit().posterior.loading_mean.copy()

    @property
    def member_(self) -> str:
        """The confirmatory member of this fit: 1f, 2f+s, 2f or mf."""

        return normalize_member(self._require_fit().posterior.member)

    @property
    def method_label_(self) -> str:
        """Label used in the paper, e.g. ``gg-BFA (2f+s)``; ``free`` is added for free primary loadings."""

        label = member_label(self.member_)
        if self.config.loading_normalization == "free":
            label = label[:-1] + ", free)"
        return label

    @property
    def cross_loading_pip_(self) -> np.ndarray:
        return self.posterior("gamma").mean.copy()

    def sample_loadings(
        self, draws: int = 1000, *, random_state: int = 1
    ) -> np.ndarray:
        """Draw loading matrices from the fitted variational posterior.

        Draws of 2f have at most one cross-loading per row; draws of 2f+s
        also satisfy every fitted community budget.
        """

        return self.sample_posterior(
            draws, random_state=random_state, parameters=("Lambda",)
        )["Lambda"]

    @property
    def row_overlap_pip_(self) -> np.ndarray:
        """Return row-level cross-loading probabilities for structured options."""

        if self.member_ == "1f":
            return np.zeros(len(self.selected_indices_))
        value = getattr(self._require_fit().posterior, "row_overlap_probability", None)
        if value is None:
            raise RuntimeError(
                "row_overlap_pip_ is available for member='2f' "
                "or '2f+s'"
            )
        return np.asarray(value, dtype=float).copy()

    @property
    def row_state_probability_(self) -> np.ndarray:
        """Return categorical row-state probabilities when available."""

        if self.member_ == "1f":
            return self._require_fit().posterior.loading_mask.copy()
        value = getattr(self._require_fit().posterior, "row_state_probability", None)
        if value is None:
            raise RuntimeError(
                "row_state_probability_ is available for member='2f' "
                "or '2f+s'"
            )
        return np.asarray(value, dtype=float).copy()

    @property
    def conditional_target_probability_(self) -> np.ndarray:
        """Return target-factor probabilities conditional on a cross-loading."""

        value = getattr(
            self._require_fit().posterior,
            "target_probability_given_overlap",
            None,
        )
        if value is None:
            raise RuntimeError(
                "conditional_target_probability_ is available for "
                "member='2f' or '2f+s'"
            )
        return np.asarray(value, dtype=float).copy()

    @property
    def overlap_budget_(self) -> np.ndarray:
        """Return the per-community budgets of a 2f+s fit."""

        value = getattr(self._require_fit().posterior, "overlap_budget_by_factor", None)
        if value is None:
            raise RuntimeError(
                "overlap_budget_ is available only for " "member='2f+s'"
            )
        return np.asarray(value, dtype=int).copy()

    @property
    def factor_covariance_(self) -> np.ndarray:
        value = self._require_fit().posterior.factor_prior_covariance
        if value is None:
            raise RuntimeError("fitted result does not contain a factor covariance")
        return value.copy()

    @property
    def factor_covariance_ci_(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return element-wise ``(lower, upper)`` intervals from ``q(Phi)``."""

        posterior = self._require_fit().posterior
        if (
            posterior.factor_covariance_ci_lower is None
            or posterior.factor_covariance_ci_upper is None
        ):
            raise RuntimeError(
                "factor-covariance intervals require a fitted model"
            )
        return (
            posterior.factor_covariance_ci_lower.copy(),
            posterior.factor_covariance_ci_upper.copy(),
        )

    @property
    def factor_correlation_(self) -> np.ndarray:
        """Return the posterior mean factor-correlation matrix from ``q(Phi)``."""

        value = self._require_fit().posterior.factor_correlation_mean
        if value is None:
            raise RuntimeError(
                "factor-correlation posterior summaries require a fitted model"
            )
        return value.copy()

    @property
    def factor_correlation_ci_(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return element-wise factor-correlation ``(lower, upper)`` intervals."""

        posterior = self._require_fit().posterior
        if (
            posterior.factor_correlation_ci_lower is None
            or posterior.factor_correlation_ci_upper is None
        ):
            raise RuntimeError(
                "factor-correlation intervals require a fitted model"
            )
        return (
            posterior.factor_correlation_ci_lower.copy(),
            posterior.factor_correlation_ci_upper.copy(),
        )

    @property
    def residual_variance_(self) -> np.ndarray:
        return self._require_fit().posterior.residual_variance_mean.copy()

    def covariance(
        self,
        *,
        selected_only: bool = False,
        standardized: bool = False,
    ) -> np.ndarray:
        """Return E_q[Lambda Phi Lambda.T + diag(psi)], never a plug-in.

        Includes within-row uncertainty and the inter-row gate dependence of 2f+s.
        Excluded features retain the explicit independent residual-only convention.
        """
        started = perf_counter()
        fit = self._require_fit().posterior
        selected_covariance = observation_covariance(self.result_)
        self.timing_["covariance_seconds"] += perf_counter() - started
        if not standardized:
            scale = np.asarray(fit.data_scale, dtype=float)
            selected_covariance = selected_covariance * np.outer(scale, scale)
        selected_covariance = 0.5 * (selected_covariance + selected_covariance.T)
        if selected_only:
            return selected_covariance

        diagonal = (
            np.ones(self.n_features_in_, dtype=float)
            if standardized and self.config.standardize_data
            else np.maximum(self.training_variance_, 0.0)
        )
        full = np.diag(diagonal)
        selected = self.selected_indices_
        full[np.ix_(selected, selected)] = selected_covariance
        return full

    def _local_factors(self, standardized_visible, observed):
        """Optimize a local Gaussian q(F) with fitted global q factors fixed."""
        p, d = self.result_.posterior, self.result_.diagnostics
        precision = d.factor_covariance_posterior_df * np.linalg.inv(
            d.factor_covariance_posterior_scale
        )
        loading = p.loading_mean[observed]
        residual_precision = (d.residual_shape / d.residual_scale)[observed]
        second = d.loading_covariance[observed] + np.einsum(
            "pi,pj->pij", loading, loading
        )
        precision = precision + np.einsum(
            "p,pij->ij", residual_precision, second, optimize=True
        )
        covariance = np.linalg.solve(precision, np.eye(precision.shape[0]))
        mean = (
            standardized_visible @ (loading * residual_precision[:, None])
        ) @ covariance
        return mean, covariance

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return local variational factor means using global posterior moments.

        Use factor_scores_ for the training fit's stored q(F) means. This local
        update is variational inference, not exact posterior conditioning.
        """
        self._require_fit()
        started = perf_counter()
        x = np.asarray(data, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if (
            x.ndim != 2
            or x.shape[1] != self.n_features_in_
            or not np.all(np.isfinite(x))
        ):
            raise ValueError("data must be complete with the fitted feature count")
        p = self.result_.posterior
        visible = (x[:, self.selected_indices_] - p.data_mean) / p.data_scale
        mean, _ = self._local_factors(
            visible, np.ones(len(self.selected_indices_), dtype=bool)
        )
        self.timing_["prediction_seconds"] += perf_counter() - started
        return mean

    def _posterior_moments(
        self,
        row: np.ndarray,
        masked: np.ndarray,
    ) -> MaskedPrediction:
        """Return local-VB posterior predictive moments, reading only visible cells."""

        fit = self._require_fit().posterior
        values, mask = visible_row(row, masked, self.n_features_in_)

        selected = self.selected_indices_
        selected_mask = mask[selected]
        selected_observed = ~selected_mask
        loading = np.asarray(fit.loading_mean, dtype=float)
        residual = np.asarray(fit.residual_variance_mean, dtype=float)
        standardized = (
            values[selected][selected_observed] - fit.data_mean[selected_observed]
        ) / fit.data_scale[selected_observed]
        means, factor_covariance = self._local_factors(
            standardized[None], selected_observed
        )
        factor_mean = means[0]

        masked_indices = np.flatnonzero(mask)
        position = {int(feature): i for i, feature in enumerate(masked_indices)}
        prediction_mean = np.empty(masked_indices.size, dtype=float)
        prediction_covariance = np.zeros(
            (masked_indices.size, masked_indices.size), dtype=float
        )

        selected_masked_indices = selected[selected_mask]
        if selected_masked_indices.size:
            selected_local = np.flatnonzero(selected_mask)
            masked_loading = loading[selected_local]
            mean_standardized = masked_loading @ factor_mean
            factor_second = factor_covariance + np.outer(factor_mean, factor_mean)
            covariance_standardized = expected_loading_quadratic(
                self.result_,
                factor_second,
                indices=selected_local,
            ) - np.outer(mean_standardized, mean_standardized)
            covariance_standardized[
                np.diag_indices_from(covariance_standardized)
            ] += residual[selected_local]
            scale = np.asarray(fit.data_scale, dtype=float)[selected_local]
            mean = (
                np.asarray(fit.data_mean, dtype=float)[selected_local]
                + scale * mean_standardized
            )
            covariance = covariance_standardized * np.outer(scale, scale)
            covariance, _ = checked_spd(
                covariance, "CFAFit local-VB selected prediction"
            )
            selected_positions = np.asarray(
                [position[int(feature)] for feature in selected_masked_indices],
                dtype=int,
            )
            prediction_mean[selected_positions] = mean
            prediction_covariance[np.ix_(selected_positions, selected_positions)] = (
                covariance
            )

        selected_set = set(int(value) for value in selected)
        residual_only_indices = np.asarray(
            [value for value in masked_indices if int(value) not in selected_set],
            dtype=int,
        )
        if residual_only_indices.size:
            residual_positions = np.asarray(
                [position[int(feature)] for feature in residual_only_indices],
                dtype=int,
            )
            prediction_mean[residual_positions] = self.training_mean_[
                residual_only_indices
            ]
            residual_variance = self.training_variance_[residual_only_indices]
            prediction_covariance[residual_positions, residual_positions] = (
                residual_variance
            )

        return MaskedPrediction(
            masked_indices=masked_indices,
            mean=prediction_mean,
            variance=np.diag(prediction_covariance).copy(),
            covariance=prediction_covariance,
            selected_masked_indices=selected_masked_indices.copy(),
            residual_only_indices=residual_only_indices,
        )

    def _draw_predictive(
        self,
        row,
        masked,
        draws,
        random_state,
        batch_size,
        *,
        include_residual_only=True,
    ):
        """Sample new observations; preserve joint loading gates across all rows."""
        fit = self._require_fit().posterior
        values, mask = visible_row(row, masked, self.n_features_in_)
        selected = self.selected_indices_
        observed = ~mask[selected]
        local = np.flatnonzero(mask[selected])
        ids = np.flatnonzero(mask)
        positions = np.searchsorted(ids, selected[local])
        outside = ~np.isin(ids, selected)
        streams = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(random_state).spawn(4)
        ]
        # Bagging can request selected-feature draws only, supplying its own
        # full-training residual-only distribution outside the components.
        output = np.zeros((draws, len(ids)))
        if len(local):
            visible = (
                values[selected[observed]] - fit.data_mean[observed]
            ) / fit.data_scale[observed]
            mean, covariance = self._local_factors(visible[None], observed)
            _, chol = checked_spd(covariance, "CFAFit predictive local q(F_new)")
            for start in range(0, draws, batch_size):
                count = min(batch_size, draws - start)
                # Do not request training F or Phi: Phi already enters local q(F)
                # through its expected precision. Loading draws stay JOINT.
                params = draw_fitted(
                    self.result_,
                    count,
                    int(streams[0].integers(2**32)),
                    ("Lambda", "psi"),
                )
                factors = (
                    mean[0] + streams[1].normal(size=(count, len(mean[0]))) @ chol.T
                )
                psi = params["psi"][:, local]
                if not np.isfinite(psi).all() or np.any(psi <= 0):
                    raise ValueError(
                        "CFAFit predictive sampling: invalid residual variance draw"
                    )
                predicted = np.einsum("dpk,dk->dp", params["Lambda"][:, local], factors)
                predicted += np.sqrt(psi) * streams[2].normal(size=(count, len(local)))
                output[start : start + count, positions] = (
                    fit.data_mean[local] + fit.data_scale[local] * predicted
                )
        if outside.any() and include_residual_only:
            variance = self.training_variance_[ids[outside]]
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise ValueError(
                    "CFAFit residual-only prediction: variance must be finite and positive"
                )
            output[:, outside] = self.training_mean_[ids[outside]] + np.sqrt(
                variance
            ) * streams[3].normal(size=(draws, outside.sum()))
        return output


__all__ = ["CFAFit", "MaskedPrediction", "cfa"]
