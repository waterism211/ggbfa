"""Two prediction targets with explicit intervals, numerical checks and timing.

No covariance regularization or invalid-cell filtering is performed here.
Only roundoff-level asymmetry (rtol=1e-9, atol=1e-10) is symmetrized.
"""

from dataclasses import dataclass, replace
from numbers import Integral, Real
from time import perf_counter
from typing import Optional

import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm


@dataclass(frozen=True)
class MaskedPrediction:
    """Raw-scale masked observation predictions, not parameter credible intervals."""

    masked_indices: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray
    selected_masked_indices: np.ndarray
    residual_only_indices: np.ndarray
    prediction_type: str = "variational_posterior"
    level: float = 0.95
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    interval_method: Optional[str] = None
    prediction_draws: Optional[int] = None
    prediction_seed: Optional[int] = None
    batch_size: Optional[int] = None
    quantile_method: Optional[str] = None
    density_approximation: str = "moment_matched_gaussian"
    draws: Optional[np.ndarray] = None


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def interval_level(level):
    if isinstance(level, (bool, np.bool_)) or not isinstance(level, Real):
        raise ValueError("level must be a finite number strictly between 0 and 1")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be a finite number strictly between 0 and 1")
    return float(level)


def visible_row(row, masked, n_features):
    """Validate shape/type without converting or inspecting hidden cell values."""
    raw = np.asarray(row)
    mask = np.asarray(masked)
    if raw.ndim != 1 or raw.shape != (n_features,) or mask.shape != raw.shape:
        raise ValueError("row and masked must be 1D with the fitted feature count")
    if mask.dtype.kind != "b":
        raise ValueError("masked must be boolean (True means hidden)")
    if not mask.any() or mask.all():
        raise ValueError("masked must contain both visible and masked cells")
    values = np.full(n_features, np.nan)
    values[~mask] = np.asarray(raw[~mask], dtype=float)
    if not np.isfinite(values[~mask]).all():
        raise ValueError("visible cells must be finite")
    return values, mask.copy()


def checked_spd(matrix, context):
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not len(value):
        raise ValueError(f"{context}: covariance must be nonempty and square")
    if not np.isfinite(value).all():
        raise ValueError(f"{context}: covariance must be finite")
    if not np.allclose(value, value.T, rtol=1e-9, atol=1e-10):
        raise ValueError(f"{context}: covariance is not symmetric within tolerance")
    value = (value + value.T) / 2
    try:
        chol = np.linalg.cholesky(value)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{context}: covariance must be positive definite; no ridge applied"
        ) from exc
    return value, chol


def checked_moments(result, context):
    if not np.isfinite(result.mean).all():
        raise ValueError(f"{context}: nonfinite predictive mean")
    covariance, _ = checked_spd(result.covariance, context)
    return replace(result, covariance=covariance, variance=np.diag(covariance).copy())


class GaussianPredictor:
    """Condition a supplied mean/covariance, reusable by non-Bayesian comparators.

    Bayesian callers must supply E_q[Sigma], not a product of parameter means.
    The arrays are copied at construction; preparation is separate from predict.
    """

    def __init__(self, mean, covariance, *, selected_indices=None):
        self.mean = np.asarray(mean, dtype=float).copy()
        if self.mean.ndim != 1 or not np.isfinite(self.mean).all():
            raise ValueError("Gaussian predict: mean must be a finite vector")
        self.covariance, _ = checked_spd(covariance, "Gaussian predict preparation")
        if self.covariance.shape != (len(self.mean), len(self.mean)):
            raise ValueError("Gaussian predict: mean/covariance shapes disagree")
        selected = (
            np.arange(len(self.mean))
            if selected_indices is None
            else np.asarray(selected_indices)
        )
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError(
                "selected_indices must be distinct integer feature indices"
            )
        if np.any(selected < 0) or np.any(selected >= len(self.mean)):
            raise ValueError("selected_indices out of range")
        self.selected_indices = selected.copy()

    def predict(self, row, masked, *, level=0.95):
        level = interval_level(level)
        values, mask = visible_row(row, masked, len(self.mean))
        ids, observed = np.flatnonzero(mask), np.flatnonzero(~mask)
        c = self.covariance
        _, chol = checked_spd(
            c[np.ix_(observed, observed)], "Gaussian predict observed block"
        )
        cross = c[np.ix_(ids, observed)]
        mean = self.mean[ids] + cross @ cho_solve(
            (chol, True), values[observed] - self.mean[observed]
        )
        covariance = c[np.ix_(ids, ids)] - cross @ cho_solve((chol, True), cross.T)
        covariance, _ = checked_spd(covariance, "Gaussian predict conditional block")
        if not np.isfinite(mean).all():
            raise ValueError("Gaussian predict: nonfinite conditional mean")
        variance = np.diag(covariance).copy()
        half_width = norm.ppf((1 + level) / 2) * np.sqrt(variance)
        selected = np.isin(ids, self.selected_indices)
        return MaskedPrediction(
            ids,
            mean,
            variance,
            covariance,
            ids[selected],
            ids[~selected],
            prediction_type="gaussian",
            level=level,
            lower=mean - half_width,
            upper=mean + half_width,
            interval_method="gaussian_conditional",
        )


class PredictionAccess:
    """Shared public interface; estimator subclasses implement local moments/draws."""

    def _reset_prediction(self):
        self._gaussian_predictor = None
        for key in (
            "gaussian_prepare_seconds",
            "gaussian_predict_seconds",
            "posterior_moments_seconds",
            "posterior_sampling_seconds",
            "posterior_quantile_seconds",
            "posterior_predict_seconds",
        ):
            self.timing_[key] = 0.0
        self.metadata_["prediction"] = {
            "gaussian": "condition training mean and posterior-mean observation covariance",
            "variational_posterior": "local q(F_new), frozen joint q(Lambda), q(psi), observation noise",
            "gbb_sampling": "uniform component per observation draw; no component averaging",
            "intervals": "Gaussian conditional / variational predictive MC equal-tail quantiles",
            "density_approximation": "moment_matched_gaussian",
            "default_draws": 10000,
            "default_seed": 1,
            "default_batch_size": 256,
            "seed_rule": "SeedSequence child streams; same seed AND batch configuration reproducible",
            "symmetry_rtol": 1e-9,
            "symmetry_atol": 1e-10,
            "residual_only": "independent Gaussian with full-training mean and variance",
        }

    def predict_gaussian(self, row, masked, *, level=0.95):
        """A: conditional Gaussian prediction based on E_q[Sigma].

        The first call prepares/caches covariance; refitting invalidates it.
        Bagging first averages covariances, then conditions that single matrix.
        """
        self.posterior("F")
        level = interval_level(level)
        values, mask = visible_row(row, masked, self.n_features_in_)
        started = perf_counter()
        if self._gaussian_predictor is None:
            self._gaussian_predictor = GaussianPredictor(
                self.training_mean_,
                self.covariance(),
                selected_indices=self.selected_indices_,
            )
            self.timing_["gaussian_prepare_seconds"] += perf_counter() - started
        prepared = perf_counter()
        result = self._gaussian_predictor.predict(values, mask, level=level)
        self.timing_["gaussian_predict_seconds"] += perf_counter() - prepared
        self.timing_["prediction_seconds"] += perf_counter() - started
        return result

    def predict_posterior(
        self,
        row,
        masked,
        *,
        level=0.95,
        draws=10000,
        random_state=1,
        return_draws=False,
        batch_size=256,
    ):
        """B: analytic local-VB moments plus predictive sample-quantile intervals.

        Draws include observation noise. This is not MCMC or an exact Bayesian
        predictive. Phi enters local inference through E[Phi^-1]. Changing the
        predictive seed never changes analytic means/covariances. Only masked
        observation draws are retained when return_draws=True; no training F
        samples are allocated. NLPD based on these moments is a Gaussian score.
        """
        self.posterior("F")
        level = interval_level(level)
        draws = integer(draws, "draws")
        random_state = integer(random_state, "random_state", 0)
        batch_size = integer(batch_size, "batch_size")
        if not isinstance(return_draws, (bool, np.bool_)):
            raise ValueError("return_draws must be boolean")
        values, mask = visible_row(row, masked, self.n_features_in_)
        started = perf_counter()
        result = checked_moments(
            self._posterior_moments(values, mask), "Variational posterior prediction"
        )
        moments_done = perf_counter()
        samples = self._draw_predictive(values, mask, draws, random_state, batch_size)
        if samples.shape != (draws, mask.sum()) or not np.isfinite(samples).all():
            raise ValueError(
                "Variational posterior prediction: invalid/nonfinite draws; no interval fallback"
            )
        sampling_done = perf_counter()
        alpha = (1 - level) / 2
        lower, upper = np.quantile(samples, [alpha, 1 - alpha], axis=0, method="linear")
        done = perf_counter()
        self.timing_["posterior_moments_seconds"] += moments_done - started
        self.timing_["posterior_sampling_seconds"] += sampling_done - moments_done
        self.timing_["posterior_quantile_seconds"] += done - sampling_done
        self.timing_["posterior_predict_seconds"] += done - started
        self.timing_["prediction_seconds"] += done - started
        return replace(
            result,
            level=level,
            lower=lower,
            upper=upper,
            interval_method="variational_predictive_mc_quantile",
            prediction_draws=draws,
            prediction_seed=random_state,
            batch_size=batch_size,
            quantile_method="linear",
            draws=samples if return_draws else None,
        )

    def predict_masked(self, row, masked, **kwargs):
        """Compatibility alias for B, now also sampling intervals (default 10,000)."""
        return self.predict_posterior(row, masked, **kwargs)
