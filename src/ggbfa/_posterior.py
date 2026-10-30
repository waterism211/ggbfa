"""Common posterior summaries, posterior draws, and portable result export."""

from dataclasses import dataclass, asdict, replace
from pathlib import Path
import csv
import json
import platform
from datetime import datetime, timezone
import numpy as np
from scipy import stats, __version__ as scipy_version


PARAMETERS = ("F", "Lambda", "gamma", "psi", "Phi")


@dataclass(frozen=True)
class PosteriorSummary:
    mean: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    distribution: str
    applicable: np.ndarray


def summary_from_draws(draws, level, distribution, applicable=None, mean=None):
    x = np.asarray(draws)
    alpha = (1.0 - level) / 2.0
    return PosteriorSummary(
        x.mean(axis=0) if mean is None else np.asarray(mean),
        x.std(axis=0, ddof=1),
        np.quantile(x, alpha, axis=0),
        np.quantile(x, 1 - alpha, axis=0),
        level,
        distribution,
        np.ones(x.shape[1:], bool) if applicable is None else applicable,
    )


def bernoulli_summary(probabilities, level, applicable):
    p = np.where(applicable, np.clip(probabilities, 0, 1), 0.0)
    alpha = (1 - level) / 2
    return PosteriorSummary(
        p,
        np.sqrt(p * (1 - p)),
        stats.bernoulli.ppf(alpha, p),
        stats.bernoulli.ppf(1 - alpha, p),
        level,
        "Bernoulli marginal; primary entries are not indicators",
        applicable,
    )


def fitted_summaries(result):
    p, d = result.posterior, result.diagnostics

    def continuous(mean, sd, lo, hi, name):
        return PosteriorSummary(
            np.asarray(mean).copy(),
            np.broadcast_to(sd, np.shape(mean)).copy(),
            np.broadcast_to(lo, np.shape(mean)).copy(),
            np.broadcast_to(hi, np.shape(mean)).copy(),
            p.credible_interval_level,
            name,
            np.ones(np.shape(mean), bool),
        )

    alpha = (1 - p.credible_interval_level) / 2
    psi_sd = stats.invgamma.std(d.residual_shape, scale=d.residual_scale)
    summaries = {
        "F": continuous(
            p.factor_score_mean,
            p.factor_score_sd,
            p.factor_score_ci_lower,
            p.factor_score_ci_upper,
            "Gaussian q(F)",
        ),
        "Lambda": continuous(
            p.loading_mean,
            p.loading_sd,
            p.loading_ci_lower,
            p.loading_ci_upper,
            "induced constrained loading posterior" if getattr(p, "enforce_loading_row_sums", True)
            else "free primary loadings (Gaussian) and cross-loadings",
        ),
        "gamma": bernoulli_summary(
            p.loading_inclusion_probability,
            p.credible_interval_level,
            ~p.loading_mask.astype(bool),
        ),
        "psi": continuous(
            p.residual_variance_mean,
            psi_sd,
            stats.invgamma.ppf(alpha, d.residual_shape, scale=d.residual_scale),
            stats.invgamma.ppf(1 - alpha, d.residual_shape, scale=d.residual_scale),
            "inverse-Gamma q(psi)",
        ),
        "Phi": continuous(
            p.factor_prior_covariance,
            np.sqrt(
                stats.invwishart.var(
                    d.factor_covariance_posterior_df,
                    scale=d.factor_covariance_posterior_scale,
                )
            ),
            p.factor_covariance_ci_lower,
            p.factor_covariance_ci_upper,
            "inverse-Wishart q(Phi)",
        ),
    }
    if p.member == "1f":
        fixed = ([("Lambda", "point mass at the fixed primary mask")]
                 if getattr(p, "enforce_loading_row_sums", True) else [])
        for name, distribution in fixed + [("gamma", "point mass at zero; no free inclusion states")]:
            summaries[name] = replace(summaries[name], distribution=distribution,
                                      applicable=np.zeros_like(summaries[name].applicable))
        summaries["psi"] = replace(summaries["psi"], distribution="block-shared inverse-Gamma q(v); psi_j=v_h")
    return summaries


def draw_residual_variance(result, draws, rng, indices=None):
    """Draw independent variable factors or the shared block factors once.

    The expanded psi array always has variable coordinates. Drawing all block
    factors before subsetting also keeps marginal quantiles identical across
    chunks that contain variables from the same block.
    """
    p, d = result.posterior, result.diagnostics
    groups = getattr(d, "residual_group_index", None)
    if groups is None:
        ids = np.arange(len(p.loading_mean)) if indices is None else np.asarray(indices)
        return stats.invgamma.rvs(d.residual_shape[ids], scale=d.residual_scale[ids],
                                 size=(draws, len(ids)), random_state=rng)
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    block_draws = stats.invgamma.rvs(d.residual_shape[first], scale=d.residual_scale[first],
                                   size=(draws, len(first)), random_state=rng)
    return block_draws[:, inverse if indices is None else inverse[indices]]


def draw_fitted(result, draws, random_state, parameters=PARAMETERS):
    from ._model import _draw_loading_components, _draw_loading_posterior

    if isinstance(draws, bool) or int(draws) != draws or draws < 1:
        raise ValueError("draws must be a positive integer")
    if (
        isinstance(random_state, bool)
        or int(random_state) != random_state
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer")
    parameters = tuple(parameters)
    if set(parameters) - set(PARAMETERS):
        raise ValueError("parameters must be drawn from F, Lambda, gamma, psi, Phi")
    streams = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(int(random_state)).spawn(4)
    ]
    p, d = result.posterior, result.diagnostics
    out = {}
    if "Lambda" in parameters or "gamma" in parameters:
        if p.member == "mf":
            loading, gamma = _draw_loading_components(
                loading_mask=p.loading_mask,
                probabilities=d.loading_slab_probability,
                component_means=d.loading_component_means,
                component_variances=d.loading_component_variances,
                enforce_loading_row_sums=getattr(d, "enforce_loading_row_sums", True),
                draws=int(draws),
                rng=streams[0],
                return_indicators=True,
            )
        else:
            loading = _draw_loading_posterior(result, draws=int(draws), rng=streams[0])
            gamma = ((loading != 0) & ~p.loading_mask.astype(bool)[None]).astype(int)
        out.update(Lambda=loading, gamma=gamma.astype(np.int8))
    if "F" in parameters:
        out["F"] = (
            streams[1].multivariate_normal(
                np.zeros(p.loading_mean.shape[1]),
                d.factor_covariance,
                size=(int(draws), len(p.factor_score_mean)),
            )
            + p.factor_score_mean
        )
    if "psi" in parameters:
        out["psi"] = draw_residual_variance(result, int(draws), streams[2])
    if "Phi" in parameters:
        k = p.loading_mean.shape[1]
        out["Phi"] = np.asarray(
            stats.invwishart.rvs(
                d.factor_covariance_posterior_df,
                scale=d.factor_covariance_posterior_scale,
                size=int(draws),
                random_state=streams[3],
            )
        ).reshape(int(draws), k, k)
    return {name: out[name] for name in parameters}


def loading_complexity(posterior, row_overlap=None, budgets=None):
    loading = posterior["Lambda"].mean
    indicator = posterior["gamma"]
    pip = indicator.mean[indicator.applicable]
    p, k = loading.shape
    values = {
        "n_variables": p,
        "n_factors": k,
        "expected_active_cross_loadings": float(pip.sum()),
        "expected_selected_loading_entries": float(p + pip.sum()),
        "expected_cross_loadings_per_row": float(pip.sum() / p),
        "pip_ge_half_cross_loadings": int((pip >= 0.5).sum()),
        "max_signed_row_sum_error": float(np.abs(loading.sum(axis=1) - 1).max()),
        "posterior_mean_density_abs_gt_0_01": float((np.abs(loading) > 0.01).mean()),
        "posterior_mean_density_abs_gt_0_05": float((np.abs(loading) > 0.05).mean()),
    }
    if row_overlap is not None:
        values["expected_overlapping_rows"] = float(np.sum(row_overlap))
        values["expected_overlap_fraction"] = float(np.mean(row_overlap))
    if budgets is not None:
        values["overlap_budget_by_block"] = np.asarray(budgets).tolist()
    return values


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def runtime_provenance(diagnostics=None):
    import os

    values = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "backend": "numpy/scipy",
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "actual_gpu_count": 0,
        "device_reason": "runtime_not_yet_resolved",
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if diagnostics is not None:
        values.update(
            backend=getattr(diagnostics, "compute_backend", "numpy/scipy"),
            requested_device=getattr(diagnostics, "requested_device", "cpu"),
            resolved_device=getattr(diagnostics, "resolved_device", "cpu"),
            actual_gpu_count=getattr(diagnostics, "actual_gpu_count", 0),
            device_reason=getattr(
                diagnostics, "device_reason", "unrecorded"
            ),
            gpu_device_id=getattr(diagnostics, "gpu_device_id", None),
            gpu_name=getattr(diagnostics, "gpu_name", None),
            cupy_version=getattr(diagnostics, "cupy_version", None),
            cuda_runtime_version=getattr(
                diagnostics, "cuda_runtime_version", None
            ),
            gpu_operation_counts=dict(
                getattr(diagnostics, "gpu_operation_counts", None) or {}
            ),
        )
    return values


class PosteriorAccess:
    """Shared posterior interface of CFAFit and BaggedCFAFit."""

    def posterior(self, parameter):
        if not hasattr(self, "posterior_"):
            raise RuntimeError("fit must be called before accessing the posterior")
        if parameter not in PARAMETERS:
            raise ValueError("parameter must be F, Lambda, gamma, psi, or Phi")
        return self.posterior_[parameter]

    @property
    def factor_scores_(self):
        return self.posterior("F").mean.copy()

    @property
    def factor_score_ci_(self):
        p = self.posterior("F")
        return p.lower.copy(), p.upper.copy()

    @property
    def loading_ci_(self):
        p = self.posterior("Lambda")
        return p.lower.copy(), p.upper.copy()

    @property
    def residual_variance_ci_(self):
        p = self.posterior("psi")
        return p.lower.copy(), p.upper.copy()

    def save_results(self, directory):
        """Save numerical posterior summaries plus config/provenance as JSON/CSV."""
        self.posterior("F")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "result_"):
            selected_mean = self.result_.posterior.data_mean
            selected_scale = self.result_.posterior.data_scale
        else:
            selected_mean, selected_scale = self.data_mean_, self.data_scale_
        metadata = {
            "config": asdict(self.config),
            "metadata": self.metadata_,
            "timing_seconds": self.timing_,
            "loading_complexity": self.complexity_,
            "selected_indices": self.selected_indices_,
            "communities": self.communities_,
            "preprocessing": {
                "selected_data_mean": selected_mean,
                "selected_data_scale": selected_scale,
                "training_mean_all_features": self.training_mean_,
                "training_variance_all_features": self.training_variance_,
            },
            "posterior": {k: asdict(v) for k, v in self.posterior_.items()},
        }
        (path / "posterior.json").write_text(
            json.dumps(_jsonable(metadata), indent=2, allow_nan=False) + "\n"
        )
        with (path / "posterior.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "parameter",
                    "index",
                    "mean",
                    "sd",
                    "lower",
                    "upper",
                    "level",
                    "applicable",
                ]
            )
            for name, summary in self.posterior_.items():
                for index in np.ndindex(summary.mean.shape):
                    writer.writerow(
                        [
                            name,
                            ":".join(map(str, index)),
                            summary.mean[index],
                            summary.sd[index],
                            summary.lower[index],
                            summary.upper[index],
                            summary.level,
                            summary.applicable[index],
                        ]
                    )
        with (path / "diagnostics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group", "name", "value"])
            for group, values in [
                ("timing_seconds", self.timing_),
                ("loading_complexity", self.complexity_),
                ("metadata", self.metadata_),
            ]:
                for key, value in values.items():
                    writer.writerow([group, key, json.dumps(_jsonable(value))])
        return path
