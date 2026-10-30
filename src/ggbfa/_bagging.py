"""Variational bagging of one confirmatory member, with bounded-memory marginal intervals."""

from dataclasses import dataclass, field, replace
from time import perf_counter
import numpy as np
from scipy import stats

from ._config import CFAConfig
from ._version import __version__
from ._cfa import CFAFit, MaskedPrediction, _as_communities
from ._prediction import PredictionAccess, visible_row, checked_spd
from ._model import make_scfa_block_structure, _draw_loading_components, free_primary_moments
from ._two import _draw_row_sparse_loadings
from ._posterior import (
    PosteriorAccess,
    PosteriorSummary,
    PARAMETERS,
    bernoulli_summary,
    loading_complexity,
    draw_fitted,
    draw_residual_variance,
    runtime_provenance,
)
from ._joint import (
    JointCovariance,
    coordinate_layout,
    calibrate_free_covariance,
    stabilize_fixed_diagonal,
)


@dataclass(frozen=True)
class BaggingConfig:
    """Paper defaults B=20, M=2n; joint covariance is never built during fit."""

    base_config: CFAConfig = field(default_factory=CFAConfig)
    n_bootstrap: int = 20
    bootstrap_size: int = None
    posterior_draws_per_bootstrap: int = 200
    random_state: int = 1
    summary_chunk_size: int = 64
    joint_max_parameters: int = 2000

    def __post_init__(self):
        if not isinstance(self.base_config, CFAConfig):
            raise TypeError("base_config must be a CFAConfig")
        for name, minimum in [
            ("n_bootstrap", 2),
            ("posterior_draws_per_bootstrap", 2),
            ("summary_chunk_size", 1),
            ("joint_max_parameters", 1),
            ("random_state", 0),
        ]:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        if self.bootstrap_size is not None and (
            isinstance(self.bootstrap_size, bool)
            or int(self.bootstrap_size) != self.bootstrap_size
            or self.bootstrap_size < 3
        ):
            raise ValueError("bootstrap_size must be None or an integer >= 3")
        if self.bootstrap_size is not None:
            object.__setattr__(self, "bootstrap_size", int(self.bootstrap_size))


class BaggedCFAFit(PredictionAccess, PosteriorAccess):
    """Raw bootstrap-mixture posterior with optional calibrated joint moments.

    posterior_ and sample_posterior describe the actual constrained mixture,
    not a Gaussian approximation built from a calibrated covariance. At M=2n,
    the requested marginal variance multiplier is one. Joint calibration is
    explicitly separate and does not define structure-preserving draws.
    """

    def __init__(self, config=None):
        if config is not None and not isinstance(config, BaggingConfig):
            raise TypeError("config must be a BaggingConfig")
        self.config = BaggingConfig() if config is None else config

    def fit(self, data, communities):
        started = perf_counter()
        x = np.asarray(data, dtype=float)
        if (
            x.ndim != 2
            or x.shape[0] < 3
            or x.shape[1] < 2
            or not np.all(np.isfinite(x))
        ):
            raise ValueError("data must be a complete finite matrix with n>=3, p>=2")
        base = self.config.base_config
        n = len(x)
        size = (
            2 * n
            if self.config.bootstrap_size is None
            else int(self.config.bootstrap_size)
        )
        self.bootstrap_ratio_ = size / n
        blocks = _as_communities(communities)
        structure = make_scfa_block_structure(blocks, x.shape[1])
        self.selected_indices_ = structure.selected_indices.copy()
        self.communities_ = tuple(b.copy() for b in structure.blocks)
        self.n_features_in_ = x.shape[1]
        self.training_mean_ = x.mean(axis=0)
        self.training_variance_ = x.var(axis=0, ddof=0)
        self.data_mean_ = self.training_mean_[self.selected_indices_]
        self.data_scale_ = (
            x[:, self.selected_indices_].std(axis=0, ddof=0)
            if base.standardize_data
            else np.ones(len(self.selected_indices_))
        )
        if np.any(self.data_scale_ <= 0):
            raise ValueError("selected variables must have positive variance")
        reference = x.copy()
        reference[:, self.selected_indices_] = (
            x[:, self.selected_indices_] - self.data_mean_
        ) / self.data_scale_
        seeds = np.random.SeedSequence(self.config.random_state).spawn(
            self.config.n_bootstrap
        )
        models, original_factors, bootstrap_indices = [], [], []
        bootstrap_seeds, fit_times = [], []
        setup_done = perf_counter()
        for child in seeds:
            seed = int(child.generate_state(1)[0])
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, n, size=size)
            fit_started = perf_counter()
            model = CFAFit(
                replace(
                    base,
                    standardize_data=False,
                    factor_covariance_interval_seed=seed,
                    factor_covariance_interval_draws=100,
                )
            ).fit(reference[indices], self.communities_)
            fit_times.append(perf_counter() - fit_started)
            # Original subjects, not bootstrap row positions, define q_b(F).
            p = model.result_.posterior
            visible = reference[:, self.selected_indices_] - p.data_mean
            mean, covariance = model._local_factors(
                visible, np.ones(len(self.selected_indices_), bool)
            )
            # Global posterior moments and original-row factor summaries do not
            # need B dense M-by-p reconstruction matrices. Keep the scalar RMSE.
            model.result_ = replace(
                model.result_,
                posterior=replace(
                    model.result_.posterior,
                    reconstruction_mean=np.empty((0, len(self.selected_indices_))),
                ),
            )
            model.metadata_["reconstruction_retained"] = False
            original_factors.append((mean, covariance))
            models.append(model)
            bootstrap_indices.append(indices)
            bootstrap_seeds.append(seed)
        self.bootstrap_models_ = tuple(models)
        self._original_factors = tuple(original_factors)
        self.bootstrap_indices_ = np.asarray(bootstrap_indices)
        self.bootstrap_seeds_ = np.asarray(bootstrap_seeds, dtype=np.uint32)
        fit_done = perf_counter()
        self.posterior_ = self._summarize()
        self.member_ = base.member
        self.method_label_ = "bagged " + models[0].method_label_
        self.row_state_probability_ = None
        self.row_overlap_pip_ = None
        self.overlap_budget_ = None
        if base.member != "mf":
            self.row_state_probability_ = np.mean(
                [m.row_state_probability_ for m in models], axis=0
            )
            primary = np.argmax(models[0].result_.posterior.loading_mask, axis=1)
            self.row_overlap_pip_ = (
                1 - self.row_state_probability_[np.arange(len(primary)), primary]
            )
            self.conditional_target_probability_ = self.row_state_probability_.copy()
            self.conditional_target_probability_[np.arange(len(primary)), primary] = 0
            np.divide(
                self.conditional_target_probability_,
                self.row_overlap_pip_[:, None],
                out=self.conditional_target_probability_,
                where=self.row_overlap_pip_[:, None] > 0,
            )
        if base.member == "2f+s":
            self.overlap_budget_ = models[0].overlap_budget_
        self.complexity_ = loading_complexity(
            self.posterior_, self.row_overlap_pip_, self.overlap_budget_
        )
        self.timing_ = {
            "setup_seconds": setup_done - started,
            "bootstrap_fit_seconds": sum(fit_times),
            "optimization_seconds": sum(m.timing_["fit_seconds"] for m in models),
            "interval_seconds": sum(m.timing_["interval_seconds"] for m in models),
            "original_subject_inference_seconds": fit_done
            - setup_done
            - sum(fit_times),
            "summary_seconds": perf_counter() - fit_done,
            "covariance_seconds": 0.0,
            "joint_covariance_seconds": 0.0,
            "prediction_seconds": 0.0,
        }
        runtime = runtime_provenance(models[0].result_.diagnostics)
        runtime["gpu_operation_counts"] = {
            key: sum(
                int((model.result_.diagnostics.gpu_operation_counts or {}).get(key, 0))
                for model in models
            )
            for key in ("factor_updates", "residual_updates")
        }
        runtime["component_count"] = len(models)
        self.metadata_ = {
            "runtime": runtime,
            "package_version": __version__,
            "member": base.member,
            "loading_normalization": base.loading_normalization,
            "n_bootstrap": self.config.n_bootstrap,
            "bootstrap_size": size,
            "n_samples": n,
            "bootstrap_ratio": size / n,
            "random_state": self.config.random_state,
            "bootstrap_seeds": bootstrap_seeds,
            "bootstrap_fit_seconds": fit_times,
            "bootstrap_converged": [m.metadata_["converged"] for m in models],
            "bootstrap_iterations": [m.metadata_["n_iterations"] for m in models],
            "joint_covariance_materialized": False,
            "bootstrap_reconstructions_retained": False,
            "covariance_definition": "average_b E_qb[Lambda Phi Lambda.T + diag(psi)]",
            "posterior_definition": "raw constrained bootstrap mixture; F evaluated on original rows",
            "joint_calibration": "optional qualified blockwise moment adjustment in free coordinates",
            "model_scale": "one training reference scale shared by every bootstrap",
        }
        self._reset_prediction()
        self.metadata_["residual_structure"] = (
            "community_shared" if base.member == "1f" else "variable_specific")
        return self

    def _component_summary(self, b, name):
        if name != "F":
            return self.bootstrap_models_[b].posterior(name)
        mean, cov = self._original_factors[b]
        sd = np.broadcast_to(np.sqrt(np.diag(cov)), mean.shape)
        level = self.config.base_config.credible_interval_level
        z = stats.norm.ppf((1 + level) / 2)
        return PosteriorSummary(
            mean,
            sd,
            mean - z * sd,
            mean + z * sd,
            level,
            "local Gaussian q_b(F)",
            np.ones(mean.shape, bool),
        )

    def _marginal_draws(self, b, name, rows, count, rng):
        """Sample only a row chunk for intervals; not exposed as joint draws."""
        model = self.bootstrap_models_[b]
        p, d = model.result_.posterior, model.result_.diagnostics
        if name == "F":
            mean, cov = self._original_factors[b]
            return (
                rng.multivariate_normal(
                    np.zeros(mean.shape[1]), cov, size=(count, len(rows))
                )
                + mean[rows]
            )
        if name == "Lambda":
            if p.member == "1f" and getattr(d, "enforce_loading_row_sums", True):
                return np.broadcast_to(p.loading_mask[rows], (count, len(rows), p.loading_mean.shape[1])).copy()
            if p.member in ("mf", "1f"):
                return _draw_loading_components(
                    loading_mask=p.loading_mask[rows],
                    probabilities=d.loading_slab_probability[rows],
                    component_means=d.loading_component_means[rows],
                    component_variances=d.loading_component_variances[rows],
                    enforce_loading_row_sums=getattr(d, "enforce_loading_row_sums", True),
                    draws=count,
                    rng=rng,
                )
            return _draw_row_sparse_loadings(
                membership=np.argmax(p.loading_mask[rows], axis=1),
                state_probability=d.row_state_probability[rows],
                active_mean=d.active_state_means[rows],
                active_variance=d.active_state_variances[rows],
                draws=count,
                rng=rng,
                primary_moments=free_primary_moments(model.result_, rows),
            )
        if name == "psi":
            return draw_residual_variance(model.result_, count, rng, indices=rows)
        k = p.loading_mean.shape[1]
        return np.asarray(
            stats.invwishart.rvs(
                d.factor_covariance_posterior_df,
                scale=d.factor_covariance_posterior_scale,
                size=count,
                random_state=rng,
            )
        ).reshape(count, k, k)[:, rows]

    def _summarize(self):
        level = self.config.base_config.credible_interval_level
        alpha = (1 - level) / 2
        out = {}
        for param_id, name in enumerate(PARAMETERS):
            parts = [
                self._component_summary(b, name)
                for b in range(len(self.bootstrap_models_))
            ]
            mean = np.mean([p.mean for p in parts], axis=0)
            variance = np.mean([p.sd**2 + (p.mean - mean) ** 2 for p in parts], axis=0)
            fixed_lambda = self.config.base_config.loading_normalization == "row_sum"
            if self.config.base_config.member == "1f" and (name == "gamma" or (name == "Lambda" and fixed_lambda)):
                out[name] = PosteriorSummary(mean, np.zeros_like(mean), mean.copy(), mean.copy(),
                                            level, parts[0].distribution, parts[0].applicable.copy())
                continue
            if name == "gamma":
                out[name] = bernoulli_summary(mean, level, parts[0].applicable)
                continue
            lower, upper = np.empty_like(mean), np.empty_like(mean)
            for start in range(0, len(mean), self.config.summary_chunk_size):
                rows = np.arange(
                    start, min(start + self.config.summary_chunk_size, len(mean))
                )
                samples = []
                for b in range(len(parts)):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                self.config.random_state,
                                param_id,
                                0 if name == "Phi" or (name == "psi" and self.config.base_config.member == "1f") else start,
                                b,
                                241,
                            ]
                        )
                    )
                    samples.append(
                        self._marginal_draws(
                            b,
                            name,
                            rows,
                            self.config.posterior_draws_per_bootstrap,
                            rng,
                        )
                    )
                pooled = np.concatenate(samples, axis=0)
                lower[rows], upper[rows] = np.quantile(
                    pooled, [alpha, 1 - alpha], axis=0
                )
            out[name] = PosteriorSummary(
                mean,
                np.sqrt(np.maximum(variance, 0)),
                lower,
                upper,
                level,
                "bootstrap mixture marginal",
                parts[0].applicable.copy(),
            )
        return out

    def sample_posterior(self, draws=1000, *, random_state=1, parameters=PARAMETERS):
        self.posterior("F")
        if isinstance(draws, bool) or int(draws) != draws or draws < 1:
            raise ValueError("draws must be a positive integer")
        if (
            isinstance(random_state, bool)
            or int(random_state) != random_state
            or random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer")
        if set(parameters) - set(PARAMETERS):
            raise ValueError("unknown posterior parameter")
        draws = int(draws)
        rng = np.random.default_rng(int(random_state))
        components = rng.integers(len(self.bootstrap_models_), size=draws)
        result = {
            name: np.empty(
                (draws,) + self.posterior(name).mean.shape,
                dtype=np.int8 if name == "gamma" else float,
            )
            for name in parameters
        }
        for b, model in enumerate(self.bootstrap_models_):
            ids = np.flatnonzero(components == b)
            if not len(ids):
                continue
            globals_ = tuple(n for n in parameters if n != "F")
            samples = draw_fitted(
                model.result_, len(ids), int(rng.integers(0, 2**32)), globals_
            )
            for name in globals_:
                result[name][ids] = samples[name]
            if "F" in parameters:
                mean, cov = self._original_factors[b]
                result["F"][ids] = (
                    rng.multivariate_normal(
                        np.zeros(mean.shape[1]), cov, size=(len(ids), len(mean))
                    )
                    + mean
                )
        return result

    def sample_loadings(self, draws=1000, *, random_state=1):
        return self.sample_posterior(
            draws, random_state=random_state, parameters=("Lambda",)
        )["Lambda"]

    @property
    def loadings_(self):
        return self.posterior("Lambda").mean.copy()

    @property
    def factor_covariance_(self):
        return self.posterior("Phi").mean.copy()

    @property
    def factor_covariance_ci_(self):
        p = self.posterior("Phi")
        return p.lower.copy(), p.upper.copy()

    @property
    def residual_variance_(self):
        return self.posterior("psi").mean.copy()

    @property
    def cross_loading_pip_(self):
        return self.posterior("gamma").mean.copy()

    def covariance(self, *, standardized=False, selected_only=False):
        self.posterior("F")
        started = perf_counter()
        result = None
        for model in self.bootstrap_models_:
            matrix = model.covariance(standardized=True, selected_only=True)
            if result is None:
                result = np.zeros_like(matrix)
            result += matrix / len(self.bootstrap_models_)
        if not standardized:
            result *= np.outer(self.data_scale_, self.data_scale_)
        self.timing_["covariance_seconds"] += perf_counter() - started
        if selected_only:
            return result
        full = np.diag(
            np.ones(self.n_features_in_)
            if standardized and self.config.base_config.standardize_data
            else self.training_variance_
        )
        full[np.ix_(self.selected_indices_, self.selected_indices_)] = result
        return full

    @property
    def factor_correlation_(self):
        self.posterior("Phi")
        return np.mean([m.factor_correlation_ for m in self.bootstrap_models_], axis=0)

    @property
    def factor_correlation_ci_(self):
        samples = self.sample_posterior(
            self.config.n_bootstrap * self.config.posterior_draws_per_bootstrap,
            random_state=self.config.random_state,
            parameters=("Phi",),
        )["Phi"]
        scale = np.sqrt(np.diagonal(samples, axis1=1, axis2=2))
        correlation = samples / (scale[:, :, None] * scale[:, None, :])
        alpha = (1 - self.config.base_config.credible_interval_level) / 2
        lo, hi = np.quantile(correlation, [alpha, 1 - alpha], axis=0)
        np.fill_diagonal(lo, 1.0)
        np.fill_diagonal(hi, 1.0)
        return lo, hi

    def transform(self, data):
        self.posterior("F")
        started = perf_counter()
        x = np.asarray(data, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if (
            x.ndim != 2
            or x.shape[1] != self.n_features_in_
            or not np.all(np.isfinite(x))
        ):
            raise ValueError("data must be finite with the fitted feature count")
        reference = x.copy()
        reference[:, self.selected_indices_] = (
            x[:, self.selected_indices_] - self.data_mean_
        ) / self.data_scale_
        mean = np.mean([m.transform(reference) for m in self.bootstrap_models_], axis=0)
        self.timing_["prediction_seconds"] += perf_counter() - started
        return mean

    def _prediction_reference(self, row, masked):
        x, mask = visible_row(row, masked, self.n_features_in_)
        reference = np.full(x.shape, np.nan)
        reference[~mask] = x[~mask]
        selected = self.selected_indices_
        visible = ~mask[selected]
        reference[selected[visible]] = (
            x[selected[visible]] - self.data_mean_[visible]
        ) / self.data_scale_[visible]
        ids = np.flatnonzero(mask)
        scales = np.ones(len(ids))
        centers = np.zeros(len(ids))
        selected_positions = np.isin(ids, selected)
        lookup = {int(j): i for i, j in enumerate(selected)}
        for pos in np.flatnonzero(selected_positions):
            local = lookup[int(ids[pos])]
            scales[pos] = self.data_scale_[local]
            centers[pos] = self.data_mean_[local]
        return reference, mask, ids, scales, centers, selected_positions

    def _posterior_moments(self, row, masked):
        self.posterior("F")
        reference, mask, ids, scales, centers, selected_positions = (
            self._prediction_reference(row, masked)
        )
        means, covariances = [], []
        for model in self.bootstrap_models_:
            result = model._posterior_moments(reference, mask)
            mean = result.mean * scales + centers
            covariance = result.covariance * np.outer(scales, scales)
            if selected_positions.any():
                checked_spd(
                    covariance[np.ix_(selected_positions, selected_positions)],
                    "bagged selected component prediction",
                )
            means.append(mean)
            covariances.append(covariance)
        mean = np.mean(means, axis=0)
        deviations = np.asarray(means) - mean
        covariance = np.mean(covariances, axis=0) + deviations.T @ deviations / len(
            means
        )
        outside = ~selected_positions
        mean[outside] = self.training_mean_[ids[outside]]
        covariance[outside, :] = 0
        covariance[:, outside] = 0
        covariance[np.flatnonzero(outside), np.flatnonzero(outside)] = (
            self.training_variance_[ids[outside]]
        )
        return MaskedPrediction(
            ids,
            mean,
            np.diag(covariance).copy(),
            (covariance + covariance.T) / 2,
            ids[selected_positions],
            ids[outside],
        )

    def _draw_predictive(self, row, masked, draws, random_state, batch_size):
        reference, mask, ids, scales, centers, selected_positions = (
            self._prediction_reference(row, masked)
        )
        streams = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(random_state).spawn(3)
        ]
        components = streams[0].integers(len(self.bootstrap_models_), size=draws)
        samples = np.empty((draws, len(ids)))
        for b, model in enumerate(self.bootstrap_models_):
            rows = np.flatnonzero(components == b)
            if len(rows):
                component = model._draw_predictive(
                    reference,
                    mask,
                    len(rows),
                    int(streams[1].integers(2**32)),
                    batch_size,
                    include_residual_only=False,
                )
                samples[rows] = component * scales + centers
        # Excluded variables use the documented full-training convention, not
        # bootstrap-specific means/variances or artificial factor posteriors.
        outside = ~selected_positions
        if outside.any():
            variance = self.training_variance_[ids[outside]]
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise ValueError(
                    "bagged residual-only prediction: variance must be finite and positive"
                )
            samples[:, outside] = self.training_mean_[ids[outside]] + np.sqrt(
                variance
            ) * streams[2].normal(size=(draws, outside.sum()))
        return samples

    def joint_covariance(
        self, *, parameters=("Lambda", "Phi", "psi"), draws=4000, random_state=1
    ):
        """Build optional raw/adjusted/PSD covariance; all outputs preserve row sums.

        Only continuous global parameters belong here. Discrete gamma and local
        F remain in the actual mixture posterior. Selecting Phi alone is cheap.
        Blockwise adjustment is qualified, not a sparse-model coverage theorem.
        """
        self.posterior("F")
        parameters = tuple(parameters)
        if (
            not parameters
            or len(set(parameters)) != len(parameters)
            or set(parameters) - {"Lambda", "Phi", "psi"}
        ):
            raise ValueError(
                "choose distinct continuous global parameters: Lambda, Phi, psi"
            )
        if int(draws) != draws or draws < 2:
            raise ValueError("at least two draws are required")
        layout = coordinate_layout(
            self.bootstrap_models_[0].result_.posterior, parameters
        )
        labels, free_labels, blocks, columns, selections, dimension = layout
        if max(len(labels), dimension) > self.config.joint_max_parameters:
            raise ValueError(
                "joint covariance exceeds joint_max_parameters; select fewer parameter families or explicitly raise the limit"
            )
        started = perf_counter()
        samples = self.sample_posterior(
            draws, random_state=random_state, parameters=parameters
        )
        free = np.concatenate(
            [samples[n].reshape(draws, -1)[:, selections[n]] for n in parameters],
            axis=1,
        )
        raw = (
            np.atleast_2d(np.cov(free, rowvar=False, ddof=1))
            if dimension
            else np.empty((0, 0))
        )
        adjusted = calibrate_free_covariance(raw, blocks, self.bootstrap_ratio_)
        stabilized = stabilize_fixed_diagonal(adjusted)
        affine = np.zeros((len(labels), dimension))
        for row, entries in enumerate(columns):
            for col, value in entries.items():
                affine[row, col] = value
        self.timing_["joint_covariance_seconds"] += perf_counter() - started
        self.metadata_["joint_covariance_materialized"] = True
        return JointCovariance(
            tuple(labels),
            tuple(free_labels),
            affine @ raw @ affine.T,
            affine @ adjusted @ affine.T,
            affine @ stabilized @ affine.T,
            raw,
            adjusted,
            stabilized,
            affine,
            np.asarray(blocks),
            int(random_state),
            int(draws),
            "Blockwise moment calibration in free coordinates; not a guarantee of coverage for discrete/structured loadings. "
            "Calibrated covariance is not the covariance defining the raw posterior marginal intervals or structural draws.",
        )
