"""Settings of the confirmatory step (one gg-BFA member fitted by variational Bayes)."""

from dataclasses import dataclass
import math

from ._backend import DEVICES
from ._members import normalize_member

LOADING_NORMALIZATIONS = ("row_sum", "free")


@dataclass(frozen=True)
class CFAConfig:
    """Settings of one confirmatory member: ``1f``, ``2f+s``, ``2f`` or ``mf``.

    All members estimate q(Phi).  With the default ``loading_normalization="row_sum"``
    every loading row sums exactly to one (1f: primary loadings one).  With
    ``"free"`` each primary loading is estimated under a
    N(primary_loading_prior_mean, primary_loading_prior_variance) prior, the cross-loading
    priors are unchanged, and the scale of each factor is fixed by these priors only.
    """

    member: str = "2f"
    max_iter: int = 1000
    min_iter: int = 5
    tolerance: float = 1e-6
    credible_interval_level: float = 0.95
    factor_prior_variance: float = 1.0
    standardize_data: bool = True
    cross_loading_inclusion_prior: float = 0.10
    spike_variance: float = 1e-4
    slab_variance: float = 0.25
    residual_prior_shape: float = 0.01
    residual_prior_scale: float = 0.01
    elbo_decrease_tolerance: float = 1e-8
    mixture_interval_method: str = "normal_approx"
    factor_covariance_prior_strength: float = 1.0
    factor_covariance_interval_draws: int = 2000
    factor_covariance_interval_seed: int = 1
    device: str = "auto"
    gpu_device_id: int = 0

    # 2f and 2f+s: one primary plus at most one categorical cross-loading.
    row_overlap_prior_alpha: float = 1.0
    row_overlap_prior_beta: float = 9.0
    target_dirichlet_concentration: float = 1.0
    active_cross_loading_variance: float = 0.25

    # 2f+s: strict per-community budget of rows with a cross-loading.
    row_overlap_prior_probability: float = 0.05
    max_overlap_fraction: float = 0.10

    # Loading normalization: "row_sum" fixes each loading row to sum to one;
    # "free" estimates the primary loadings under a normal prior.
    loading_normalization: str = "row_sum"
    primary_loading_prior_mean: float = 1.0
    primary_loading_prior_variance: float = 1.0

    def __post_init__(self):
        object.__setattr__(
            self, "member", normalize_member(self.member)
        )
        if not isinstance(self.device, str) or self.device.strip().lower() not in DEVICES:
            raise ValueError("device must be one of auto, cpu, or gpu")
        object.__setattr__(self, "device", self.device.strip().lower())
        for name, minimum in [
            ("max_iter", 1),
            ("min_iter", 1),
            ("factor_covariance_interval_draws", 100),
            ("factor_covariance_interval_seed", 0),
            ("gpu_device_id", 0),
        ]:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        if self.min_iter > self.max_iter:
            raise ValueError("min_iter must not exceed max_iter")
        if self.loading_normalization not in LOADING_NORMALIZATIONS:
            raise ValueError("loading_normalization must be 'row_sum' or 'free'")
        if not math.isfinite(self.primary_loading_prior_mean):
            raise ValueError("primary_loading_prior_mean must be finite")
        if not math.isfinite(self.primary_loading_prior_variance) or self.primary_loading_prior_variance <= 0:
            raise ValueError("primary_loading_prior_variance must be finite and positive")

    def _normalization_kwargs(self):
        return {
            "loading_normalization": self.loading_normalization,
            "primary_loading_prior_mean": float(self.primary_loading_prior_mean),
            "primary_loading_prior_variance": float(self.primary_loading_prior_variance),
        }

    def _common_fit_kwargs(self):
        """Return settings for the multiple loading family."""

        return {
            "max_iter": self.max_iter,
            "min_iter": self.min_iter,
            "tolerance": self.tolerance,
            "credible_interval_level": self.credible_interval_level,
            "factor_prior_variance": self.factor_prior_variance,
            "standardize_data": self.standardize_data,
            "cross_loading_inclusion_prior": self.cross_loading_inclusion_prior,
            "spike_variance": self.spike_variance,
            "slab_variance": self.slab_variance,
            "residual_prior_shape": self.residual_prior_shape,
            "residual_prior_scale": self.residual_prior_scale,
            "elbo_decrease_tolerance": self.elbo_decrease_tolerance,
            "mixture_interval_method": self.mixture_interval_method,
            "factor_covariance_prior_strength": self.factor_covariance_prior_strength,
            "device": self.device,
            "gpu_device_id": self.gpu_device_id,
        }

    def fit_kwargs(self):
        """Return the member-specific keyword arguments of the fitting routine."""

        option = normalize_member(self.member)
        interval = {
            "factor_covariance_interval_draws": (self.factor_covariance_interval_draws),
            "factor_covariance_interval_seed": (self.factor_covariance_interval_seed),
        }
        if option == "1f":
            return {
                "member": option,
                **{name: getattr(self, name) for name in (
                    "max_iter", "min_iter", "tolerance", "credible_interval_level",
                    "factor_prior_variance", "standardize_data",
                    "residual_prior_shape", "residual_prior_scale",
                    "elbo_decrease_tolerance", "factor_covariance_prior_strength",
                    "device", "gpu_device_id",
                )},
                **interval,
                **self._normalization_kwargs(),
            }
        if option == "mf":
            return {
                **self._common_fit_kwargs(),
                "member": option,
                **interval,
                **self._normalization_kwargs(),
            }

        shared = {
            "member": option,
            "max_iter": self.max_iter,
            "min_iter": self.min_iter,
            "tolerance": self.tolerance,
            "credible_interval_level": self.credible_interval_level,
            "factor_prior_variance": self.factor_prior_variance,
            "standardize_data": self.standardize_data,
            "target_dirichlet_concentration": (self.target_dirichlet_concentration),
            "active_cross_loading_variance": (self.active_cross_loading_variance),
            "residual_prior_shape": self.residual_prior_shape,
            "residual_prior_scale": self.residual_prior_scale,
            "elbo_decrease_tolerance": self.elbo_decrease_tolerance,
            "factor_covariance_prior_strength": (self.factor_covariance_prior_strength),
            "device": self.device,
            "gpu_device_id": self.gpu_device_id,
            **interval,
            **self._normalization_kwargs(),
        }
        if option == "2f":
            return {
                **shared,
                "row_overlap_prior_alpha": self.row_overlap_prior_alpha,
                "row_overlap_prior_beta": self.row_overlap_prior_beta,
            }
        return {
            **shared,
            "row_overlap_prior_probability": (self.row_overlap_prior_probability),
            "max_overlap_fraction": self.max_overlap_fraction,
        }
