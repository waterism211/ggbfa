"""Runtime selection and CuPy workspaces for the confirmatory fits.

CuPy is imported only when ``device`` is ``auto`` or ``gpu``.  Public results
remain NumPy arrays; the GPU workspace keeps the selected training matrix on
device and performs the iteration-scale factor and residual sufficient
statistics there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Dict, Optional, Tuple

import numpy as np


DEVICES = ("auto", "cpu", "gpu")


class GPUBackendUnavailable(RuntimeError):
    """Raised when an explicit GPU request cannot be satisfied."""


@dataclass
class ComputeBackend:
    """Resolved numerical backend for one confirmatory fit."""

    requested_device: str
    resolved_device: str
    name: str
    actual_gpu_count: int
    device_reason: str
    gpu_device_id: Optional[int] = None
    gpu_name: Optional[str] = None
    cupy_version: Optional[str] = None
    cuda_runtime_version: Optional[int] = None
    _cupy: object = field(default=None, repr=False)
    operation_counts: Dict[str, int] = field(
        default_factory=lambda: {"factor_updates": 0, "residual_updates": 0}
    )

    @property
    def is_gpu(self) -> bool:
        return self.resolved_device == "gpu"

    def workspace(self, data: np.ndarray) -> Optional["CuPyWorkspace"]:
        if not self.is_gpu:
            return None
        return CuPyWorkspace(self, data)

    def diagnostics(self) -> dict:
        """Return immutable-at-construction provenance for result diagnostics."""

        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "compute_backend": self.name,
            "actual_gpu_count": self.actual_gpu_count,
            "device_reason": self.device_reason,
            "gpu_device_id": self.gpu_device_id,
            "gpu_name": self.gpu_name,
            "cupy_version": self.cupy_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "gpu_operation_counts": dict(self.operation_counts),
        }


class CuPyWorkspace:
    """GPU-resident training matrix and iteration-scale sufficient statistics."""

    def __init__(self, backend: ComputeBackend, data: np.ndarray):
        self.backend = backend
        self.cp = backend._cupy
        with self.cp.cuda.Device(int(backend.gpu_device_id)):
            self.data = self.cp.asarray(data, dtype=self.cp.float64)
            self.cp.cuda.get_current_stream().synchronize()

    def factor_update(
        self,
        phi_precision: np.ndarray,
        expected_precision: np.ndarray,
        loading_second: np.ndarray,
        loading_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute q(F), its second moment, and X.T@E[F] on the GPU."""

        cp = self.cp
        with cp.cuda.Device(int(self.backend.gpu_device_id)):
            tau = cp.asarray(expected_precision, dtype=cp.float64)
            second = cp.asarray(loading_second, dtype=cp.float64)
            loading = cp.asarray(loading_mean, dtype=cp.float64)
            precision = cp.asarray(phi_precision, dtype=cp.float64) + cp.einsum(
                "p,pkl->kl", tau, second, optimize=True
            )
            precision = 0.5 * (precision + precision.T)
            chol = cp.linalg.cholesky(precision)
            identity = cp.eye(precision.shape[0], dtype=cp.float64)
            covariance = cp.linalg.solve(chol.T, cp.linalg.solve(chol, identity))
            covariance = 0.5 * (covariance + covariance.T)
            weighted_loading = loading * tau[:, None]
            mean = (self.data @ weighted_loading) @ covariance
            second_sum = self.data.shape[0] * covariance + mean.T @ mean
            response_sums = self.data.T @ mean
            cp.cuda.get_current_stream().synchronize()
            result = tuple(
                cp.asnumpy(value)
                for value in (covariance, mean, second_sum, response_sums)
            )
        self.backend.operation_counts["factor_updates"] += 1
        return result

    def residual_ss(
        self,
        factor_mean: np.ndarray,
        factor_covariance: np.ndarray,
        loading_mean: np.ndarray,
        loading_second: np.ndarray,
        loading_covariance: np.ndarray,
    ) -> np.ndarray:
        """Compute variable-wise expected residual sums of squares on the GPU."""

        cp = self.cp
        with cp.cuda.Device(int(self.backend.gpu_device_id)):
            factor = cp.asarray(factor_mean, dtype=cp.float64)
            factor_cov = cp.asarray(factor_covariance, dtype=cp.float64)
            loading = cp.asarray(loading_mean, dtype=cp.float64)
            loading_second_gpu = cp.asarray(loading_second, dtype=cp.float64)
            loading_cov_gpu = cp.asarray(loading_covariance, dtype=cp.float64)
            residual = self.data - factor @ loading.T
            factor_gram = factor.T @ factor
            loading_uncertainty = cp.einsum(
                "kl,pkl->p", factor_gram, loading_cov_gpu, optimize=True
            )
            factor_uncertainty = self.data.shape[0] * cp.einsum(
                "kl,pkl->p", factor_cov, loading_second_gpu, optimize=True
            )
            value = (
                cp.sum(residual * residual, axis=0)
                + loading_uncertainty
                + factor_uncertainty
            )
            cp.cuda.get_current_stream().synchronize()
            result = cp.asnumpy(value)
        self.backend.operation_counts["residual_updates"] += 1
        return result


def _cpu_backend(requested: str, reason: str) -> ComputeBackend:
    return ComputeBackend(
        requested_device=requested,
        resolved_device="cpu",
        name="numpy/scipy",
        actual_gpu_count=0,
        device_reason=reason,
    )


def resolve_compute_backend(
    requested_device: str = "auto", gpu_device_id: int = 0
) -> ComputeBackend:
    """Resolve CPU or CuPy before fitting, with no explicit-GPU fallback."""

    if not isinstance(requested_device, str):
        raise TypeError("device must be one of auto, cpu, or gpu")
    requested = requested_device.strip().lower()
    if requested not in DEVICES:
        raise ValueError("device must be one of auto, cpu, or gpu")
    if isinstance(gpu_device_id, bool) or int(gpu_device_id) != gpu_device_id:
        raise ValueError("gpu_device_id must be a nonnegative integer")
    gpu_device_id = int(gpu_device_id)
    if gpu_device_id < 0:
        raise ValueError("gpu_device_id must be a nonnegative integer")
    if requested == "cpu":
        return _cpu_backend(requested, "cpu_requested")

    try:
        cp = importlib.import_module("cupy")
    except Exception as error:
        if requested == "gpu":
            raise GPUBackendUnavailable(
                "GPU was explicitly requested, but CuPy is unavailable; no CPU fallback"
            ) from error
        return _cpu_backend(
            requested, f"cupy_import_unavailable:{type(error).__name__}"
        )

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= gpu_device_id:
            raise RuntimeError(
                f"requested CUDA device {gpu_device_id}, but only {count} are visible"
            )
        with cp.cuda.Device(gpu_device_id):
            properties = cp.cuda.runtime.getDeviceProperties(gpu_device_id)
            raw_name = properties.get(
                "name", properties.get(b"name", "unknown CUDA GPU")
            )
            gpu_name = (
                raw_name.decode("utf-8", errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            probe = cp.arange(4, dtype=cp.float64)
            if float(cp.asnumpy((probe * probe).sum())) != 14.0:
                raise RuntimeError("CuPy device probe returned an incorrect result")
            cp.cuda.get_current_stream().synchronize()
            runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as error:
        if requested == "gpu":
            raise GPUBackendUnavailable(
                f"GPU was explicitly requested, but CUDA device {gpu_device_id} "
                f"could not run CuPy: {error}; no CPU fallback"
            ) from error
        return _cpu_backend(requested, f"cuda_unavailable:{type(error).__name__}")

    return ComputeBackend(
        requested_device=requested,
        resolved_device="gpu",
        name="cupy",
        actual_gpu_count=1,
        device_reason="cupy_cuda_available",
        gpu_device_id=gpu_device_id,
        gpu_name=gpu_name,
        cupy_version=str(cp.__version__),
        cuda_runtime_version=runtime_version,
        _cupy=cp,
    )


__all__ = [
    "ComputeBackend",
    "DEVICES",
    "GPUBackendUnavailable",
    "resolve_compute_backend",
]
