"""CPU contracts and lazy CuPy import of the compute backend."""

import numpy as np
import pytest

from ggbfa import CFAFit, CFAConfig
from ggbfa._backend import GPUBackendUnavailable, resolve_compute_backend


def test_cpu_resolution_never_imports_cupy(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f"CPU resolution imported {name}")

    monkeypatch.setattr("ggbfa._backend.importlib.import_module", unexpected_import)
    backend = resolve_compute_backend("cpu", 0)
    assert backend.resolved_device == "cpu"
    assert backend.name == "numpy/scipy"
    assert backend.actual_gpu_count == 0
    assert backend.operation_counts == {"factor_updates": 0, "residual_updates": 0}


def test_auto_and_explicit_gpu_when_cupy_is_missing(monkeypatch):
    def missing_cupy(name):
        assert name == "cupy"
        raise ImportError("registered missing CuPy")

    monkeypatch.setattr("ggbfa._backend.importlib.import_module", missing_cupy)
    automatic = resolve_compute_backend("auto", 0)
    assert automatic.resolved_device == "cpu"
    assert automatic.device_reason == "cupy_import_unavailable:ImportError"
    with pytest.raises(GPUBackendUnavailable, match="no CPU fallback"):
        resolve_compute_backend("gpu", 0)


@pytest.mark.parametrize("device", ["cuda", "", 1, None])
def test_config_rejects_unknown_device(device):
    with pytest.raises(ValueError):
        CFAConfig(device=device)


@pytest.mark.parametrize("gpu_device_id", [-1, 0.5, True])
def test_config_rejects_invalid_gpu_device_id(gpu_device_id):
    with pytest.raises(ValueError):
        CFAConfig(gpu_device_id=gpu_device_id)


@pytest.mark.parametrize("option", ["mf", "2f", "2f+s", "1f"])
def test_all_loading_options_forward_device_configuration(option):
    kwargs = CFAConfig(
        member=option, device="gpu", gpu_device_id=3
    ).fit_kwargs()
    assert kwargs["device"] == "gpu"
    assert kwargs["gpu_device_id"] == 3


def test_explicit_cpu_fit_records_backend_and_numpy_results():
    rng = np.random.default_rng(1401)
    factors = rng.normal(size=(24, 2))
    loading = np.repeat(np.eye(2), 4, axis=0)
    data = factors @ loading.T + 0.4 * rng.normal(size=(24, 8))
    model = CFAFit(
        CFAConfig(
            member="mf",
            device="cpu",
            max_iter=5,
            min_iter=2,
            factor_covariance_interval_draws=100,
        )
    ).fit(data, [np.arange(4), np.arange(4, 8)])
    runtime = model.metadata_["runtime"]
    assert runtime["requested_device"] == "cpu"
    assert runtime["resolved_device"] == "cpu"
    assert runtime["backend"] == "numpy/scipy"
    assert runtime["actual_gpu_count"] == 0
    assert runtime["gpu_operation_counts"] == {
        "factor_updates": 0,
        "residual_updates": 0,
    }
    assert isinstance(model.loadings_, np.ndarray)
