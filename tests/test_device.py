"""Device selection + backend factory (``compute/device.py``).

These run with or without ``torch`` installed: the torch-absent behaviour is simulated
with monkeypatching, so the file is green in both the CPU ``.venv`` and the ``.venv-gpu``.
"""

import sys
import warnings

import pytest

from imputed_prs.compute import ComputeBackend, get_backend, select_device
from imputed_prs.compute.cpu_backend import CpuBackend
from imputed_prs.compute.device import GPU_AUTO_MIN_SAMPLES, resolve_streaming_backend


def test_select_device_explicit():
    assert select_device("cpu") == "cpu"
    assert select_device("mps") == "mps"
    assert select_device("cuda") == "cuda"


def test_select_device_invalid():
    with pytest.raises(ValueError):
        select_device("tpu")


def test_select_device_auto_without_torch(monkeypatch):
    """``auto`` resolves to ``cpu`` when torch cannot be imported."""
    monkeypatch.setitem(sys.modules, "torch", None)  # -> `import torch` raises ImportError
    assert select_device("auto") == "cpu"


def test_get_backend_cpu_satisfies_protocol():
    b = get_backend("cpu")
    assert b.device_name == "cpu"
    assert isinstance(b, CpuBackend)
    assert isinstance(b, ComputeBackend)


class _BoomBackend:
    def __init__(self, *a, **k):
        raise RuntimeError("no gpu here")


def test_get_backend_gpu_unavailable_falls_back(monkeypatch):
    """A GPU device whose backend can't be constructed degrades to CPU (non-strict)."""
    monkeypatch.setattr("imputed_prs.compute.gpu_backend.GpuBackend", _BoomBackend)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = get_backend("mps")
    assert b.device_name == "cpu"
    assert any("falling back" in str(x.message) for x in w)


def test_get_backend_gpu_unavailable_strict_raises(monkeypatch):
    monkeypatch.setattr("imputed_prs.compute.gpu_backend.GpuBackend", _BoomBackend)
    with pytest.raises(RuntimeError):
        get_backend("mps", strict=True)


# --------------------------------------------------------------------------- size guard
class _SentinelGpu:
    """Stand-in GPU backend so the guard's engage/skip decision is testable torch-free."""

    device_name = "mps"

    def __init__(self, *a, **k):
        pass


def test_size_guard_small_n_auto_stays_cpu():
    """Below GPU_AUTO_MIN_SAMPLES, ``auto`` never touches the GPU path (CPU regardless of torch)."""
    b = resolve_streaming_backend("auto", GPU_AUTO_MIN_SAMPLES - 1)
    assert isinstance(b, CpuBackend)


def test_size_guard_explicit_device_bypasses_guard(monkeypatch):
    """An explicit GPU device is honored even at tiny n (the guard only shapes ``auto``)."""
    monkeypatch.setattr("imputed_prs.compute.gpu_backend.GpuBackend", _SentinelGpu)
    b = resolve_streaming_backend("mps", 1)
    assert isinstance(b, _SentinelGpu)


def test_size_guard_boundary_engages_gpu_only_at_scale(monkeypatch):
    """``auto`` engages the GPU at/above the threshold and stays on CPU just below it."""
    monkeypatch.setattr(
        "imputed_prs.compute.device.select_device",
        lambda pref: "mps" if pref == "auto" else pref,
    )
    monkeypatch.setattr("imputed_prs.compute.gpu_backend.GpuBackend", _SentinelGpu)
    below = resolve_streaming_backend("auto", GPU_AUTO_MIN_SAMPLES - 1)
    at = resolve_streaming_backend("auto", GPU_AUTO_MIN_SAMPLES)
    assert isinstance(below, CpuBackend)   # guard forces cpu; select_device("auto") never reached
    assert isinstance(at, _SentinelGpu)    # guard lets "auto" through -> GPU path
