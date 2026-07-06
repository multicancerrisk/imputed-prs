"""Device selection and compute-backend factory.

``torch`` is imported lazily and only when a GPU device is requested/probed, so the CPU
path is fully functional with no ``torch`` installed (the ``gpu`` extra is opt-in). See
:mod:`imputed_prs.compute.backend`.
"""

from __future__ import annotations

import warnings

_VALID_DEVICES = ("auto", "cpu", "mps", "cuda")

# Below this sample count, ``device="auto"`` stays on the CPU backend even when a GPU is
# present. Rationale (measured, ``benchmarks/verify_gpu_scale.py`` accum): the accumulation
# cross-product ``C=ZᵀY`` is memory-bound and only crosses over to a GPU win around here
# (≈0.2× at n=3,202 → ≈2.5× at n=25,000), and a fit at small n runs many tiny per-unit solves
# whose MPS kernel-launch overhead the M-series' fast Accelerate BLAS avoids. An explicit
# ``device="mps"``/``"cuda"`` is always honored; this guard only shapes ``"auto"``.
GPU_AUTO_MIN_SAMPLES = 25_000


def select_device(preference: str = "auto") -> str:
    """Resolve a device preference to a concrete device string.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``. ``"auto"``
            probes for CUDA, then Apple MPS, then falls back to CPU. Probing imports
            ``torch`` lazily; if ``torch`` is not installed, ``"auto"`` resolves to
            ``"cpu"``.

    Returns:
        ``"cpu"``, ``"mps"``, or ``"cuda"``.
    """
    if preference not in _VALID_DEVICES:
        raise ValueError(
            f"device must be one of {_VALID_DEVICES}, got {preference!r}"
        )
    if preference != "auto":
        return preference

    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - a flaky torch/driver probe must not crash fit()
        return "cpu"
    return "cpu"


def get_backend(device: str = "auto", *, strict: bool = False):
    """Return a :class:`~imputed_prs.compute.backend.ComputeBackend` for ``device``.

    Args:
        device: A preference passed to :func:`select_device`.
        strict: If ``True``, a requested GPU device that cannot be constructed raises.
            If ``False`` (default, the fit-time behaviour), it degrades gracefully to
            the CPU backend with a :class:`RuntimeWarning` — a broken/absent GPU should
            never crash a fit. Parity tests pass ``strict=True`` so a broken GPU backend
            fails loudly instead of silently running on CPU.
    """
    resolved = select_device(device)
    if resolved == "cpu":
        from imputed_prs.compute.cpu_backend import CpuBackend

        return CpuBackend()

    try:
        from imputed_prs.compute.gpu_backend import GpuBackend

        return GpuBackend(resolved)
    except Exception as exc:  # noqa: BLE001 - graceful CPU fallback (see strict=)
        if strict:
            raise
        warnings.warn(
            f"GPU backend for device {resolved!r} is unavailable ({type(exc).__name__}: "
            f"{exc}); falling back to the CPU backend.",
            RuntimeWarning,
            stacklevel=2,
        )
        from imputed_prs.compute.cpu_backend import CpuBackend

        return CpuBackend()


def resolve_streaming_backend(device: str = "auto", n_samples: int = 0, *, strict: bool = False):
    """Backend for the streaming fitters, applying the small-``n`` guard to ``device="auto"``.

    ``device="auto"`` engages a GPU only when ``n_samples >= GPU_AUTO_MIN_SAMPLES`` (below that
    the CPU backend is faster — see :data:`GPU_AUTO_MIN_SAMPLES`). Explicit ``"cpu"``/``"mps"``/
    ``"cuda"`` are passed straight through to :func:`get_backend` (the user's choice wins).
    """
    pref = "cpu" if (device == "auto" and n_samples < GPU_AUTO_MIN_SAMPLES) else device
    return get_backend(pref, strict=strict)
