"""Sample-free ("sufficient statistics") training kernels for scaling.

This package contracts the sample dimension into local Gram blocks so that
per-variant / per-region linear models can be fit without ever materializing the
full reference dosage matrix. See ``compute/gram_solve.py`` for the sample-free
solver (Phase 2) and ``compute/sufficient_stats.py`` for the streaming driver.

The batched Gram kernels dispatch through a :class:`~imputed_prs.compute.backend.ComputeBackend`
(Phase 3): ``compute/cpu_backend.py`` (numpy/scipy, the oracle) or ``compute/gpu_backend.py``
(torch on MPS/CUDA). :func:`~imputed_prs.compute.device.get_backend` selects one for a device.
"""

from imputed_prs.compute.backend import ComputeBackend
from imputed_prs.compute.device import get_backend, select_device

__all__ = ["ComputeBackend", "get_backend", "select_device"]
