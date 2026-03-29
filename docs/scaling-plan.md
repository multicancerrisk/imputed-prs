# Scaling imputed-prs to 2M+ Variant PRSs with CPU/GPU Support

This document provides a phased implementation plan for refactoring the `imputed-prs` library to efficiently handle Polygenic Risk Scores with 2M+ variants using the 1000 Genomes Project reference dataset (~2,504 samples). The plan supports both CPU and GPU (Apple Metal/MPS and NVIDIA CUDA) computation. **Note:** You are currently developing on a device with Macbook M5 Pro chip.

Each phase is designed to be implemented independently by a Claude Code agent, building on the previous phase. Each phase produces testable functionality and maintains backward compatibility with existing tests.

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Scaling Bottlenecks](#scaling-bottlenecks)
3. [Phase 0: Compute Backend Abstraction](#phase-0-compute-backend-abstraction)
4. [Phase 1: Chromosome-Level Batching](#phase-1-chromosome-level-batching)
5. [Phase 2: Streaming Calibration](#phase-2-streaming-calibration)
6. [Phase 3: 10-Fold Outer Cross-Validation](#phase-3-10-fold-outer-cross-validation)
7. [Phase 4: GPU-Accelerated Training](#phase-4-gpu-accelerated-training)
8. [Phase 5: Checkpointing and Progress](#phase-5-checkpointing-and-progress)
9. [Phase 6: Optimized Prediction](#phase-6-optimized-prediction)
10. [Phase 7: Integration and Documentation](#phase-7-integration-and-documentation)
11. [Phase Dependency Graph](#phase-dependency-graph)
12. [Memory Budget Summary](#memory-budget-summary)
13. [Runtime Estimates](#runtime-estimates)

---

## Current Architecture Analysis

### Core Algorithm

The library trains independent ElasticNet regression models for each missing PRS variant, predicting dosages from observed platform variants within a local genomic window ($\pm 1$ Mb). The imputed PRS is:

$$S_{\text{imputed}} = \sum_{j \in \mathcal{O}} z_j \beta_j + \sum_{j \in \mathcal{M}} \hat{x}_j \beta_j$$

where $\hat{x}_j = \mathbf{z}^\top \mathbf{w}_j + \gamma_j$ is the linearly imputed dosage.

### Key Source Files

| File | Role | Lines |
|------|------|-------|
| `imputed_prs/models/elastic_net.py` | Per-variant ElasticNet + K-fold CV | 210 |
| `imputed_prs/models/trainer.py` | Training loop over all missing variants | 422 |
| `imputed_prs/core/harmonizer.py` | Variant partitioning and window filtering | 672 |
| `imputed_prs/core/linear_imputation_prs.py` | Main API class orchestrating fit/predict/export | 1019 |
| `imputed_prs/models/predictor.py` | Per-user PRS prediction | 236 |
| `imputed_prs/models/tuning.py` | Hyperparameter grid search | 462 |
| `imputed_prs/evaluation/calibration.py` | CV-based calibration regression | ~150 |
| `imputed_prs/io/genotype_loader.py` | VCF/PLINK loading into dense numpy | ~300 |
| `imputed_prs/models/bounding.py` | Dosage truncation with variance adjustment | 199 |

### Data Flow

```
Reference VCF/PLINK
    |
    v
genotype_loader.py --> dense np.ndarray (n_samples x n_variants)
    |
    v
linear_imputation_prs.py:fit()
    |-- partition_variants() --> observed set O, missing set M
    |-- Build Z (platform dosages) and X (missing dosages) as dense slices
    |-- global_hyperparameter_search() on sampled variants
    |-- ImputationModelTrainer.fit_all_variants()
    |     |-- For each missing variant (joblib parallel):
    |     |     |-- filter_to_local_window() --> predictor indices
    |     |     |-- fit_single_variant_model() --> ElasticNet + 5-fold CV
    |     |     |-- Return model + cv_predictions
    |     |-- Collect all models and cv_predictions into memory
    |-- Compute calibration from cv_predictions + true dosages
    |-- Store fitted state
```

---

## Scaling Bottlenecks

### Memory Bottlenecks (2M variants, 2,504 samples)

| Component | Current Size | Location |
|-----------|-------------|----------|
| Full dosage matrix (all variants) | ~25 GB | `genotype_loader.py` loads into single `np.ndarray` |
| Z matrix (platform variants, ~600K) | ~6 GB | `linear_imputation_prs.py:319` `genotype_data.dosage_matrix[:, platform_variant_indices]` |
| X matrix (missing variants, ~1.4M) | ~14 GB | `linear_imputation_prs.py:338` `genotype_data.dosage_matrix[:, missing_variant_indices]` |
| CV predictions (all variants) | ~40 GB | `TrainingResult.cv_predictions: Dict[str, np.ndarray]` -- 2M arrays of length 2,504 |
| X_full for calibration | ~20 GB | `linear_imputation_prs.py:449` `genotype_data.dosage_matrix[:, all_prs_indices]` |
| **Peak total** | **~65 GB** | |

### Compute Bottlenecks

| Bottleneck | Location | Complexity | Impact |
|-----------|----------|------------|--------|
| ElasticNet per variant | `elastic_net.py:147-186` | $O(k \cdot \text{cv\_folds} \cdot w \cdot n)$ | 2M variants x 5 folds = 10M model fits |
| `filter_to_local_window()` | `harmonizer.py:461` | $O(n_{\text{platform}})$ per call via pandas `apply(lambda)` | Called 2M times; pandas overhead dominates |
| Hyperparameter grid search | `tuning.py:177-218` | $O(\text{grid\_size} \cdot \text{sample\_variants} \cdot \text{cv\_folds} \cdot w \cdot n)$ | Retrains for each (l1_ratio, alpha) pair |
| Python prediction loop | `predictor.py:83-123` | $O(k \cdot w)$ with dict lookups per predictor | 2M model iterations with Python overhead |

---

## Phase 0: Compute Backend Abstraction

**Goal**: Create a clean interface that lets the same algorithmic code run on CPU (numpy/sklearn) or GPU (PyTorch on CUDA/MPS) without rewriting the core algorithm. This is the foundation for all subsequent phases.

**Estimated effort**: Medium (1-2 sessions)

### Phase 0.1: Backend Protocol and Device Selection

**New file**: `imputed_prs/compute/__init__.py`

```python
from imputed_prs.compute.backend import ComputeBackend
from imputed_prs.compute.device import select_device, get_backend

__all__ = ["ComputeBackend", "select_device", "get_backend"]
```

**New file**: `imputed_prs/compute/backend.py`

```python
"""Compute backend protocol for CPU/GPU abstraction."""

from typing import Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol for compute backends (CPU or GPU).

    All methods accept and return numpy arrays. GPU backends handle
    the transfer to/from device internally.
    """

    @property
    def device_name(self) -> str:
        """Device identifier: 'cpu', 'cuda', or 'mps'."""
        ...

    def fit_elastic_net(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        l1_ratio: float,
        alpha: float,
        max_iter: int = 10000,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Fit ElasticNet on training data and predict on validation data.

        Args:
            X_train: Training predictors (n_train, n_features).
            y_train: Training targets (n_train,).
            X_val: Validation predictors (n_val, n_features).
            l1_ratio: L1/L2 mixing (0=Ridge, 1=Lasso).
            alpha: Regularization strength.
            max_iter: Maximum iterations.

        Returns:
            (coefficients, intercept, val_predictions)
        """
        ...

    def fit_elastic_net_final(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l1_ratio: float,
        alpha: float,
        max_iter: int = 10000,
    ) -> Tuple[np.ndarray, float]:
        """Fit final ElasticNet model on all data.

        Returns:
            (coefficients, intercept)
        """
        ...
```

**New file**: `imputed_prs/compute/device.py`

```python
"""Device selection and backend factory."""


def select_device(preference: str = "auto") -> str:
    """Select compute device.

    Args:
        preference: One of "auto", "cpu", "cuda", "mps".
            "auto" checks CUDA first, then MPS, then falls back to CPU.

    Returns:
        Device string: "cpu", "cuda", or "mps".
    """
    if preference != "auto":
        return preference

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def get_backend(device: str = "auto"):
    """Get a ComputeBackend for the specified device.

    Args:
        device: "auto", "cpu", "cuda", or "mps".

    Returns:
        ComputeBackend instance.
    """
    resolved = select_device(device)

    if resolved == "cpu":
        from imputed_prs.compute.cpu_backend import CpuBackend
        return CpuBackend()
    else:
        from imputed_prs.compute.gpu_backend import GpuBackend
        return GpuBackend(resolved)
```

#### Test Plan for 0.1

**New file**: `tests/test_device.py`

```python
def test_select_device_cpu():
    assert select_device("cpu") == "cpu"

def test_select_device_auto_no_torch(monkeypatch):
    """With torch unavailable, auto should return cpu."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_device("auto") == "cpu"

def test_get_backend_cpu():
    backend = get_backend("cpu")
    assert backend.device_name == "cpu"
```

---

### Phase 0.2: CPU Backend (sklearn wrapper)

**New file**: `imputed_prs/compute/cpu_backend.py`

```python
"""CPU compute backend using scikit-learn ElasticNet."""

from typing import Tuple
import numpy as np
from sklearn.linear_model import ElasticNet


class CpuBackend:
    """CPU backend wrapping sklearn's ElasticNet coordinate descent."""

    @property
    def device_name(self) -> str:
        return "cpu"

    def fit_elastic_net(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        l1_ratio: float,
        alpha: float,
        max_iter: int = 10000,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=max_iter,
        )
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        return model.coef_.copy(), float(model.intercept_), val_preds

    def fit_elastic_net_final(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l1_ratio: float,
        alpha: float,
        max_iter: int = 10000,
    ) -> Tuple[np.ndarray, float]:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=max_iter,
        )
        model.fit(X, y)
        return model.coef_.copy(), float(model.intercept_)
```

#### Test Plan for 0.2

**New file**: `tests/test_cpu_backend.py`

```python
def test_cpu_backend_matches_sklearn():
    """CpuBackend must produce identical results to direct sklearn usage."""
    # Generate synthetic data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = X @ [1.0, -0.5, 0.0, 0.3, 0.0] + rng.standard_normal(100) * 0.1

    backend = CpuBackend()
    coef, intercept, preds = backend.fit_elastic_net(
        X[:80], y[:80], X[80:], l1_ratio=0.5, alpha=0.01
    )
    # Compare against direct sklearn
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=10000)
    model.fit(X[:80], y[:80])
    np.testing.assert_allclose(coef, model.coef_, atol=1e-10)

def test_cpu_backend_final_fit():
    """fit_elastic_net_final uses all data."""
    ...

def test_cpu_backend_protocol():
    """CpuBackend satisfies ComputeBackend protocol."""
    assert isinstance(CpuBackend(), ComputeBackend)
```

---

### Phase 0.3: GPU Backend (PyTorch FISTA)

**New file**: `imputed_prs/compute/gpu_backend.py`

Implements ElasticNet via FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) in PyTorch. FISTA is chosen because:
1. Uses only matrix multiplications and element-wise operations -- fully compatible with CUDA and MPS
2. Unlike coordinate descent (sklearn), it is parallelizable on GPU
3. Well-understood convergence for convex objectives

**Algorithm**: Proximal gradient descent with Nesterov momentum for the elastic net objective:

$$\min_{\mathbf{w}, \gamma} \frac{1}{2n} \|\mathbf{X}\mathbf{w} + \gamma\mathbf{1} - \mathbf{y}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{w}\|_2^2$$

```python
"""GPU compute backend using PyTorch FISTA for ElasticNet."""

import numpy as np
from typing import Tuple


class GpuBackend:
    """GPU backend using FISTA proximal gradient descent in PyTorch.

    Supports both CUDA and Apple Metal (MPS) via torch.device.
    Uses float32 on MPS (float64 not supported), float64 on CUDA.
    """

    def __init__(self, device: str = "cuda"):
        import torch
        self._device = torch.device(device)
        # MPS only supports float32
        self._dtype = torch.float32 if device == "mps" else torch.float64

    @property
    def device_name(self) -> str:
        return str(self._device)

    def _soft_threshold(self, w, threshold):
        """Proximal operator for L1: sign(w) * max(|w| - threshold, 0)."""
        import torch
        return torch.sign(w) * torch.clamp(torch.abs(w) - threshold, min=0.0)

    def _fista_elastic_net(
        self, X, y, l1_ratio, alpha, max_iter, tol=1e-5
    ):
        """FISTA solver for elastic net on GPU tensors.

        Args:
            X: (n_samples, n_features) tensor on device.
            y: (n_samples,) tensor on device.
            l1_ratio: L1/L2 mixing parameter.
            alpha: Regularization strength.
            max_iter: Maximum FISTA iterations.
            tol: Convergence tolerance on coefficient change.

        Returns:
            (w, intercept) where w is (n_features,) and intercept is scalar.
        """
        import torch
        n, p = X.shape

        # Center y for intercept
        y_mean = y.mean()
        y_centered = y - y_mean
        X_mean = X.mean(dim=0)
        X_centered = X - X_mean

        # Lipschitz constant: L = ||X^T X|| / n + alpha * (1 - l1_ratio)
        # Use spectral norm estimate via power iteration for efficiency
        # Conservative: use Frobenius norm (works on MPS, unlike eigvalsh)
        XtX = X_centered.T @ X_centered
        L = torch.trace(XtX).item() / n + alpha * (1 - l1_ratio)
        if L < 1e-12:
            L = 1.0  # Avoid division by zero

        step_size = 1.0 / L
        l1_threshold = step_size * alpha * l1_ratio

        # Initialize
        w = torch.zeros(p, device=self._device, dtype=self._dtype)
        w_prev = w.clone()
        t = 1.0

        gradient_base = X_centered.T  # (p, n) -- reuse across iterations

        for iteration in range(max_iter):
            # Nesterov momentum
            t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
            momentum = (t - 1) / t_new
            v = w + momentum * (w - w_prev)

            # Gradient of smooth part: (1/n) * X^T(Xv - y) + alpha*(1-l1_ratio)*v
            residual = X_centered @ v - y_centered
            grad = gradient_base @ residual / n + alpha * (1 - l1_ratio) * v

            # Proximal gradient step
            w_prev = w.clone()
            w = self._soft_threshold(v - step_size * grad, l1_threshold)
            t = t_new

            # Check convergence
            if iteration > 0 and torch.max(torch.abs(w - w_prev)).item() < tol:
                break

        # Recover intercept
        intercept = y_mean - X_mean @ w

        return w, intercept

    def fit_elastic_net(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        l1_ratio: float,
        alpha: float,
        max_iter: int = 10000,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        import torch

        X_t = torch.tensor(X_train, device=self._device, dtype=self._dtype)
        y_t = torch.tensor(y_train, device=self._device, dtype=self._dtype)
        X_v = torch.tensor(X_val, device=self._device, dtype=self._dtype)

        w, intercept = self._fista_elastic_net(X_t, y_t, l1_ratio, alpha, max_iter)

        # Predict on validation
        val_preds = (X_v @ w + intercept).cpu().numpy()
        coef = w.cpu().numpy()

        return coef.astype(np.float64), float(intercept.item()), val_preds.astype(np.float64)

    def fit_elastic_net_final(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l1_ratio: float,
        alpha: float,
        max_iter: int = 10000,
    ) -> Tuple[np.ndarray, float]:
        import torch

        X_t = torch.tensor(X, device=self._device, dtype=self._dtype)
        y_t = torch.tensor(y, device=self._device, dtype=self._dtype)

        w, intercept = self._fista_elastic_net(X_t, y_t, l1_ratio, alpha, max_iter)

        return w.cpu().numpy().astype(np.float64), float(intercept.item())
```

**MPS-specific constraints**:
- `float64` is NOT supported on MPS; use `float32` (acceptable since genotype dosages are inherently low-precision: 0, 1, 2)
- `torch.linalg.eigvalsh` is unavailable on MPS; use `torch.trace(XtX)` as conservative Lipschitz estimate
- Unified memory on Apple Silicon means GPU tensors share RAM with CPU -- no separate VRAM budget

#### Test Plan for 0.3

**New file**: `tests/test_gpu_backend.py`

```python
import pytest
import numpy as np

torch = pytest.importorskip("torch")

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    true_w = np.array([1.0, -0.5, 0.0, 0.3, 0.0, 0.8, 0.0, -0.2, 0.0, 0.1])
    y = X @ true_w + 0.5 + rng.standard_normal(200) * 0.1
    return X, y, true_w

def _get_gpu_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    pytest.skip("No GPU available")

class TestGpuBackend:
    def test_fista_converges_to_sklearn(self, synthetic_data):
        """GPU FISTA solution should be close to sklearn coordinate descent."""
        X, y, _ = synthetic_data
        device = _get_gpu_device()

        from imputed_prs.compute.gpu_backend import GpuBackend
        from imputed_prs.compute.cpu_backend import CpuBackend

        gpu = GpuBackend(device)
        cpu = CpuBackend()

        coef_gpu, int_gpu, pred_gpu = gpu.fit_elastic_net(
            X[:160], y[:160], X[160:], l1_ratio=0.5, alpha=0.01
        )
        coef_cpu, int_cpu, pred_cpu = cpu.fit_elastic_net(
            X[:160], y[:160], X[160:], l1_ratio=0.5, alpha=0.01
        )

        # float32 on MPS means lower tolerance
        tol = 1e-2 if device == "mps" else 1e-4
        np.testing.assert_allclose(coef_gpu, coef_cpu, atol=tol)
        np.testing.assert_allclose(pred_gpu, pred_cpu, atol=tol)

    def test_sparse_coefficients(self, synthetic_data):
        """High alpha should produce sparse solutions (many zero coefficients)."""
        X, y, _ = synthetic_data
        device = _get_gpu_device()
        gpu = GpuBackend(device)
        coef, _, _ = gpu.fit_elastic_net(
            X[:160], y[:160], X[160:], l1_ratio=0.9, alpha=0.5
        )
        assert np.sum(np.abs(coef) < 1e-6) > 3  # At least some zeros

    def test_protocol_compliance(self):
        device = _get_gpu_device()
        from imputed_prs.compute.backend import ComputeBackend
        gpu = GpuBackend(device)
        assert isinstance(gpu, ComputeBackend)
```

---

### Phase 0.4: Refactor elastic_net.py to Use Backend

**Modified file**: `imputed_prs/models/elastic_net.py`

Add an optional `backend` parameter to `fit_single_variant_model()`. The default is `None`, which creates a `CpuBackend()` internally for backward compatibility.

```python
def fit_single_variant_model(
    target_dosages: np.ndarray,
    predictor_dosages: np.ndarray,
    l1_ratio: float = 0.5,
    alpha: float = 0.01,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    backend: Optional["ComputeBackend"] = None,  # NEW PARAMETER
) -> SingleVariantModelResult:
```

**Key change in the CV loop** (lines 147-176):

Replace:
```python
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, ...)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

With:
```python
if backend is None:
    from imputed_prs.compute.cpu_backend import CpuBackend
    backend = CpuBackend()

coef, intercept, y_pred = backend.fit_elastic_net(
    X_train, y_train, X_val, l1_ratio=l1_ratio, alpha=alpha
)
```

And for the final model (lines 179-186):

```python
final_coef, final_intercept = backend.fit_elastic_net_final(
    X_valid, y_valid, l1_ratio=l1_ratio, alpha=alpha
)
```

**Backward compatibility**: All existing callers pass no `backend` argument and get the default `CpuBackend()`, producing identical results to the current sklearn-direct code.

#### Test Plan for 0.4

- **All 27 existing test files must pass unchanged.** This is the critical validation.
- New test: explicitly pass `CpuBackend()` and verify identical results to `backend=None`.
- New test: pass `GpuBackend("mps")` or `GpuBackend("cuda")` and verify results within tolerance.

### Phase 0 Dependency Changes

**Modified file**: `pyproject.toml`

```toml
[project.optional-dependencies]
gpu = [
    "torch>=2.0.0",
]
```

The GPU backend imports `torch` lazily -- CPU-only users never need it installed.

---

## Phase 1: Chromosome-Level Batching

**Goal**: Break the monolithic all-chromosomes-at-once loading and training into per-chromosome streaming, reducing peak memory from ~25 GB to ~2 GB.

**Estimated effort**: Large (2-3 sessions)

**Depends on**: Phase 0 (for backend parameter plumbing)

### Phase 1.1: Window Index Pre-computation

**New file**: `imputed_prs/core/window_index.py`

The current `filter_to_local_window()` in `harmonizer.py` (line 461) runs `variant_info["chromosome"].apply(lambda x: _normalize_chromosome(str(x)))` on **every call**. With 2M missing variants, this is catastrophic -- each call is O(n_platform_variants) with pandas overhead.

Replace with a pre-computed sorted index using `np.searchsorted` for O(log n) lookups:

```python
"""Pre-computed spatial index for fast genomic window queries."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from imputed_prs.core.harmonizer import _normalize_chromosome, WindowFilterResult


@dataclass
class ChromosomeIndex:
    """Sorted index for one chromosome's platform variants."""
    chromosome: str
    positions: np.ndarray       # sorted int64 positions
    variant_ids: List[str]      # aligned with positions
    original_indices: np.ndarray  # indices into source DataFrame
    _sort_order: np.ndarray     # argsort for position ordering

    def query(
        self,
        target_pos: int,
        window_size: int = 1_000_000,
        exclude_position: bool = True,
        max_variants: Optional[int] = None,
    ) -> WindowFilterResult:
        """Find variants within [target_pos - window_size, target_pos + window_size].

        Uses np.searchsorted for O(log n) lookup.

        Args:
            target_pos: Target genomic position.
            window_size: Window size in base pairs on each side.
            exclude_position: Exclude variants at exact target position.
            max_variants: If set, keep only the closest N variants.

        Returns:
            WindowFilterResult with variant_ids, indices, distances.
        """
        lo = np.searchsorted(self.positions, target_pos - window_size, side="left")
        hi = np.searchsorted(self.positions, target_pos + window_size, side="right")

        if lo >= hi:
            return WindowFilterResult(
                variant_ids=[], variant_indices=np.array([], dtype=int),
                distances=np.array([], dtype=int), n_variants=0,
            )

        pos_slice = self.positions[lo:hi]
        idx_slice = self.original_indices[lo:hi]
        id_slice = self.variant_ids[lo:hi]
        distances = np.abs(pos_slice - target_pos)

        if exclude_position:
            mask = pos_slice != target_pos
            pos_slice = pos_slice[mask]
            idx_slice = idx_slice[mask]
            id_slice = [id_slice[i] for i, m in enumerate(mask) if m]
            distances = distances[mask]

        if max_variants is not None and len(idx_slice) > max_variants:
            closest = np.argsort(distances)[:max_variants]
            idx_slice = idx_slice[closest]
            distances = distances[closest]
            id_slice = [id_slice[i] for i in closest]

        return WindowFilterResult(
            variant_ids=list(id_slice),
            variant_indices=idx_slice,
            distances=distances,
            n_variants=len(idx_slice),
        )


def build_window_indices(
    platform_variant_info: pd.DataFrame,
) -> Dict[str, ChromosomeIndex]:
    """Build per-chromosome sorted indices for fast window queries.

    Args:
        platform_variant_info: DataFrame with variant_id, chromosome, position.

    Returns:
        Dict mapping normalized chromosome string to ChromosomeIndex.
    """
    indices = {}

    # Normalize chromosomes once
    chroms = platform_variant_info["chromosome"].apply(
        lambda x: _normalize_chromosome(str(x))
    ).values

    for chrom in np.unique(chroms):
        mask = chroms == chrom
        chrom_df = platform_variant_info[mask]

        positions = chrom_df["position"].values.astype(np.int64)
        sort_order = np.argsort(positions)

        indices[chrom] = ChromosomeIndex(
            chromosome=chrom,
            positions=positions[sort_order],
            variant_ids=[chrom_df.iloc[i]["variant_id"] for i in sort_order],
            original_indices=np.where(mask)[0][sort_order],
            _sort_order=sort_order,
        )

    return indices
```

**Performance comparison**:
- Current: 2M calls x O(600K) pandas apply = ~hours
- New: 2M calls x O(log 600K) binary search = ~seconds

#### Test Plan for 1.1

**New file**: `tests/test_window_index.py`

```python
class TestWindowIndex:
    def test_matches_filter_to_local_window(self):
        """WindowIndex.query() must return identical results to filter_to_local_window()."""
        # Create platform_variant_info with 1000 variants across 3 chromosomes
        # For each of 100 random target positions, verify both methods return
        # the same variant_ids and indices (order may differ)
        ...

    def test_binary_search_correctness(self):
        """Verify boundary conditions of the binary search."""
        # Variants at positions [100, 200, 300, 400, 500]
        # Query at 250 with window=100 should return [200, 300]
        # Query at 100 with window=0, exclude=True should return []
        ...

    def test_max_variants_closest(self):
        """max_variants should select the closest variants."""
        ...

    def test_empty_chromosome(self):
        """Query on chromosome with no variants should return empty."""
        ...

    def test_benchmark_10k_queries(self):
        """10,000 queries on 100K variants should complete in < 1 second."""
        import time
        # Build index with 100K variants
        # Time 10K random queries
        # assert elapsed < 1.0
        ...
```

---

### Phase 1.2: Chromosome-Aware Genotype Loader

**New file**: `imputed_prs/io/chromosome_loader.py`

```python
"""Chromosome-level genotype loading for memory-efficient processing."""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from imputed_prs.core.harmonizer import _normalize_chromosome
from imputed_prs.core.types import GenotypeData


def get_chromosomes_in_file(
    path: Union[str, Path],
) -> List[str]:
    """Get list of chromosomes present in a genotype file.

    For VCF: reads the header/index.
    For PLINK: reads the .bim file.

    Returns:
        Sorted list of normalized chromosome strings.
    """
    ...


def load_genotypes_by_chromosome(
    path: Union[str, Path],
    variant_ids: Optional[Set[str]] = None,
    chromosomes: Optional[List[str]] = None,
) -> Iterator[Tuple[str, GenotypeData]]:
    """Yield (chromosome, GenotypeData) pairs, one chromosome at a time.

    This is the memory-efficient alternative to load_genotypes().
    Only one chromosome's data is in memory at a time.

    Args:
        path: Path to VCF or PLINK file.
        variant_ids: If provided, only load these variants.
        chromosomes: If provided, only load these chromosomes.

    Yields:
        (chromosome_str, GenotypeData) tuples in chromosome order.
    """
    ...
```

**Implementation notes**:
- For VCF with tabix index (`.tbi`): use `cyvcf2.VCF` region queries `vcf("chr1:1-250000000")` to load one chromosome at a time without reading the entire file
- For PLINK: read `.bim` to identify variant indices per chromosome, use `pandas_plink.read_plink` with variant filtering
- For unindexed VCF: single pass with chromosome grouping (accumulate variants until chromosome changes)

#### Test Plan for 1.2

**New file**: `tests/test_chromosome_loader.py`

```python
class TestChromosomeLoader:
    def test_union_equals_full_load(self, tmp_path):
        """Union of per-chromosome loads must equal loading all at once."""
        # Create a synthetic VCF with 3 chromosomes, 50 variants each
        # Load all at once with load_genotypes()
        # Load per-chromosome with load_genotypes_by_chromosome()
        # Verify the union of per-chromosome data matches
        ...

    def test_variant_filtering(self, tmp_path):
        """variant_ids filter should work per-chromosome."""
        ...

    def test_chromosome_selection(self, tmp_path):
        """chromosomes parameter should skip unselected chromosomes."""
        ...

    def test_memory_one_at_a_time(self):
        """Only one chromosome's data should be in memory at once.
        (This is a design test -- verify the generator doesn't eagerly load.)
        """
        ...
```

---

### Phase 1.3: Chromosome-Level Training Loop

**New file**: `imputed_prs/models/chromosome_trainer.py`

```python
"""Chromosome-level training for memory-efficient large-scale imputation."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from imputed_prs.compute.backend import ComputeBackend
from imputed_prs.core.types import ImputedVariantModel, SingleVariantModelResult
from imputed_prs.core.window_index import ChromosomeIndex
from imputed_prs.models.elastic_net import fit_single_variant_model
from imputed_prs.models.trainer import _convert_to_imputed_model


@dataclass
class ChromosomeTrainingResult:
    """Result from training all variants on one chromosome."""
    chromosome: str
    models: Dict[str, ImputedVariantModel]
    n_variants_trained: int
    n_variants_failed: int
    n_intercept_only: int
    mean_r2: float


class ChromosomeTrainer:
    """Train imputation models for all missing variants on a single chromosome.

    Uses pre-computed WindowIndex for fast predictor selection and
    delegates computation to a ComputeBackend (CPU or GPU).
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        backend: Optional[ComputeBackend] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.window_size = window_size
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.backend = backend
        self.progress_callback = progress_callback

    def fit_chromosome(
        self,
        chromosome: str,
        Z_chrom: np.ndarray,
        X_chrom: np.ndarray,
        prs_variants_chrom: pd.DataFrame,
        window_index: ChromosomeIndex,
        calibration_callback=None,
    ) -> ChromosomeTrainingResult:
        """Train all missing variants on one chromosome.

        Args:
            chromosome: Chromosome identifier.
            Z_chrom: Platform variant dosages for this chromosome (n_samples, n_platform).
            X_chrom: Missing variant dosages for this chromosome (n_samples, n_missing).
            prs_variants_chrom: DataFrame of missing variants on this chromosome.
            window_index: Pre-computed ChromosomeIndex for platform variants.
            calibration_callback: Called with (true_dosage, cv_predictions, beta)
                for each variant, enabling streaming calibration (Phase 2).

        Returns:
            ChromosomeTrainingResult with trained models.
        """
        ...
```

**Key difference from current `ImputationModelTrainer`**: Uses `window_index.query()` instead of `filter_to_local_window()`, accepts a `calibration_callback` for streaming calibration (Phase 2), and operates on a single chromosome's data.

#### Test Plan for 1.3

**New file**: `tests/test_chromosome_trainer.py`

```python
class TestChromosomeTrainer:
    def test_matches_full_trainer(self):
        """ChromosomeTrainer on a single chromosome should produce
        identical models to ImputationModelTrainer on the same data."""
        # Create synthetic single-chromosome data
        # Train with both trainers
        # Compare model coefficients, intercepts, R2 values
        ...

    def test_window_index_used(self):
        """Verify that ChromosomeTrainer uses WindowIndex (not filter_to_local_window)."""
        # Mock WindowIndex.query and verify it is called
        ...

    def test_calibration_callback_called(self):
        """calibration_callback should be called for each variant."""
        calls = []
        def callback(true_dosage, cv_preds, beta):
            calls.append((true_dosage.shape, cv_preds.shape, beta))

        trainer.fit_chromosome(..., calibration_callback=callback)
        assert len(calls) == n_missing_variants
        ...

    def test_parallel_results_match_sequential(self):
        """n_jobs=4 should produce same results as n_jobs=1."""
        ...
```

---

### Phase 1.4: Update LinearImputationPRS.fit() for Chromosome Batching

**Modified file**: `imputed_prs/core/linear_imputation_prs.py`

Add parameters to `fit()`:

```python
def fit(
    self,
    reference_genotypes: Union[str, Path],
    prs_definition: ...,
    ...,
    device: str = "auto",              # NEW: CPU/GPU selection
    chromosome_batch: Optional[bool] = None,  # NEW: None = auto-detect
) -> "LinearImputationPRS":
```

When `chromosome_batch` is `True` (or auto-detected when `len(missing_variant_ids) > 10_000`), the training flow becomes:

```
1. Load PRS definition and platform variants (same as before)
2. Partition variants (same as before)
3. Build global platform_variant_info DataFrame (same as before)
4. Build WindowIndex from platform_variant_info  # NEW
5. Get backend from device selection  # NEW
6. For each chromosome:  # NEW LOOP
   a. Load chromosome genotypes via load_genotypes_by_chromosome()
   b. Extract Z_chrom and X_chrom for this chromosome
   c. Train with ChromosomeTrainer
   d. Accumulate calibration statistics (Phase 2)
   e. Release chromosome data
7. Finalize calibration from accumulated statistics
8. Store fitted state (same as before)
```

**Auto-detection threshold**: 10,000 missing variants triggers chromosome batching. Below this, the current monolithic approach works fine.

#### Test Plan for 1.4

```python
class TestChromosomeBatchFit:
    def test_batched_matches_monolithic(self, synthetic_multi_chrom_fixture):
        """chromosome_batch=True should produce equivalent results to False."""
        model_batch = LinearImputationPRS(random_state=42).fit(
            ..., chromosome_batch=True
        )
        model_mono = LinearImputationPRS(random_state=42).fit(
            ..., chromosome_batch=False
        )
        # Compare models, calibration params
        ...

    def test_auto_detection(self):
        """Large PRS should auto-enable chromosome batching."""
        ...

    def test_gpu_device_parameter(self):
        """device='cpu' should use CpuBackend."""
        ...
```

---

## Phase 2: Streaming Calibration

**Goal**: Eliminate the ~40 GB `cv_predictions` storage by computing calibration statistics incrementally as each variant is trained.

**Estimated effort**: Medium (1 session)

**Depends on**: Phase 1 (for chromosome-level callback structure)

### Phase 2.1: Streaming Calibration Accumulator

**New file**: `imputed_prs/evaluation/streaming_calibration.py`

The calibration regression requires two vectors over samples:

$$S^{\text{CV}}_i = \sum_{j \in \mathcal{O}} x_{ij} \beta_j + \sum_{j \in \mathcal{M}} \hat{x}_{ij}^{(-i)} \beta_j$$

$$S^{\text{true}}_i = \sum_{j=1}^{p} x_{ij} \beta_j$$

Both are sums over variants. We can accumulate them incrementally:

```python
"""Streaming calibration accumulator for memory-efficient PRS calibration."""

import numpy as np
from imputed_prs.core.types import CalibrationParams
from imputed_prs.evaluation.calibration import estimate_cv_calibration


class StreamingCalibrationAccumulator:
    """Accumulate PRS sums incrementally across variants/chromosomes.

    Instead of storing all 2M cv_prediction vectors (40 GB), this class
    maintains two running sums of shape (n_samples,) requiring only ~40 KB.

    Usage:
        acc = StreamingCalibrationAccumulator(n_samples=2504)
        # For each chromosome:
        acc.add_observed_variants(dosages, betas)  # observed contribute equally
        for each imputed variant:
            acc.add_imputed_variant(true_dosage, cv_predictions, beta)
        # After all chromosomes:
        params = acc.finalize()
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.s_cv = np.zeros(n_samples, dtype=np.float64)
        self.s_true = np.zeros(n_samples, dtype=np.float64)
        self._n_observed = 0
        self._n_imputed = 0

    def add_observed_variants(
        self,
        dosages: np.ndarray,
        betas: np.ndarray,
    ) -> None:
        """Add observed variant contributions (identical in s_cv and s_true).

        Args:
            dosages: (n_samples, n_variants) dosage matrix.
            betas: (n_variants,) effect sizes.
        """
        contribution = dosages @ betas
        self.s_cv += contribution
        self.s_true += contribution
        self._n_observed += len(betas)

    def add_imputed_variant(
        self,
        true_dosage: np.ndarray,
        cv_predictions: np.ndarray,
        beta: float,
    ) -> None:
        """Add one imputed variant's contribution.

        Args:
            true_dosage: (n_samples,) actual dosages from reference.
            cv_predictions: (n_samples,) out-of-fold predictions. May contain NaN.
            beta: PRS effect size for this variant.
        """
        # For s_true, always use actual dosage
        valid_true = ~np.isnan(true_dosage)
        if np.any(valid_true):
            np.add.at(self.s_true, np.where(valid_true)[0],
                       true_dosage[valid_true] * beta)

        # For s_cv, use cv_predictions (NaN samples get intercept fallback)
        valid_cv = ~np.isnan(cv_predictions)
        if np.any(valid_cv):
            np.add.at(self.s_cv, np.where(valid_cv)[0],
                       cv_predictions[valid_cv] * beta)

        self._n_imputed += 1

    def finalize(self) -> CalibrationParams:
        """Compute calibration parameters from accumulated sums.

        Returns:
            CalibrationParams from regression of s_true on s_cv.
        """
        return estimate_cv_calibration(self.s_cv, self.s_true)
```

**Memory cost**: 2 x 2,504 x 8 bytes = **40 KB**, replacing the ~40 GB `Dict[str, np.ndarray]`.

#### Test Plan for 2.1

**New file**: `tests/test_streaming_calibration.py`

```python
class TestStreamingCalibration:
    def test_matches_batch_calibration(self):
        """Streaming calibration must produce identical CalibrationParams
        as the current batch approach."""
        # Setup: 200 samples, 50 observed, 30 imputed variants
        # Compute calibration the old way (batch)
        # Compute calibration the new way (streaming, one variant at a time)
        # Compare scaling_factor, calibration_intercept, calibration_r2
        np.testing.assert_allclose(
            streaming_params.scaling_factor,
            batch_params.scaling_factor,
            atol=1e-10,
        )
        ...

    def test_incremental_accumulation(self):
        """Adding variants one-by-one vs. all at once should be identical."""
        ...

    def test_nan_handling(self):
        """NaN cv_predictions should not corrupt accumulation."""
        ...

    def test_observed_only(self):
        """With no imputed variants, s_cv == s_true, scaling_factor ~= 1."""
        ...
```

---

### Phase 2.2: Per-Variant CV Prediction Discard

**Modified file**: `imputed_prs/models/elastic_net.py`

Add parameter `return_cv_predictions: bool = True`:

```python
def fit_single_variant_model(
    ...,
    return_cv_predictions: bool = True,  # NEW
) -> SingleVariantModelResult:
```

When `False`, the function still computes out-of-fold predictions internally (needed for R2 calculation), but sets `cv_predictions=None` in the return value. The `ChromosomeTrainer` passes the cv_predictions to the `StreamingCalibrationAccumulator` before they are discarded.

**Modified type**: `SingleVariantModelResult.cv_predictions` becomes `Optional[np.ndarray]` (was `np.ndarray`).

**Modified file**: `imputed_prs/core/types.py`

```python
@dataclass
class SingleVariantModelResult:
    ...
    cv_predictions: Optional[np.ndarray]  # None when streaming calibration is used
    ...
```

Also make `TrainingResult.cv_predictions` optional:

```python
@dataclass
class TrainingResult:
    ...
    cv_predictions: Optional[Dict[str, np.ndarray]]  # None when streaming
    ...
```

#### Test Plan for 2.2

- All existing tests pass (since default is `return_cv_predictions=True`)
- New test: `return_cv_predictions=False` still computes correct R2
- New test: streaming calibration through `ChromosomeTrainer` produces same CalibrationParams as batch

---

## Phase 3: 10-Fold Outer Cross-Validation

**Goal**: Implement K-fold cross-validation on the reference panel itself, where models are trained on K-1 folds and evaluated on the held-out fold.

**Estimated effort**: Large (2-3 sessions)

**Depends on**: Phase 1 (chromosome batching) and Phase 2 (streaming calibration)

### Phase 3.1: Fold Manager

**New file**: `imputed_prs/evaluation/fold_manager.py`

```python
"""Manage K-fold splits of the reference panel for outer cross-validation."""

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
from sklearn.model_selection import KFold


@dataclass
class FoldSpec:
    """Specification for a single fold in outer cross-validation."""
    fold_id: int
    n_folds: int
    train_indices: np.ndarray  # indices into the full sample set
    test_indices: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_test(self) -> int:
        return len(self.test_indices)


class FoldManager:
    """Manage K-fold splits that are consistent across all chromosomes.

    The fold assignment is based on sample indices, not variants.
    All chromosomes use the same train/test split for a given fold.

    Args:
        n_samples: Total number of samples in reference panel.
        n_folds: Number of folds. Default: 10.
        random_state: Random seed for fold assignment. Default: None.
    """

    def __init__(
        self,
        n_samples: int,
        n_folds: int = 10,
        random_state: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.n_folds = n_folds
        self.random_state = random_state
        self._kfold = KFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        # Pre-compute all fold assignments
        self._folds: List[FoldSpec] = []
        dummy = np.zeros(n_samples)
        for fold_id, (train_idx, test_idx) in enumerate(self._kfold.split(dummy)):
            self._folds.append(FoldSpec(
                fold_id=fold_id,
                n_folds=n_folds,
                train_indices=train_idx,
                test_indices=test_idx,
            ))

    def get_fold(self, fold_id: int) -> FoldSpec:
        """Get a specific fold by ID."""
        return self._folds[fold_id]

    def iter_folds(self) -> Iterator[FoldSpec]:
        """Iterate over all folds."""
        return iter(self._folds)
```

#### Test Plan for 3.1

```python
class TestFoldManager:
    def test_folds_non_overlapping(self):
        """Test sets across all folds should be disjoint."""
        fm = FoldManager(n_samples=100, n_folds=5)
        all_test = np.concatenate([f.test_indices for f in fm.iter_folds()])
        assert len(set(all_test)) == 100

    def test_folds_exhaustive(self):
        """Union of all test sets should equal all sample indices."""
        ...

    def test_reproducibility(self):
        """Same random_state should produce same folds."""
        ...

    def test_train_test_exclusive(self):
        """Train and test indices within a fold should not overlap."""
        ...
```

---

### Phase 3.2: Reference Panel Cross-Validation

**New file**: `imputed_prs/evaluation/reference_cv.py`

```python
"""K-fold cross-validation evaluation on the reference panel."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from imputed_prs.compute.backend import ComputeBackend
from imputed_prs.compute.device import get_backend
from imputed_prs.core.types import CalibrationParams
from imputed_prs.evaluation.fold_manager import FoldManager


@dataclass
class FoldResult:
    """Result from one fold of outer cross-validation."""
    fold_id: int
    n_train_samples: int
    n_test_samples: int
    calibration_params: Optional[CalibrationParams]
    test_prs_predicted: np.ndarray   # PRS values for held-out samples
    test_prs_true: np.ndarray        # True PRS for held-out samples
    mean_imputation_r2: float
    training_time_seconds: float


@dataclass
class ReferenceCVResult:
    """Aggregated result from K-fold reference panel cross-validation."""
    n_folds: int
    n_samples: int
    fold_results: List[FoldResult]

    # Aggregated metrics (computed from all held-out predictions)
    overall_pearson_r: float
    overall_r2: float
    overall_mae: float
    overall_rmse: float
    overall_spearman_rho: float
    overall_calibration_slope: float

    # Per-fold metrics for variability assessment
    per_fold_pearson_r: List[float]
    per_fold_r2: List[float]


class ReferencePanelCV:
    """Run K-fold cross-validation on the reference panel.

    For each fold:
      1. Split reference into train/test using FoldManager
      2. Train imputation models on train samples (with inner CV)
      3. Predict PRS on test samples using trained models
      4. Collect test predictions

    After all folds, aggregate metrics across all held-out predictions.
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        n_inner_cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        device: str = "auto",
        verbose: int = 1,
    ):
        self.window_size = window_size
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.n_inner_cv_folds = n_inner_cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.device = device
        self.verbose = verbose

    def run(
        self,
        reference_genotypes: Union[str, Path],
        prs_definition: Union[str, Path, pd.DataFrame],
        platform_name: Optional[str] = None,
        platform_manifest: Optional[Union[str, Path]] = None,
        platform_variants: Optional[List[str]] = None,
        n_outer_folds: int = 10,
        genome_build: Optional[str] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> ReferenceCVResult:
        """Run reference panel cross-validation.

        The outer loop:
          For each of K outer folds:
            1. Split samples into train (~90%) and test (~10%)
            2. For each chromosome:
               a. Load chromosome genotypes
               b. Subset to train samples
               c. Train models (with inner CV for calibration)
               d. Subset to test samples
               e. Predict test PRS using trained models
            3. Accumulate test predictions

        The inner loop (within training):
          5-fold CV used for calibration parameter estimation.

        Args:
            reference_genotypes: Path to reference genotype file.
            prs_definition: PRS definition source.
            platform_name/manifest/variants: Platform specification.
            n_outer_folds: Number of outer CV folds. Default: 10.
            genome_build: Genome build. Auto-detected if None.
            checkpoint_dir: If set, save/resume per-fold checkpoints.

        Returns:
            ReferenceCVResult with per-fold and aggregated metrics.
        """
        ...

    def _train_and_evaluate_fold(
        self,
        fold: "FoldSpec",
        genotype_path: Union[str, Path],
        prs_df: pd.DataFrame,
        platform_variant_set: set,
        platform_variant_info: pd.DataFrame,
        backend: ComputeBackend,
    ) -> FoldResult:
        """Train models on one fold's training data, evaluate on test data.

        Uses chromosome-level batching (Phase 1) and
        streaming calibration (Phase 2) internally.
        """
        ...
```

**Compute analysis**: 10 outer folds x 22 chromosomes x ~90K missing variants per chromosome = ~20M variant-level model fits. With 5 inner CV folds each, this is ~100M ElasticNet fits total.

#### Test Plan for 3.2

**New file**: `tests/test_reference_cv.py`

```python
class TestReferencePanelCV:
    @pytest.fixture
    def synthetic_reference(self, tmp_path):
        """Create a synthetic multi-chromosome reference with known PRS."""
        # 3 chromosomes, 100 platform + 50 missing variants each
        # 200 samples
        # Known ground-truth PRS weights
        ...

    def test_all_samples_predicted_once(self, synthetic_reference):
        """Every sample should appear in exactly one test fold."""
        cv = ReferencePanelCV(random_state=42)
        result = cv.run(..., n_outer_folds=5)
        all_test_prs = np.concatenate([f.test_prs_predicted for f in result.fold_results])
        assert len(all_test_prs) == 200  # All samples predicted

    def test_metrics_computed_correctly(self, synthetic_reference):
        """Verify aggregated metrics match manual calculation."""
        ...

    def test_per_fold_variability(self, synthetic_reference):
        """Per-fold metrics should show reasonable variability."""
        ...

    def test_reproducibility(self, synthetic_reference):
        """Same random_state should produce identical results."""
        ...
```

---

### Phase 3.3: Integration with LinearImputationPRS

**Modified file**: `imputed_prs/core/linear_imputation_prs.py`

Add a new public method:

```python
def evaluate_reference_cv(
    self,
    reference_genotypes: Union[str, Path],
    prs_definition: Union[str, Path, pd.DataFrame],
    platform_name: Optional[str] = None,
    platform_manifest: Optional[Union[str, Path]] = None,
    platform_variants: Optional[List[str]] = None,
    n_outer_folds: int = 10,
    genome_build: Optional[str] = None,
    device: str = "auto",
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> "ReferenceCVResult":
    """Run K-fold cross-validation on the reference panel.

    This evaluates how well the imputation + calibration pipeline
    performs by training on K-1 folds and predicting on the held-out fold,
    repeating for all K folds.

    Note: This does NOT fit the model for prediction. Call fit() separately
    after evaluation to train the final production model.

    Args:
        reference_genotypes: Path to reference genotype file.
        prs_definition: PRS definition (PGS ID, file path, or DataFrame).
        platform_name/manifest/variants: Exactly one must be provided.
        n_outer_folds: Number of CV folds. Default: 10.
        genome_build: Genome build. Default: auto-detect.
        device: Compute device. Default: "auto".
        checkpoint_dir: Directory for fold checkpoints. Default: None.

    Returns:
        ReferenceCVResult with metrics and per-fold details.
    """
    cv_runner = ReferencePanelCV(
        window_size=self.window_size,
        l1_ratio=self.l1_ratio,
        alpha=self.alpha,
        n_inner_cv_folds=self.cv_folds,
        n_jobs=self.n_jobs,
        random_state=self.random_state,
        max_predictors=self.max_predictors,
        device=device,
        verbose=self.verbose,
    )
    return cv_runner.run(
        reference_genotypes=reference_genotypes,
        prs_definition=prs_definition,
        platform_name=platform_name,
        platform_manifest=platform_manifest,
        platform_variants=platform_variants,
        n_outer_folds=n_outer_folds,
        genome_build=genome_build,
        checkpoint_dir=checkpoint_dir,
    )
```

---

## Phase 4: GPU-Accelerated Training

**Goal**: Leverage the backend abstraction from Phase 0 to run the full training pipeline on GPU with batched variant fitting.

**Estimated effort**: Large (2-3 sessions)

**Depends on**: Phase 0 (GPU backend), Phase 1 (chromosome batching)

### Phase 4.1: GPU Memory Manager

**New file**: `imputed_prs/compute/gpu_memory.py`

```python
"""GPU memory management for chromosome-level data transfer."""

from typing import Optional, Tuple
import numpy as np


class GpuMemoryManager:
    """Manage GPU memory for chromosome-level processing.

    Handles data transfer between CPU and GPU, dtype conversion
    (float32 for MPS, float64 for CUDA), and memory lifecycle.
    """

    def __init__(self, device: str):
        import torch
        self._device = torch.device(device)
        self._dtype = torch.float32 if device == "mps" else torch.float64
        self._Z_gpu = None
        self._X_gpu = None

    def load_chromosome(
        self,
        Z_chrom: np.ndarray,
        X_chrom: np.ndarray,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Transfer chromosome matrices to GPU.

        Args:
            Z_chrom: Platform dosages (n_samples, n_platform_on_chrom).
            X_chrom: Missing dosages (n_samples, n_missing_on_chrom).

        Returns:
            (Z_gpu, X_gpu) as GPU tensors.
        """
        import torch
        self.release()  # Free previous chromosome
        self._Z_gpu = torch.tensor(
            Z_chrom, device=self._device, dtype=self._dtype
        )
        self._X_gpu = torch.tensor(
            X_chrom, device=self._device, dtype=self._dtype
        )
        return self._Z_gpu, self._X_gpu

    def release(self):
        """Free GPU memory from current chromosome."""
        import torch
        self._Z_gpu = None
        self._X_gpu = None
        if str(self._device) == "cuda":
            torch.cuda.empty_cache()

    def estimated_memory_mb(
        self, Z_shape: Tuple[int, int], X_shape: Tuple[int, int]
    ) -> float:
        """Estimate GPU memory needed for given matrix shapes."""
        bytes_per_element = 4 if self._dtype.itemsize == 4 else 8
        total_elements = Z_shape[0] * Z_shape[1] + X_shape[0] * X_shape[1]
        return total_elements * bytes_per_element / (1024 * 1024)
```

---

### Phase 4.2: Batched Variant Fitting on GPU

**New method in** `imputed_prs/compute/gpu_backend.py`:

Many adjacent missing variants share the same (or very similar) predictor window. Batching them exploits GPU parallelism:

```python
def fit_elastic_net_batch(
    self,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    l1_ratio: float,
    alpha: float,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ElasticNet for multiple targets simultaneously.

    All targets share the same predictor matrix X. This amortizes
    the GPU data transfer and enables parallel FISTA across targets.

    Args:
        X_train: (n_train, n_features) shared predictors.
        Y_train: (n_train, n_targets) multiple targets.
        X_val: (n_val, n_features) validation predictors.
        l1_ratio: L1/L2 mixing.
        alpha: Regularization strength.
        max_iter: Maximum iterations.

    Returns:
        (W, intercepts, val_predictions)
        W: (n_features, n_targets) coefficient matrix.
        intercepts: (n_targets,) intercept vector.
        val_predictions: (n_val, n_targets) predictions.
    """
```

The FISTA update for batched targets becomes:

```python
# W is (n_features, n_targets) -- all targets updated simultaneously
residual = X @ V - Y  # (n_samples, n_targets) -- one matmul for all targets
grad = X.T @ residual / n + alpha * (1 - l1_ratio) * V
W_prev = W.clone()
W = soft_threshold(V - step * grad, l1_threshold)  # element-wise on full matrix
```

**Grouping strategy**: Group missing variants by chromosomal region such that all variants in a group share the same predictor window (within tolerance). A simple approach: sort variants by position, form groups of up to `batch_size` consecutive variants that share at least 80% of their predictors.

#### Test Plan for 4.2

```python
class TestBatchedGpuFitting:
    def test_batch_matches_individual(self):
        """Batched fit of 10 variants should match 10 individual fits."""
        X = rng.standard_normal((100, 20))
        Y = X @ rng.standard_normal((20, 10)) + rng.standard_normal((100, 10)) * 0.1

        # Batch fit
        W_batch, int_batch, pred_batch = gpu.fit_elastic_net_batch(
            X[:80], Y[:80], X[80:], l1_ratio=0.5, alpha=0.01
        )

        # Individual fits
        for i in range(10):
            coef_i, int_i, pred_i = gpu.fit_elastic_net(
                X[:80], Y[:80, i], X[80:], l1_ratio=0.5, alpha=0.01
            )
            np.testing.assert_allclose(W_batch[:, i], coef_i, atol=1e-2)
```

---

### Phase 4.3: GPU Cross-Validation Loop

**Modified file**: `imputed_prs/models/elastic_net.py`

For GPU, optimize the CV loop to keep data on GPU across folds:

```python
def fit_single_variant_model_gpu(
    target_gpu: "torch.Tensor",
    predictors_gpu: "torch.Tensor",
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    l1_ratio: float,
    alpha: float,
    backend: "GpuBackend",
) -> SingleVariantModelResult:
    """GPU-optimized variant fitting with pre-transferred data.

    Data stays on GPU. Fold splits are done by index masking,
    avoiding CPU-GPU transfer per fold.
    """
```

---

### Phase 4.4: MPS-Specific Workarounds

**Modified file**: `imputed_prs/compute/gpu_backend.py`

Document and handle known MPS limitations:

| Limitation | Workaround |
|-----------|------------|
| No float64 | Use float32; results within ~1e-3 of sklearn |
| No `torch.linalg.eigvalsh` | Use `torch.trace(XtX)` for Lipschitz estimate |
| Unified memory | No separate VRAM budget; share with system RAM |
| Possible fallback | If chromosome too large for available memory, fall back to CPU for that chromosome |

#### Test Plan for Phase 4

All GPU tests use `@pytest.mark.gpu` marker and are skipped when no GPU is available.

```python
# In conftest.py or pytest configuration:
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")

# In test files:
@pytest.mark.gpu
class TestGpuTraining:
    def test_chromosome_training_gpu(self):
        """Full chromosome training on GPU matches CPU results."""
        ...

    def test_mps_float32_precision(self):
        """MPS float32 results should be within 1e-2 of CPU float64."""
        ...
```

---

## Phase 5: Checkpointing and Progress

**Goal**: Support saving/resuming progress for multi-hour runs.

**Estimated effort**: Medium (1 session)

**Depends on**: Phase 1 (chromosome-level processing)

### Phase 5.1: Checkpoint Format

**New file**: `imputed_prs/io/checkpoint.py`

```python
"""Checkpoint save/resume for long-running training jobs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import datetime

import h5py
import numpy as np

from imputed_prs.core.types import ImputedVariantModel


@dataclass
class TrainingCheckpoint:
    """State of a partially completed training run."""
    completed_chromosomes: List[str]
    current_chromosome: Optional[str]
    completed_variant_count: int
    total_variant_count: int
    models: Dict[str, ImputedVariantModel]
    calibration_state: Dict[str, np.ndarray]  # s_cv, s_true accumulators
    hyperparameters: Dict[str, float]  # l1_ratio, alpha
    fold_id: Optional[int]  # For outer CV
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


def save_checkpoint(checkpoint: TrainingCheckpoint, path: Path) -> None:
    """Save training checkpoint to HDF5.

    Saves after each chromosome completes. On resume, completed
    chromosomes are skipped.
    """
    with h5py.File(path, "w") as f:
        # Metadata
        f.attrs["completed_chromosomes"] = checkpoint.completed_chromosomes
        f.attrs["current_chromosome"] = checkpoint.current_chromosome or ""
        f.attrs["completed_variant_count"] = checkpoint.completed_variant_count
        f.attrs["total_variant_count"] = checkpoint.total_variant_count
        f.attrs["timestamp"] = checkpoint.timestamp
        f.attrs["fold_id"] = checkpoint.fold_id if checkpoint.fold_id is not None else -1

        # Hyperparameters
        hp = f.create_group("hyperparameters")
        for k, v in checkpoint.hyperparameters.items():
            hp.attrs[k] = v

        # Calibration state
        cal = f.create_group("calibration_state")
        for k, v in checkpoint.calibration_state.items():
            cal.create_dataset(k, data=v)

        # Models (one group per variant)
        models_grp = f.create_group("models")
        for var_id, model in checkpoint.models.items():
            mg = models_grp.create_group(var_id)
            mg.attrs["variant_id"] = model.variant_id
            mg.attrs["chromosome"] = model.chromosome
            mg.attrs["position"] = model.position
            mg.attrs["effect_allele"] = model.effect_allele
            mg.attrs["beta"] = model.beta
            mg.attrs["intercept"] = model.intercept
            mg.attrs["imputation_r2"] = model.imputation_r2
            mg.attrs["residual_variance"] = model.residual_variance
            mg.attrs["allele_frequency"] = model.allele_frequency
            mg.attrs["is_intercept_only"] = model.is_intercept_only
            mg.create_dataset("coefficients", data=model.coefficients)
            mg.attrs["predictor_variant_ids"] = model.predictor_variant_ids


def load_checkpoint(path: Path) -> TrainingCheckpoint:
    """Load training checkpoint from HDF5."""
    ...
```

#### Test Plan for 5.1

```python
class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should produce identical checkpoint."""
        ...

    def test_resume_skips_completed(self, tmp_path):
        """After loading checkpoint, completed chromosomes should be skipped."""
        ...

    def test_partial_training_resume(self, tmp_path):
        """Train 2 chromosomes, checkpoint, resume, verify final == full run."""
        ...
```

---

### Phase 5.2: Progress Reporting

**Modified files**: `imputed_prs/models/chromosome_trainer.py`, `imputed_prs/core/linear_imputation_prs.py`

Extend the existing progress_callback interface:

```python
@dataclass
class TrainingProgress:
    """Structured progress update."""
    chromosome: str
    variants_completed: int
    variants_total: int
    chromosomes_completed: int
    chromosomes_total: int
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float]
    mean_r2_so_far: float
    gpu_memory_mb: Optional[float]
```

The `ChromosomeTrainer` and `LinearImputationPRS.fit()` emit `TrainingProgress` objects to a callback. Default callback prints formatted progress to stdout.

---

## Phase 6: Optimized Prediction

**Goal**: Make the prediction path efficient for 2M-variant models.

**Estimated effort**: Medium (1 session)

**Depends on**: None (independent of other phases)

### Phase 6.1: Vectorized Predictor

**New file**: `imputed_prs/models/vectorized_predictor.py`

The current `compute_imputed_prs()` (predictor.py:83-123) is a Python loop over 2M models with dict lookups. Replace with vectorized computation using sparse matrices:

```python
"""Vectorized PRS prediction using sparse coefficient matrices."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from imputed_prs.core.types import (
    CalibrationParams,
    ImputedVariantModel,
    PredictionResult,
    VariantInfo,
)
from imputed_prs.models.bounding import clip_and_adjust_variance


class VectorizedPredictor:
    """Batch-vectorized PRS prediction for large models.

    Pre-builds sparse coefficient matrices from imputed models,
    enabling efficient matrix-vector prediction instead of Python loops.

    For 2M variants with ~100 non-zero coefficients each:
    - Sparse matrix: ~200M entries x 12 bytes (CSR) = ~2.4 GB
    - Prediction: Single sparse matrix-vector multiply

    Compared to current approach (Python loop + dict lookups):
    - Current: ~30 seconds for 2M variants per user
    - Vectorized: ~0.5 seconds for 2M variants per user
    """

    def __init__(
        self,
        observed_variants: List[VariantInfo],
        imputed_models: List[ImputedVariantModel],
        calibration_params: Optional[CalibrationParams] = None,
    ):
        self.observed_variants = observed_variants
        self.imputed_models = imputed_models
        self.calibration_params = calibration_params
        self._build_prediction_structures()

    def _build_prediction_structures(self):
        """Pre-compute sparse coefficient matrix and index mappings.

        Builds:
        - self._predictor_variant_ids: sorted list of all unique predictor variant IDs
        - self._predictor_id_to_idx: dict mapping variant_id to column index
        - self._coef_matrix: sparse CSR matrix (n_imputed x n_unique_predictors)
        - self._intercepts: (n_imputed,) array
        - self._betas: (n_imputed,) array of effect sizes
        - self._residual_variances: (n_imputed,) array
        """
        # Collect all unique predictor variant IDs
        predictor_set = set()
        for model in self.imputed_models:
            predictor_set.update(model.predictor_variant_ids)

        self._predictor_variant_ids = sorted(predictor_set)
        self._predictor_id_to_idx = {
            vid: i for i, vid in enumerate(self._predictor_variant_ids)
        }

        n_imputed = len(self.imputed_models)
        n_predictors = len(self._predictor_variant_ids)

        # Build sparse coefficient matrix in COO format, then convert to CSR
        rows, cols, data = [], [], []
        self._intercepts = np.zeros(n_imputed)
        self._betas = np.zeros(n_imputed)
        self._residual_variances = np.zeros(n_imputed)
        self._is_intercept_only = np.zeros(n_imputed, dtype=bool)

        for i, model in enumerate(self.imputed_models):
            self._intercepts[i] = model.intercept
            self._betas[i] = model.beta
            self._residual_variances[i] = model.residual_variance
            self._is_intercept_only[i] = model.is_intercept_only

            for j, pred_id in enumerate(model.predictor_variant_ids):
                if j < len(model.coefficients):
                    col_idx = self._predictor_id_to_idx[pred_id]
                    rows.append(i)
                    cols.append(col_idx)
                    data.append(model.coefficients[j])

        if rows:
            self._coef_matrix = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(n_imputed, n_predictors),
            )
        else:
            self._coef_matrix = sparse.csr_matrix((n_imputed, n_predictors))

    def predict(
        self,
        user_genotypes: Dict[str, Optional[float]],
        apply_calibration: bool = True,
    ) -> PredictionResult:
        """Compute PRS using vectorized sparse operations.

        Steps:
        1. Build predictor dosage vector from user_genotypes
        2. Sparse matrix-vector multiply: raw_preds = coef_matrix @ dosages + intercepts
        3. Apply clipping and variance adjustment (vectorized)
        4. Sum contributions weighted by betas
        """
        # Build predictor dosage vector
        dosage_vec = np.zeros(len(self._predictor_variant_ids))
        available_mask = np.zeros(len(self._predictor_variant_ids), dtype=bool)

        for i, vid in enumerate(self._predictor_variant_ids):
            d = user_genotypes.get(vid)
            if d is not None:
                dosage_vec[i] = d
                available_mask[i] = True

        # Sparse multiply: (n_imputed, n_predictors) @ (n_predictors,) -> (n_imputed,)
        raw_predictions = self._coef_matrix @ dosage_vec + self._intercepts

        # For intercept-only models, prediction is just the intercept (already correct)
        # For models with missing predictors, fall back to intercept
        # (this is an approximation -- same as current behavior)

        # Vectorized clipping and variance adjustment
        clipped = np.clip(raw_predictions, 0.0, 2.0)
        n_truncated = int(np.sum(clipped != raw_predictions))

        # Vectorized variance adjustment (simplified -- Phase 6 can refine)
        adjusted_variances = self._residual_variances.copy()

        # PRS components
        prs_imputed = float(np.sum(clipped * self._betas))
        total_variance = float(np.sum(self._betas**2 * adjusted_variances))

        # Observed component
        prs_observed = 0.0
        n_observed_used = 0
        for var in self.observed_variants:
            d = user_genotypes.get(var.variant_id)
            if d is not None:
                prs_observed += d * var.beta
                n_observed_used += 1

        # Combine
        prs_raw = prs_observed + prs_imputed
        se = np.sqrt(total_variance) if total_variance > 0 else 0.0

        # ... build PredictionResult (same as current PRSPredictor) ...
```

#### Test Plan for 6.1

```python
class TestVectorizedPredictor:
    def test_matches_prs_predictor(self):
        """VectorizedPredictor must produce identical results to PRSPredictor."""
        # Create models with known coefficients
        # Predict with both predictors
        # Compare all fields of PredictionResult
        np.testing.assert_allclose(
            vec_result.prs, orig_result.prs, atol=1e-10
        )
        ...

    def test_sparse_matrix_structure(self):
        """Coefficient matrix should be sparse (most entries zero)."""
        ...

    def test_2m_variant_prediction_time(self):
        """2M variant prediction should complete in < 5 seconds."""
        # Create 2M synthetic models with ~100 predictors each
        # Time a single prediction
        ...
```

---

## Phase 7: Integration and Documentation

**Goal**: Verify all phases work together end-to-end and document the scaling workflow.

**Estimated effort**: Medium (1-2 sessions)

**Depends on**: All previous phases

### Phase 7.1: End-to-End Integration Tests

**New file**: `tests/test_large_scale_integration.py`

```python
"""Integration tests for the scaled pipeline."""

import pytest
import numpy as np

class TestFullPipeline:
    @pytest.fixture
    def mini_1000g(self, tmp_path):
        """Create a synthetic 'mini 1000 Genomes' test fixture.

        - 3 chromosomes
        - 500 platform variants per chromosome
        - 200 missing PRS variants per chromosome (600 total)
        - 100 samples
        - Known ground-truth PRS with planted signal
        """
        ...

    def test_fit_with_chromosome_batching(self, mini_1000g):
        """Full fit() with chromosome batching succeeds."""
        model = LinearImputationPRS(
            window_size=500_000, cv_folds=3, random_state=42
        )
        model.fit(
            reference_genotypes=mini_1000g["vcf_path"],
            prs_definition=mini_1000g["prs_df"],
            platform_variants=mini_1000g["platform_variants"],
            chromosome_batch=True,
            device="cpu",
        )
        assert model.is_fitted
        assert len(model.imputed_models) > 0

    def test_reference_cv(self, mini_1000g):
        """10-fold CV on mini reference panel."""
        model = LinearImputationPRS(
            window_size=500_000, cv_folds=3, random_state=42
        )
        result = model.evaluate_reference_cv(
            reference_genotypes=mini_1000g["vcf_path"],
            prs_definition=mini_1000g["prs_df"],
            platform_variants=mini_1000g["platform_variants"],
            n_outer_folds=5,
            device="cpu",
        )
        assert result.n_folds == 5
        assert len(result.fold_results) == 5
        assert result.overall_pearson_r > 0  # Signal should be detectable

    def test_checkpoint_resume(self, mini_1000g, tmp_path):
        """Training with checkpointing can be interrupted and resumed."""
        ...

    def test_export_and_reload(self, mini_1000g, tmp_path):
        """Trained model can be exported and reloaded for prediction."""
        ...

    @pytest.mark.gpu
    def test_gpu_matches_cpu(self, mini_1000g):
        """GPU training produces equivalent results to CPU."""
        ...
```

---

### Phase 7.2: Configuration Interface

**Modified file**: `imputed_prs/core/linear_imputation_prs.py`

Final `fit()` signature with all new parameters:

```python
def fit(
    self,
    reference_genotypes: Union[str, Path],
    prs_definition: Union[str, Path, pd.DataFrame],
    platform_name: Optional[str] = None,
    platform_manifest: Optional[Union[str, Path]] = None,
    platform_variants: Optional[List[str]] = None,
    genome_build: Optional[str] = None,
    prs_id: Optional[str] = None,
    model_name: Optional[str] = None,
    evaluation_genotypes: Optional[Union[str, Path]] = None,
    # NEW parameters for scaling:
    device: str = "auto",
    chromosome_batch: Optional[bool] = None,  # None = auto-detect
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> "LinearImputationPRS":
```

---

### Phase 7.3: Documentation

**New file**: `docs/scaling-guide.md`

Content:
- How to run on GPU (install torch, set device="mps" or "cuda")
- Expected memory usage by dataset size
- Expected runtime by dataset size and device
- Checkpoint/resume workflow
- Reference panel CV workflow and interpreting results
- Troubleshooting MPS issues

---

## Phase Dependency Graph

```
Phase 0: Backend Abstraction
├── 0.1: Protocol + device selection
├── 0.2: CPU backend (sklearn wrapper)
├── 0.3: GPU backend (PyTorch FISTA)
└── 0.4: Refactor elastic_net.py to use backend
         |
         ├──────────────────────┐
         v                      v
Phase 1: Chromosome Batching   Phase 4: GPU Training
├── 1.1: WindowIndex           ├── 4.1: GPU memory manager
├── 1.2: Chromosome loader     ├── 4.2: Batched variant fitting
├── 1.3: ChromosomeTrainer     ├── 4.3: GPU CV loop
└── 1.4: Update fit() API      └── 4.4: MPS workarounds
         |
         v
Phase 2: Streaming Calibration   Phase 6: Optimized Prediction
├── 2.1: Accumulator             ├── 6.1: VectorizedPredictor
└── 2.2: CV prediction discard   └── (independent of other phases)
         |
         v
Phase 3: 10-Fold Outer CV       Phase 5: Checkpointing
├── 3.1: FoldManager             ├── 5.1: Checkpoint format
├── 3.2: ReferencePanelCV        └── 5.2: Progress reporting
└── 3.3: Integration
         |
         v
Phase 7: Integration & Docs
├── 7.1: End-to-end tests
├── 7.2: Configuration
└── 7.3: Documentation
```

**Parallelizable by separate agents**:
- Phase 0.1-0.2 and Phase 1.1-1.2 can proceed in parallel
- Phase 0.3 (GPU backend) can proceed in parallel with Phase 1.3-1.4
- Phase 6 (prediction optimization) is fully independent

---

## Memory Budget Summary

| Component | Current (2M variants) | After Optimization |
|-----------|----------------------|-------------------|
| Full dosage matrix (all chromosomes) | ~25 GB | Not loaded (streaming) |
| One chromosome dosage (largest: chr1) | N/A | ~2 GB |
| Z + X matrices (all) | ~20 GB | ~2 GB (per chromosome) |
| CV predictions (all variants) | ~40 GB | ~40 KB (streaming accumulator) |
| Trained models (coefficients) | ~2 GB | ~2 GB (streamed to checkpoint) |
| GPU memory (one chromosome) | N/A | ~2-4 GB |
| **Peak RAM** | **~65 GB** | **~4-6 GB** |

---

## Runtime Estimates

Training 2M missing variants, 2,504 reference samples, ~100 predictors per window.

### Single training run (no outer CV)

| Device | Per-variant fit (ms) | Total training | Notes |
|--------|---------------------|----------------|-------|
| CPU (1 core) | ~1.0 | ~33 min | Sequential ElasticNet fits |
| CPU (8 cores) | ~0.15 effective | ~5 min | joblib parallelism |
| GPU (NVIDIA, batched) | ~0.02 | ~40 sec | 256-variant FISTA batches |
| GPU (M5 MPS, batched) | ~0.05 | ~100 sec | Float32, unified memory |

### 10-fold outer cross-validation

| Device | Total time | Notes |
|--------|-----------|-------|
| CPU (1 core) | ~5.5 hours | 10x training |
| CPU (8 cores) | ~50 min | Near-linear scaling |
| GPU (NVIDIA, batched) | ~7 min | Amortized data transfer |
| GPU (M5 MPS, batched) | ~17 min | Float32 computation |

---

## File Change Summary

### New Files

| Phase | File | Purpose |
|-------|------|---------|
| 0 | `imputed_prs/compute/__init__.py` | Package init |
| 0 | `imputed_prs/compute/backend.py` | ComputeBackend protocol |
| 0 | `imputed_prs/compute/device.py` | Device selection and factory |
| 0 | `imputed_prs/compute/cpu_backend.py` | CPU backend (sklearn wrapper) |
| 0 | `imputed_prs/compute/gpu_backend.py` | GPU backend (PyTorch FISTA) |
| 1 | `imputed_prs/core/window_index.py` | Pre-computed window index |
| 1 | `imputed_prs/io/chromosome_loader.py` | Per-chromosome genotype loading |
| 1 | `imputed_prs/models/chromosome_trainer.py` | Chromosome-level training |
| 2 | `imputed_prs/evaluation/streaming_calibration.py` | Streaming calibration accumulator |
| 3 | `imputed_prs/evaluation/fold_manager.py` | K-fold split management |
| 3 | `imputed_prs/evaluation/reference_cv.py` | Reference panel CV runner |
| 4 | `imputed_prs/compute/gpu_memory.py` | GPU memory management |
| 5 | `imputed_prs/io/checkpoint.py` | Checkpoint save/resume |
| 6 | `imputed_prs/models/vectorized_predictor.py` | Sparse vectorized prediction |

### Modified Files

| Phase | File | Change |
|-------|------|--------|
| 0 | `imputed_prs/models/elastic_net.py` | Add `backend` parameter |
| 0 | `pyproject.toml` | Add `gpu` optional dependency |
| 1 | `imputed_prs/core/linear_imputation_prs.py` | Add `device`, `chromosome_batch` params |
| 2 | `imputed_prs/core/types.py` | Make `cv_predictions` optional |
| 3 | `imputed_prs/core/linear_imputation_prs.py` | Add `evaluate_reference_cv()` method |
| 5 | `imputed_prs/core/linear_imputation_prs.py` | Add `checkpoint_dir` parameter |

### New Test Files

| Phase | File |
|-------|------|
| 0 | `tests/test_device.py`, `tests/test_cpu_backend.py`, `tests/test_gpu_backend.py` |
| 1 | `tests/test_window_index.py`, `tests/test_chromosome_loader.py`, `tests/test_chromosome_trainer.py` |
| 2 | `tests/test_streaming_calibration.py` |
| 3 | `tests/test_fold_manager.py`, `tests/test_reference_cv.py` |
| 5 | `tests/test_checkpoint.py` |
| 6 | `tests/test_vectorized_predictor.py` |
| 7 | `tests/test_large_scale_integration.py` |

---

## How to Use This Plan

Each phase is designed to be picked up by a Claude Code agent with the following workflow:

1. **Read this document** and the relevant source files listed in the phase
2. **Implement** the new files and modifications described
3. **Run existing tests** to verify backward compatibility: `.venv/bin/pytest tests/ -v`
4. **Run new tests** for the phase: `.venv/bin/pytest tests/test_<new_file>.py -v`
5. **Commit** with a descriptive message referencing the phase

Agents should start with Phase 0 and proceed sequentially, except where the dependency graph indicates phases can be parallelized.
