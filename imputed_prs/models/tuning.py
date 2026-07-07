"""Hyperparameter tuning for elastic net imputation and projection models.

All tuning evaluates candidate ``(l1_ratio, alpha)`` pairs on the **same**
local-window / region matrices that training uses, over a bounded, stratified
sample of variants (imputation) or regions (projection). A single grid-search
engine (:func:`_grid_search_over_datasets`) runs over pre-built
``(predictor_matrix, target)`` datasets and is parameterized by an injectable
CV fitter (``fit_single_variant_model`` for imputation, ``fit_single_region_model``
for projection), so imputation and projection share one code path.
"""

import time
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.harmonizer import _normalize_chromosome
from imputed_prs.core.types import GridSearchResult, OptunaSearchResult
from imputed_prs.core.window_index import ChromosomeIndex
from imputed_prs.models.elastic_net import fit_single_variant_model

# Default hyperparameter grids
DEFAULT_L1_RATIOS = [0.1, 0.5, 0.9]
DEFAULT_ALPHAS = [0.001, 0.01, 0.1]

# Number of quantile bins for continuous stratification features (MAF, |beta|).
_N_STRATA_BINS = 4


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin ``values`` into at most ``n_bins`` buckets by interior data quantiles.

    Data-driven edges keep buckets populated regardless of the feature's scale.
    Deterministic for a given input. Returns integer bin labels in ``[0, n_bins)``.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0, dtype=int)
    interior_q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if len(interior_q) == 0:
        return np.zeros(len(values), dtype=int)
    edges = np.quantile(values, interior_q)
    return np.digitize(values, edges)


def select_stratified_sample(
    stratum_keys: Sequence[Hashable],
    n_target: Optional[int],
    random_state: Optional[int] = None,
) -> List[int]:
    """Deterministically select up to ``n_target`` item indices, spread across strata.

    Items are grouped by their stratum key. A largest-remainder allocation gives
    each stratum a quota proportional to its size; within each stratum a seeded
    draw (or a deterministic head-slice when ``random_state`` is None) picks the
    members. Returns all indices (sorted ascending) when ``n_target`` is None or
    ``>=`` the number of items; an empty list when ``n_target <= 0``.

    The result is bit-stable for a given ``(stratum_keys, n_target, random_state)``:
    strata are visited in sorted-key order, members in ascending index order, and a
    single seeded ``np.random.RandomState`` is consumed in that fixed order.
    """
    n = len(stratum_keys)
    if n == 0:
        return []
    if n_target is None or n_target >= n:
        return list(range(n))
    if n_target <= 0:
        return []

    groups: Dict[Hashable, List[int]] = {}
    for idx, key in enumerate(stratum_keys):
        groups.setdefault(key, []).append(idx)
    sorted_keys = sorted(groups.keys(), key=repr)

    # Largest-remainder allocation across strata by size.
    quotas: Dict[Hashable, int] = {}
    remainders: List[Tuple[float, Hashable]] = []
    allocated = 0
    for key in sorted_keys:
        exact = n_target * len(groups[key]) / n
        q = int(np.floor(exact))
        quotas[key] = q
        allocated += q
        remainders.append((exact - q, key))
    remaining = n_target - allocated
    remainders.sort(key=lambda t: (-t[0], repr(t[1])))
    for i in range(remaining):
        quotas[remainders[i][1]] += 1

    rng = np.random.RandomState(random_state) if random_state is not None else None
    chosen: List[int] = []
    for key in sorted_keys:
        members = sorted(groups[key])
        q = min(quotas[key], len(members))
        if q <= 0:
            continue
        if rng is not None:
            positions = rng.permutation(len(members))[:q]
            chosen.extend(members[p] for p in sorted(positions))
        else:
            chosen.extend(members[:q])

    # Guard against rounding shortfalls: top up from unpicked indices in order.
    if len(chosen) < n_target:
        chosen_set = set(chosen)
        for idx in range(n):
            if idx not in chosen_set:
                chosen.append(idx)
                if len(chosen) >= n_target:
                    break
    return sorted(chosen)


def _imputation_stratum_keys(
    X_missing: np.ndarray,
    missing_variant_info: "Any",
) -> List[Tuple[str, int, int]]:
    """Per-variant stratum keys: (chromosome, MAF bin, |beta| bin).

    MAF is derived from the missing-variant dosage column (mean / 2); |beta| from
    the PRS weight. Both are bucketed by data quantiles. All computed in O(n) with
    no local-window work, so stratification stays cheap.
    """
    n = X_missing.shape[1]
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(X_missing, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    af = col_means / 2.0
    maf = np.minimum(af, 1.0 - af)
    abs_beta = np.abs(missing_variant_info["beta"].to_numpy(dtype=float))
    maf_bins = _quantile_bin(maf, _N_STRATA_BINS)
    beta_bins = _quantile_bin(abs_beta, _N_STRATA_BINS)
    chroms = [
        _normalize_chromosome(str(c))
        for c in missing_variant_info["chromosome"].tolist()
    ]
    return [(chroms[i], int(maf_bins[i]), int(beta_bins[i])) for i in range(n)]


def _bucket_predictor_count(n_predictors: int) -> str:
    """Bucket a region's predictor count for stratification."""
    if n_predictors == 0:
        return "0"
    if n_predictors == 1:
        return "1"
    if n_predictors <= 5:
        return "2-5"
    if n_predictors <= 20:
        return "6-20"
    return "21+"


def _bucket_prs_count(n_prs: int) -> str:
    """Bucket a region's PRS-variant count for stratification."""
    if n_prs <= 1:
        return "1"
    if n_prs <= 3:
        return "2-3"
    return "4+"


# ---------------------------------------------------------------------------
# Grid-search engine over pre-built datasets
# ---------------------------------------------------------------------------


def _resolve_grids(
    l1_ratios: Optional[List[float]],
    alphas: Optional[List[float]],
) -> Tuple[List[float], List[float]]:
    """Apply defaults and validate the hyperparameter grids."""
    if l1_ratios is None:
        l1_ratios = DEFAULT_L1_RATIOS.copy()
    if alphas is None:
        alphas = DEFAULT_ALPHAS.copy()
    if len(l1_ratios) == 0:
        raise ValidationError("l1_ratios cannot be empty")
    if len(alphas) == 0:
        raise ValidationError("alphas cannot be empty")
    for l1 in l1_ratios:
        if not (0.0 <= l1 <= 1.0):
            raise ValidationError(f"l1_ratio must be between 0 and 1, got {l1}")
    for a in alphas:
        if a < 0:
            raise ValidationError(f"alpha must be non-negative, got {a}")
    return l1_ratios, alphas


def _check_sample_shapes(Z: np.ndarray, X: np.ndarray, x_name: str = "X_missing") -> None:
    """Validate that predictor and target matrices share the sample axis."""
    if Z.shape[0] != X.shape[0]:
        raise ValidationError(
            f"Shape mismatch: Z has {Z.shape[0]} samples but "
            f"{x_name} has {X.shape[0]} samples"
        )


def _dataset_has_predictors(dataset: Tuple[np.ndarray, np.ndarray]) -> bool:
    predictor = dataset[0]
    return predictor.size > 0 and predictor.ndim == 2 and predictor.shape[1] > 0


def _tally_failure_reasons(error_types: List[str]) -> Dict[str, int]:
    """Count occurrences of each exception class name (P5.1 diagnostics)."""
    reasons: Dict[str, int] = {}
    for name in error_types:
        reasons[name] = reasons.get(name, 0) + 1
    return reasons


def _evaluate_one_dataset(
    predictor_dosages: np.ndarray,
    target: np.ndarray,
    l1_ratio: float,
    alpha: float,
    cv_folds: int,
    random_state: Optional[int],
    fit_fn: Callable[..., Any] = fit_single_variant_model,
    failure_sink: Optional[List[str]] = None,
) -> Optional[float]:
    """Fit one ``(predictor, target)`` dataset and return its CV MSE, or None.

    Returns None when the model is intercept-only (the hyperparameters had no
    effect) or the fit raises. ``fit_fn`` is the CV fitter; both
    ``fit_single_variant_model`` and ``fit_single_region_model`` accept
    ``(target, predictor_dosages, l1_ratio=, alpha=, cv_folds=, random_state=)``.

    When ``failure_sink`` is provided, the exception class name is appended to it
    on a genuine fit failure (P5.1) — this distinguishes a raised exception from
    the intercept-only ``None``, which does not touch the sink.
    """
    try:
        result = fit_fn(
            target,
            predictor_dosages,
            l1_ratio=l1_ratio,
            alpha=alpha,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        if result.is_intercept_only:
            return None
        return result.cv_mse
    except Exception as exc:
        if failure_sink is not None:
            failure_sink.append(type(exc).__name__)
        return None


def _grid_search_over_datasets(
    datasets: List[Tuple[np.ndarray, np.ndarray]],
    l1_ratios: List[float],
    alphas: List[float],
    cv_folds: int,
    random_state: Optional[int],
    n_variants_sampled: int,
    fit_fn: Callable[..., Any] = fit_single_variant_model,
) -> GridSearchResult:
    """Run the l1_ratio x alpha grid over pre-built ``(predictor, target)`` datasets.

    Edge-case semantics (matching the historical search):
    - no dataset has any predictor -> inf-result with the first grid point as best;
    - datasets have predictors but every fit fails for every grid point ->
      ``ValidationError``.
    """
    if not any(_dataset_has_predictors(d) for d in datasets):
        grid_results = [
            {
                "l1_ratio": l1,
                "alpha": a,
                "mean_cv_mse": float("inf"),
                "std_cv_mse": 0.0,
                "n_variants_evaluated": 0,
            }
            for l1 in l1_ratios
            for a in alphas
        ]
        return GridSearchResult(
            best_l1_ratio=l1_ratios[0],
            best_alpha=alphas[0],
            best_mean_cv_mse=float("inf"),
            grid_results=grid_results,
            n_variants_sampled=n_variants_sampled,
            n_variants_failed=n_variants_sampled,
        )

    grid_results: List[Dict[str, Any]] = []
    failures_by_point: Dict[Tuple[float, float], List[str]] = {}
    best_mean_mse = float("inf")
    best_l1_ratio = l1_ratios[0]
    best_alpha = alphas[0]

    for l1_ratio in l1_ratios:
        for alpha in alphas:
            mse_values = []
            point_failures: List[str] = []
            for predictor_dosages, target in datasets:
                mse = _evaluate_one_dataset(
                    predictor_dosages,
                    target,
                    l1_ratio,
                    alpha,
                    cv_folds,
                    random_state,
                    fit_fn,
                    failure_sink=point_failures,
                )
                if mse is not None:
                    mse_values.append(mse)
            failures_by_point[(l1_ratio, alpha)] = point_failures

            if len(mse_values) > 0:
                mean_mse = float(np.mean(mse_values))
                std_mse = float(np.std(mse_values))
            else:
                mean_mse = float("inf")
                std_mse = 0.0

            grid_results.append({
                "l1_ratio": l1_ratio,
                "alpha": alpha,
                "mean_cv_mse": mean_mse,
                "std_cv_mse": std_mse,
                "n_variants_evaluated": len(mse_values),
            })

            if mean_mse < best_mean_mse:
                best_mean_mse = mean_mse
                best_l1_ratio = l1_ratio
                best_alpha = alpha

    total_failed = 0
    for result in grid_results:
        if result["l1_ratio"] == best_l1_ratio and result["alpha"] == best_alpha:
            total_failed = n_variants_sampled - result["n_variants_evaluated"]
            break

    if best_mean_mse == float("inf"):
        raise ValidationError(
            "All models failed during hyperparameter search. "
            "Check that input data has sufficient variance and valid samples."
        )

    # Genuine fit exceptions at the best grid point, by exception class. A subset
    # of ``n_variants_failed`` (which also counts intercept-only/no-eval
    # datasets); surfaces *why* fits raised (P5.1).
    failure_reasons = _tally_failure_reasons(
        failures_by_point.get((best_l1_ratio, best_alpha), [])
    )

    return GridSearchResult(
        best_l1_ratio=best_l1_ratio,
        best_alpha=best_alpha,
        best_mean_cv_mse=best_mean_mse,
        grid_results=grid_results,
        n_variants_sampled=n_variants_sampled,
        n_variants_failed=total_failed,
        failure_reasons=failure_reasons,
    )


# ---------------------------------------------------------------------------
# Dataset builders (mirror training's matrix construction exactly)
# ---------------------------------------------------------------------------


def _build_local_window_datasets(
    Z: np.ndarray,
    X_missing: np.ndarray,
    missing_variant_info: "Any",
    platform_variant_info: "Any",
    window_size: int,
    max_predictors: Optional[int],
    sample_indices: Sequence[int],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """Build ``(predictor, target)`` datasets for the sampled missing variants.

    Mirrors ``ImputationModelTrainer._fit_one_variant`` exactly: each predictor
    matrix is ``Z[:, chrom_index.window(...).variant_indices]`` (with
    ``exclude_target=True`` and ``max_variants=max_predictors``), and each target
    is ``X_missing[:, col]``. ``missing_variant_info`` must be row-aligned with the
    columns of ``X_missing``.
    """
    datasets: List[Tuple[np.ndarray, np.ndarray]] = []
    kept: List[int] = []
    n_samples = Z.shape[0]
    # Phase 9: build the window index once over the platform frame (O(log n)
    # lookups) — returns the identical WindowFilterResult to filter_to_local_window.
    chrom_index = ChromosomeIndex(platform_variant_info)
    for col in sample_indices:
        row = missing_variant_info.iloc[col]
        window_result = chrom_index.window(
            target_chrom=str(row["chromosome"]),
            target_pos=int(row["position"]),
            window_size=window_size,
            exclude_target=True,
            max_variants=max_predictors,
        )
        if window_result.n_variants > 0:
            predictor = Z[:, window_result.variant_indices]
        else:
            predictor = np.empty((n_samples, 0))
        datasets.append((predictor, X_missing[:, col]))
        kept.append(int(col))
    return datasets, kept


# ---------------------------------------------------------------------------
# Per-variant tuning (used by tuning_scope="per_variant")
# ---------------------------------------------------------------------------


def tune_single_variant_model(
    target: np.ndarray,
    predictor_dosages: np.ndarray,
    l1_ratios: Optional[List[float]] = None,
    alphas: Optional[List[float]] = None,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    fit_fn: Callable[..., Any] = fit_single_variant_model,
) -> Any:
    """Grid-search a single ``(predictor, target)`` dataset; return the best fit.

    Fits every ``(l1_ratio, alpha)`` and returns the result with the lowest CV MSE,
    preferring non-intercept-only models. Falls back to an intercept-only result
    when no grid point produces signal. Returns the same result dataclass as
    ``fit_fn`` (so the trainer's downstream conversion is unchanged).
    """
    l1_ratios, alphas = _resolve_grids(l1_ratios, alphas)
    best_result = None
    best_mse = float("inf")
    intercept_only_result = None
    for l1_ratio in l1_ratios:
        for alpha in alphas:
            try:
                result = fit_fn(
                    target,
                    predictor_dosages,
                    l1_ratio=l1_ratio,
                    alpha=alpha,
                    cv_folds=cv_folds,
                    random_state=random_state,
                )
            except Exception:
                continue
            if result.is_intercept_only:
                if intercept_only_result is None:
                    intercept_only_result = result
                continue
            if result.cv_mse < best_mse:
                best_mse = result.cv_mse
                best_result = result
    if best_result is not None:
        return best_result
    if intercept_only_result is not None:
        return intercept_only_result
    # Every grid point raised: return a plain fit at the first grid point.
    return fit_fn(
        target,
        predictor_dosages,
        l1_ratio=l1_ratios[0],
        alpha=alphas[0],
        cv_folds=cv_folds,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Imputation: global grid search on local windows over a stratified sample
# ---------------------------------------------------------------------------


def global_hyperparameter_search(
    Z: np.ndarray,
    X_missing: np.ndarray,
    missing_variant_info: "Any",
    platform_variant_info: "Any",
    window_size: int,
    max_predictors: Optional[int] = None,
    max_tuning_variants: Optional[int] = None,
    l1_ratios: Optional[List[float]] = None,
    alphas: Optional[List[float]] = None,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> GridSearchResult:
    """Grid search for imputation on the same local-window matrices training uses.

    A stratified sample (by chromosome / MAF / |beta|) of at most
    ``max_tuning_variants`` missing variants is selected; for each, the predictor
    matrix is built with the identical local-window query the trainer makes
    (``ChromosomeIndex.window``), and the grid is scored with
    ``fit_single_variant_model``.

    Args:
        Z: Platform predictor dosages. Shape: (n_samples, n_platform_variants).
        X_missing: Missing-variant dosages. Shape: (n_samples, n_missing_variants).
            Column ``i`` is the target for ``missing_variant_info.iloc[i]``.
        missing_variant_info: DataFrame of missing variants (columns include
            ``chromosome``, ``position``, ``beta``), row-aligned with X_missing.
        platform_variant_info: DataFrame of platform variants (``chromosome``,
            ``position`` required), columns of Z.
        window_size: Local-window size in base pairs (must match training).
        max_predictors: Cap on predictors per window (must match training).
        max_tuning_variants: Sample size. None means use all missing variants.
        l1_ratios, alphas: Grids. Default to the module defaults.
        cv_folds, random_state: CV configuration.

    Returns:
        GridSearchResult with the best hyperparameters and the full grid.

    Raises:
        ValidationError: empty grids, out-of-range grid values, shape mismatch,
            or all fits failing despite available predictors.
    """
    Z = np.asarray(Z, dtype=np.float64)
    X_missing = np.asarray(X_missing, dtype=np.float64)
    if X_missing.ndim == 1:
        X_missing = X_missing.reshape(-1, 1)

    l1_ratios, alphas = _resolve_grids(l1_ratios, alphas)
    _check_sample_shapes(Z, X_missing)

    n_missing = X_missing.shape[1]
    if n_missing == 0:
        return _grid_search_over_datasets(
            [], l1_ratios, alphas, cv_folds, random_state,
            n_variants_sampled=0, fit_fn=fit_single_variant_model,
        )

    stratum_keys = _imputation_stratum_keys(X_missing, missing_variant_info)
    sample_indices = select_stratified_sample(
        stratum_keys, max_tuning_variants, random_state
    )
    datasets, kept = _build_local_window_datasets(
        Z, X_missing, missing_variant_info, platform_variant_info,
        window_size, max_predictors, sample_indices,
    )
    return _grid_search_over_datasets(
        datasets, l1_ratios, alphas, cv_folds, random_state,
        n_variants_sampled=len(kept), fit_fn=fit_single_variant_model,
    )


def optuna_hyperparameter_search(
    Z: np.ndarray,
    X_missing: np.ndarray,
    missing_variant_info: "Any",
    platform_variant_info: "Any",
    window_size: int,
    max_predictors: Optional[int] = None,
    max_tuning_variants: Optional[int] = None,
    n_trials: int = 50,
    l1_ratio_range: Tuple[float, float] = (0.0, 1.0),
    alpha_range: Tuple[float, float] = (1e-4, 1.0),
    cv_folds: int = 5,
    seed: Optional[int] = None,
    timeout: Optional[float] = None,
    show_progress: bool = False,
) -> OptunaSearchResult:
    """Bayesian (Optuna TPE) search for imputation on local-window matrices.

    Like :func:`global_hyperparameter_search` but with continuous ranges and a TPE
    sampler. The local-window datasets are built once (windows are independent of
    the hyperparameters) and reused across all trials.

    Raises:
        ImportError: if optuna is not installed.
        ValidationError: if inputs are invalid.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as e:
        raise ImportError(
            "Optuna is required for Bayesian hyperparameter optimization. "
            "Install it with: pip install optuna"
        ) from e

    Z = np.asarray(Z, dtype=np.float64)
    X_missing = np.asarray(X_missing, dtype=np.float64)
    if X_missing.ndim == 1:
        X_missing = X_missing.reshape(-1, 1)

    if len(l1_ratio_range) != 2:
        raise ValidationError("l1_ratio_range must be a tuple of (min, max)")
    l1_min, l1_max = l1_ratio_range
    if not (0.0 <= l1_min <= l1_max <= 1.0):
        raise ValidationError(
            f"l1_ratio_range must satisfy 0 <= min <= max <= 1, "
            f"got ({l1_min}, {l1_max})"
        )

    if len(alpha_range) != 2:
        raise ValidationError("alpha_range must be a tuple of (min, max)")
    alpha_min, alpha_max = alpha_range
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValidationError(
            f"alpha_range values must be positive, got ({alpha_min}, {alpha_max})"
        )
    if alpha_min > alpha_max:
        raise ValidationError(
            f"alpha_range min must be <= max, got ({alpha_min}, {alpha_max})"
        )

    _check_sample_shapes(Z, X_missing)

    n_missing = X_missing.shape[1]
    if n_missing == 0:
        return OptunaSearchResult(
            best_l1_ratio=l1_min,
            best_alpha=alpha_min,
            best_mean_cv_mse=float("inf"),
            n_trials=0,
            n_variants_sampled=0,
            n_variants_failed=0,
            trial_history=[],
            optimization_time_seconds=0.0,
        )

    stratum_keys = _imputation_stratum_keys(X_missing, missing_variant_info)
    sample_indices = select_stratified_sample(stratum_keys, max_tuning_variants, seed)
    datasets, kept = _build_local_window_datasets(
        Z, X_missing, missing_variant_info, platform_variant_info,
        window_size, max_predictors, sample_indices,
    )
    n_variants_sampled = len(kept)

    # Edge case: no predictors anywhere.
    if not any(_dataset_has_predictors(d) for d in datasets):
        return OptunaSearchResult(
            best_l1_ratio=l1_min,
            best_alpha=alpha_min,
            best_mean_cv_mse=float("inf"),
            n_trials=0,
            n_variants_sampled=n_variants_sampled,
            n_variants_failed=n_variants_sampled,
            trial_history=[],
            optimization_time_seconds=0.0,
        )

    trial_history: List[dict] = []
    trial_failures: Dict[int, List[str]] = {}

    def objective(trial: "optuna.Trial") -> float:
        l1_ratio = trial.suggest_float("l1_ratio", l1_min, l1_max)
        alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)
        mse_values = []
        failure_sink: List[str] = []
        for predictor_dosages, target in datasets:
            mse = _evaluate_one_dataset(
                predictor_dosages, target, l1_ratio, alpha,
                cv_folds, seed, fit_single_variant_model,
                failure_sink=failure_sink,
            )
            if mse is not None:
                mse_values.append(mse)
        trial_failures[trial.number] = failure_sink
        mean_mse = float(np.mean(mse_values)) if mse_values else float("inf")
        trial_history.append({
            "trial_number": trial.number,
            "l1_ratio": l1_ratio,
            "alpha": alpha,
            "mean_cv_mse": mean_mse,
            "n_variants_evaluated": len(mse_values),
        })
        return mean_mse

    sampler = TPESampler(seed=seed)
    if not show_progress:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    start_time = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
    )
    optimization_time = time.time() - start_time

    if study.best_trial is not None:
        best_l1_ratio = study.best_trial.params["l1_ratio"]
        best_alpha = study.best_trial.params["alpha"]
        best_mean_cv_mse = study.best_value
        best_trial_entry = None
        for entry in trial_history:
            if entry["trial_number"] == study.best_trial.number:
                best_trial_entry = entry
                break
        if best_trial_entry is not None:
            n_variants_failed = n_variants_sampled - best_trial_entry["n_variants_evaluated"]
        else:
            n_variants_failed = 0
        failure_reasons = _tally_failure_reasons(
            trial_failures.get(study.best_trial.number, [])
        )
    else:
        best_l1_ratio = l1_min
        best_alpha = alpha_min
        best_mean_cv_mse = float("inf")
        n_variants_failed = n_variants_sampled
        failure_reasons = {}

    return OptunaSearchResult(
        best_l1_ratio=best_l1_ratio,
        best_alpha=best_alpha,
        best_mean_cv_mse=best_mean_cv_mse,
        n_trials=len(study.trials),
        n_variants_sampled=n_variants_sampled,
        n_variants_failed=n_variants_failed,
        trial_history=trial_history,
        optimization_time_seconds=optimization_time,
        failure_reasons=failure_reasons,
    )


# ---------------------------------------------------------------------------
# Projection: global grid search on region matrices over a stratified sample
# ---------------------------------------------------------------------------


def projection_hyperparameter_search(
    Z: np.ndarray,
    X: np.ndarray,
    prs_variants: "Any",
    platform_variant_info: "Any",
    window_size: int = 1_000_000,
    max_predictors: Optional[int] = None,
    max_tuning_regions: Optional[int] = None,
    l1_ratios: Optional[List[float]] = None,
    alphas: Optional[List[float]] = None,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> GridSearchResult:
    """Grid search for projection on the same region matrices training uses.

    The PRS variants are decomposed into regions with the identical
    ``merge_variant_windows`` call the trainer makes; each region's target is
    ``X[:, region.prs_variant_indices] @ betas`` and its predictors are
    ``Z[:, _find_platform_variants_in_region(...)]`` — exactly
    ``ProjectionRegionTrainer._fit_one_region``. A stratified sample (by predictor
    count / PRS-variant count) of at most ``max_tuning_regions`` regions is scored
    with ``fit_single_region_model``.

    Note: ``GridSearchResult.n_variants_sampled`` carries the number of **regions**
    sampled for projection.

    Raises:
        ValidationError: empty grids, out-of-range grid values, shape mismatch, or
            all region fits failing despite available predictors.
    """
    # Imported lazily to keep the module import graph acyclic regardless of the
    # order in which imputed_prs.models submodules are first imported.
    from imputed_prs.core.regions import merge_variant_windows
    from imputed_prs.models.projection import fit_single_region_model
    from imputed_prs.models.projection_trainer import _find_platform_variants_in_region

    Z = np.asarray(Z, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    l1_ratios, alphas = _resolve_grids(l1_ratios, alphas)
    _check_sample_shapes(Z, X, x_name="X")

    if X.shape[1] == 0:
        return _grid_search_over_datasets(
            [], l1_ratios, alphas, cv_folds, random_state,
            n_variants_sampled=0, fit_fn=fit_single_region_model,
        )

    prs_variants = prs_variants.reset_index(drop=True)
    decomposition = merge_variant_windows(prs_variants, window_size)

    datasets: List[Tuple[np.ndarray, np.ndarray]] = []
    stratum_keys: List[Tuple[str, str]] = []
    n_samples = Z.shape[0]
    for region in decomposition.regions:
        indices = region.prs_variant_indices
        betas = prs_variants.iloc[indices]["beta"].to_numpy(dtype=np.float64)
        target = X[:, indices] @ betas
        _, platform_indices = _find_platform_variants_in_region(
            region, platform_variant_info, max_predictors
        )
        if len(platform_indices) > 0:
            predictor = Z[:, platform_indices]
        else:
            predictor = np.empty((n_samples, 0))
        datasets.append((predictor, target))
        stratum_keys.append((
            _bucket_predictor_count(len(platform_indices)),
            _bucket_prs_count(len(indices)),
        ))

    sample_indices = select_stratified_sample(
        stratum_keys, max_tuning_regions, random_state
    )
    sampled = [datasets[i] for i in sample_indices]
    return _grid_search_over_datasets(
        sampled, l1_ratios, alphas, cv_folds, random_state,
        n_variants_sampled=len(sampled), fit_fn=fit_single_region_model,
    )
