"""Parent-side measurement harness.

Runs one unit of work in a *fresh subprocess* and records wall-clock, an authoritative
peak RSS, and (optionally) per-allocation-site attribution, emitting one structured JSON
record per run.

Why a subprocess per measurement (both verified on the target machine, Darwin/arm64,
CPython 3.14):

* ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is a process-wide **monotonic
  high-water mark** — it never decreases, so sequential in-process measurements cannot be
  isolated. A new interpreter per unit of work yields a clean per-operation peak.
* ``os.wait4`` returns the child's rusage **even when the child was SIGKILLed** — which is
  exactly how an OOM (or a soft-ceiling watchdog abort) becomes a recorded data point
  instead of taking the driver down with it.

``ru_maxrss`` is in bytes on Darwin and KiB on Linux (:func:`normalize_maxrss`).
``RLIMIT_AS`` is rejected on Darwin, so memory is bounded by the in-child watchdog in
:mod:`benchmarks.meters`, not a hard rlimit.

File protocol per run (all under ``results/raw/<run_id>/``):
    ``spec.json``          parent → child work spec
    ``child_result.json``  child → parent partial result (also written by the watchdog)
    ``result.json``        parent-written final canonical record
    ``stdout.log`` / ``stderr.log``   child's redirected streams (library prints/logging)
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform as _platform
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1"
HARNESS_VERSION = "0.1.0"

#: Child exit code used by the soft-memory-ceiling watchdog (distinct from an OS kill).
MEM_EXIT_CODE = 42
#: Child exit code after a caught scenario exception (details in child_result.json).
ERR_EXIT_CODE = 3

OUTCOME_COMPLETED = "completed"
OUTCOME_MEMORY_CEILING = "memory_ceiling_exceeded"
OUTCOME_KILLED_SIGNAL = "killed_signal"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_CRASHED = "crashed"
OUTCOME_ERROR = "error"
OUTCOME_SPAWN_FAILED = "spawn_failed"

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_POLL_INTERVAL_S = 0.05


# --------------------------------------------------------------------------------------
# Peak-RSS unit normalization
# --------------------------------------------------------------------------------------
def normalize_maxrss(raw: int, system: Optional[str] = None) -> int:
    """Convert ``ru_maxrss`` to **bytes**.

    Darwin already reports bytes; Linux reports KiB (×1024). Verified: a 256 MiB
    allocation moves ``ru_maxrss`` by 268,451,840 on this Darwin machine.
    """
    if system is None:
        system = _platform.system()
    return int(raw) if system == "Darwin" else int(raw) * 1024


def _instantiate(cls, d: Dict[str, Any]):
    """Construct a dataclass from a dict, ignoring unknown keys (forward-compatible)."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def json_default(o: Any):
    """``json.dumps`` fallback that serializes numpy scalars/arrays, sets, and Paths.

    Model ``summary`` dicts and calibration params carry numpy floats, which the stdlib
    encoder rejects; this keeps every result JSON round-trippable.
    """
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def dumps(obj: Any) -> str:
    """``json.dumps`` with the numpy-aware default and stable indentation."""
    return json.dumps(obj, indent=2, default=json_default)


# --------------------------------------------------------------------------------------
# Schema dataclasses
# --------------------------------------------------------------------------------------
@dataclass
class RunMetadata:
    """Reproducibility context captured once per run (parent side)."""

    schema_version: str
    harness_version: str
    timestamp_utc: str
    git_sha: str
    git_short_sha: str
    git_dirty: Optional[bool]
    hostname: str
    platform: str
    system: str
    machine: str
    os_maxrss_unit: str
    python_version: str
    cpu_count: Optional[int]
    total_ram_bytes: Optional[int]
    lib_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunMetadata":
        return _instantiate(cls, d)


@dataclass
class WorkSpec:
    """Describes one unit of work for the child to execute."""

    operation: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    n_samples: Optional[int] = None
    n_variants: Optional[int] = None
    seed: Optional[int] = None
    soft_ceiling_bytes: Optional[int] = None
    tracemalloc: bool = False
    tracemalloc_depth: int = 30
    attribution_top_n: int = 15

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkSpec":
        return _instantiate(cls, d)


@dataclass
class SubphaseTiming:
    """Wall-time (and optional tracemalloc deltas) for one sub-phase of a run."""

    name: str
    wall_seconds: float
    tm_current_bytes: Optional[int] = None
    tm_delta_bytes: Optional[int] = None


@dataclass
class SiteAttribution:
    """Bytes attributed to one source location (file:line) via tracemalloc."""

    filename: str
    lineno: int
    size_bytes: int
    count: int
    function: Optional[str] = None


@dataclass
class MeasurementResult:
    """The canonical record for one measured run."""

    spec: WorkSpec
    outcome: str
    run_id: str
    metadata: Optional[RunMetadata] = None
    wall_seconds: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    peak_rss_is_authoritative: bool = True
    child_ru_utime_s: Optional[float] = None
    child_ru_stime_s: Optional[float] = None
    tracemalloc_peak_bytes: Optional[int] = None
    attribution_method: str = "none"
    per_site: List[SiteAttribution] = field(default_factory=list)
    phases: List[SubphaseTiming] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    exception_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback_tail: Optional[str] = None
    stderr_tail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MeasurementResult":
        d = dict(d)
        if d.get("metadata") is not None:
            d["metadata"] = RunMetadata.from_dict(d["metadata"])
        if d.get("spec") is not None:
            d["spec"] = WorkSpec.from_dict(d["spec"])
        d["per_site"] = [_instantiate(SiteAttribution, s) for s in d.get("per_site") or []]
        d["phases"] = [_instantiate(SubphaseTiming, p) for p in d.get("phases") or []]
        return _instantiate(cls, d)

    @property
    def ok(self) -> bool:
        return self.outcome == OUTCOME_COMPLETED


# --------------------------------------------------------------------------------------
# Metadata collection
# --------------------------------------------------------------------------------------
def _git(args: List[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _total_ram_bytes() -> Optional[int]:
    system = _platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            return int(out) if out else None
        if system == "Linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _lib_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for mod in ("numpy", "scipy", "sklearn", "pandas", "cyvcf2"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "unavailable"
    return out


def collect_metadata() -> RunMetadata:
    """Capture git/platform/library context for reproducibility and cross-phase diffing."""
    sha = _git(["rev-parse", "HEAD"])
    porcelain = _git(["status", "--porcelain"]) if sha else ""
    system = _platform.system()
    return RunMetadata(
        schema_version=SCHEMA_VERSION,
        harness_version=HARNESS_VERSION,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=sha or "unknown",
        git_short_sha=(sha[:10] if sha else "unknown"),
        git_dirty=(bool(porcelain) if sha else None),
        hostname=_platform.node(),
        platform=_platform.platform(),
        system=system,
        machine=_platform.machine(),
        os_maxrss_unit=("bytes" if system == "Darwin" else "kib"),
        python_version=_platform.python_version(),
        cpu_count=os.cpu_count(),
        total_ram_bytes=_total_ram_bytes(),
        lib_versions=_lib_versions(),
    )


# --------------------------------------------------------------------------------------
# Subprocess spawn + reap
# --------------------------------------------------------------------------------------
def _make_run_id(spec: WorkSpec) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in spec.label)[:40]
    return f"{ts}__{spec.operation}__{safe}__{uuid.uuid4().hex[:8]}"


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"
    # Pin hash randomization so measured oracles are reproducible run-to-run. The library's
    # fit has a set/dict iteration-order dependence (calibration/R^2 drift ~1e-3 across
    # PYTHONHASHSEED values) beyond `random_state`; pinning it keeps the parity oracle exact.
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def _reap(pid: int, timeout_s: Optional[float]):
    """Wait for ``pid``. Returns ``(status, rusage, timed_out)``.

    On timeout the child is SIGKILLed and reaped (its rusage is still returned).
    """
    if timeout_s is None:
        _p, status, rusage = os.wait4(pid, 0)
        return status, rusage, False
    deadline = time.perf_counter() + timeout_s
    while True:
        wpid, status, rusage = os.wait4(pid, os.WNOHANG)
        if wpid == pid:
            return status, rusage, False
        if time.perf_counter() >= deadline:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _p, status, rusage = os.wait4(pid, 0)
            return status, rusage, True
        time.sleep(_POLL_INTERVAL_S)


def _tail(path: Path, n_chars: int = 4000) -> str:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    return text[-n_chars:]


def _finalize(res: MeasurementResult, run_dir: Path) -> MeasurementResult:
    (run_dir / "result.json").write_text(dumps(res.to_dict()))
    return res


def measure(
    spec: WorkSpec,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    timeout_s: Optional[float] = None,
    metadata: Optional[RunMetadata] = None,
) -> MeasurementResult:
    """Run ``spec`` in an isolated subprocess and return its :class:`MeasurementResult`.

    Never raises on a *child* failure — an OOM/kill/exception/timeout is classified into
    ``outcome`` and returned as a data point.
    """
    results_dir = Path(results_dir)
    run_id = _make_run_id(spec)
    run_dir = results_dir / "raw" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(dumps(spec.to_dict()))
    if metadata is None:
        metadata = collect_metadata()

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    file_actions = [
        (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
        (os.POSIX_SPAWN_OPEN, 1, str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644),
        (os.POSIX_SPAWN_OPEN, 2, str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644),
    ]
    argv = [sys.executable, "-m", "benchmarks._child", str(run_dir)]

    t0 = time.perf_counter()
    try:
        pid = os.posix_spawn(sys.executable, argv, _child_env(), file_actions=file_actions)
    except Exception as exc:  # spawn itself failed
        return _finalize(
            MeasurementResult(
                spec=spec,
                outcome=OUTCOME_SPAWN_FAILED,
                run_id=run_id,
                metadata=metadata,
                exception_type=type(exc).__name__,
                error_message=str(exc),
            ),
            run_dir,
        )

    status, rusage, timed_out = _reap(pid, timeout_s)
    parent_wall = time.perf_counter() - t0

    child: Optional[MeasurementResult] = None
    child_path = run_dir / "child_result.json"
    if child_path.exists():
        try:
            child = MeasurementResult.from_dict(json.loads(child_path.read_text()))
        except Exception:
            child = None

    peak = normalize_maxrss(rusage.ru_maxrss, metadata.system)
    exited = os.WIFEXITED(status)
    signaled = os.WIFSIGNALED(status)
    exit_code = os.WEXITSTATUS(status) if exited else None
    term_sig = os.WTERMSIG(status) if signaled else None

    if timed_out:
        outcome = OUTCOME_TIMEOUT
    elif child is not None:
        outcome = child.outcome
    elif signaled:
        outcome = OUTCOME_KILLED_SIGNAL
    elif exited and exit_code == MEM_EXIT_CODE:
        outcome = OUTCOME_MEMORY_CEILING
    else:
        outcome = OUTCOME_CRASHED

    res = MeasurementResult(
        spec=spec,
        outcome=outcome,
        run_id=run_id,
        metadata=metadata,
        wall_seconds=(child.wall_seconds if child and child.wall_seconds is not None else parent_wall),
        peak_rss_bytes=peak,
        peak_rss_is_authoritative=(not spec.tracemalloc),
        child_ru_utime_s=float(rusage.ru_utime),
        child_ru_stime_s=float(rusage.ru_stime),
        tracemalloc_peak_bytes=(child.tracemalloc_peak_bytes if child else None),
        attribution_method=(child.attribution_method if child else "none"),
        per_site=(child.per_site if child else []),
        phases=(child.phases if child else []),
        result=(child.result if child else None),
        exit_code=exit_code,
        signal=term_sig,
        exception_type=(child.exception_type if child else None),
        error_message=(child.error_message if child else None),
        traceback_tail=(child.traceback_tail if child else None),
        stderr_tail=_tail(stderr_path),
    )
    return _finalize(res, run_dir)


def sweep(
    specs: Iterable[WorkSpec],
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    timeout_s: Optional[float] = None,
) -> List[MeasurementResult]:
    """Run specs sequentially (shared metadata). Suitable for grow-until-OOM sweeps."""
    metadata = collect_metadata()
    return [measure(s, results_dir, timeout_s=timeout_s, metadata=metadata) for s in specs]


# --------------------------------------------------------------------------------------
# Results IO
# --------------------------------------------------------------------------------------
def write_result(res: MeasurementResult, path: Path) -> None:
    Path(path).write_text(dumps(res.to_dict()))


def read_result(path: Path) -> MeasurementResult:
    return MeasurementResult.from_dict(json.loads(Path(path).read_text()))


def load_results(results_dir: Path = DEFAULT_RESULTS_DIR) -> List[MeasurementResult]:
    out: List[MeasurementResult] = []
    for p in sorted(Path(results_dir).glob("raw/*/result.json")):
        try:
            out.append(read_result(p))
        except Exception:
            continue
    return out


def to_dataframe(results: List[MeasurementResult]):
    """Flatten results into a pandas DataFrame for before/after comparison."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append(
            {
                "run_id": r.run_id,
                "operation": r.spec.operation,
                "label": r.spec.label,
                "outcome": r.outcome,
                "n_samples": r.spec.n_samples,
                "n_variants": r.spec.n_variants,
                "wall_seconds": r.wall_seconds,
                "peak_rss_bytes": r.peak_rss_bytes,
                "peak_rss_is_authoritative": r.peak_rss_is_authoritative,
                "tracemalloc_peak_bytes": r.tracemalloc_peak_bytes,
                "n_variants_loaded": (r.result or {}).get("n_variants"),
            }
        )
    return pd.DataFrame(rows)
