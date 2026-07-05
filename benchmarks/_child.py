"""Isolated worker entry point: ``python -m benchmarks._child <run_dir>``.

Reads ``spec.json``, runs the scenario under a soft-memory watchdog and (optionally)
tracemalloc attribution, and writes ``child_result.json`` (atomically, fsync'd). Peak RSS
is measured by the *parent* via ``os.wait4``; this process reports wall time, the
tracemalloc peak, per-site attribution, and sub-phase timings.

Exit codes: 0 = completed, ``ERR_EXIT_CODE`` = caught exception, ``MEM_EXIT_CODE`` =
watchdog soft-ceiling abort. A real OS OOM SIGKILLs the process (no child_result.json);
the parent classifies that as ``killed_signal``.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from benchmarks.harness import (
    ERR_EXIT_CODE,
    MEM_EXIT_CODE,
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    OUTCOME_MEMORY_CEILING,
    MeasurementResult,
    WorkSpec,
    dumps,
)
from benchmarks.meters import PhaseRegistry, RSSWatchdog, TraceMallocAttribution
from benchmarks.scenarios import run_scenario


def _write(run_dir: Path, res: MeasurementResult) -> None:
    """Atomically write child_result.json, fsync'd so it survives an immediate _exit."""
    tmp = run_dir / "child_result.json.tmp"
    with open(tmp, "w") as fh:
        fh.write(dumps(res.to_dict()))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, run_dir / "child_result.json")


def main(run_dir_str: str) -> None:
    run_dir = Path(run_dir_str)
    spec = WorkSpec.from_dict(json.loads((run_dir / "spec.json").read_text()))

    registry = PhaseRegistry()
    write_lock = threading.Lock()
    done = {"written": False}

    def on_breach(current_bytes: int) -> None:
        with write_lock:
            if not done["written"]:
                done["written"] = True
                _write(
                    run_dir,
                    MeasurementResult(
                        spec=spec,
                        outcome=OUTCOME_MEMORY_CEILING,
                        run_id=run_dir.name,
                        phases=registry.to_list(),
                        error_message=f"soft memory ceiling exceeded (~{current_bytes} bytes)",
                    ),
                )
        os._exit(MEM_EXIT_CODE)

    watchdog = None
    if spec.soft_ceiling_bytes:
        watchdog = RSSWatchdog(int(spec.soft_ceiling_bytes), on_breach)
        watchdog.start()

    attr = None
    if spec.tracemalloc:
        attr = TraceMallocAttribution(depth=spec.tracemalloc_depth, top_n=spec.attribution_top_n)
        attr.__enter__()

    t0 = time.perf_counter()
    try:
        payload = run_scenario(spec, registry)
        wall = time.perf_counter() - t0
        tm_peak = None
        per_site = []
        method = "none"
        if attr is not None:
            attr.__exit__(None, None, None)
            tm_peak = attr.peak_bytes
            per_site = attr.sites
            method = "tracemalloc"
        if watchdog is not None:
            watchdog.stop()
        with write_lock:
            if done["written"]:
                return
            done["written"] = True
            _write(
                run_dir,
                MeasurementResult(
                    spec=spec,
                    outcome=OUTCOME_COMPLETED,
                    run_id=run_dir.name,
                    wall_seconds=wall,
                    tracemalloc_peak_bytes=tm_peak,
                    attribution_method=method,
                    per_site=per_site,
                    phases=registry.to_list(),
                    result=payload,
                ),
            )
    except BaseException as exc:  # report every failure as a data point
        wall = time.perf_counter() - t0
        if attr is not None:
            try:
                attr.__exit__(None, None, None)
            except Exception:
                pass
        if watchdog is not None:
            watchdog.stop()
        with write_lock:
            if done["written"]:
                raise
            done["written"] = True
            _write(
                run_dir,
                MeasurementResult(
                    spec=spec,
                    outcome=OUTCOME_ERROR,
                    run_id=run_dir.name,
                    wall_seconds=wall,
                    phases=registry.to_list(),
                    exception_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                    traceback_tail="".join(traceback.format_exc())[-4000:],
                ),
            )
        os._exit(ERR_EXIT_CODE)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m benchmarks._child <run_dir>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
