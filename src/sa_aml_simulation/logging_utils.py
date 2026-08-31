"""Runtime logging helpers for script-based simulation runs."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO


class TeeStream:
    """Write text to multiple streams, for example terminal plus a log file."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _safe_filename_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return cleaned.strip("_") or "run"


def format_runtime(seconds: float) -> str:
    """Return a compact HH:MM:SS runtime string."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@contextmanager
def tee_run_log(log_dir: Path, run_type: str, run_name: str = "run", command_args: Iterable[str] | None = None):
    """Mirror stdout/stderr to a timestamped run log.

    The context keeps normal terminal output visible while also saving the full
    run transcript to disk. It records start time, finish time, runtime and
    failure status if an exception is raised.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    safe_run_type = _safe_filename_part(run_type)
    safe_run_name = _safe_filename_part(run_name)
    log_path = log_dir / f"{timestamp}_{safe_run_type}_{safe_run_name}.log"

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file = log_path.open("w", encoding="utf-8")
    start_counter = time.perf_counter()
    status = "COMPLETED"

    try:
        sys.stdout = TeeStream(old_stdout, log_file)
        sys.stderr = TeeStream(old_stderr, log_file)
        print("=" * 88, flush=True)
        print(f"Run type: {run_type}", flush=True)
        print(f"Run name: {run_name}", flush=True)
        print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
        if command_args is not None:
            print(f"Command arguments: {' '.join(command_args) if command_args else '(none)'}", flush=True)
        print(f"Log file: {log_path.resolve()}", flush=True)
        print("=" * 88, flush=True)
        yield log_path
    except BaseException as exc:
        status = f"FAILED ({type(exc).__name__}: {exc})"
        print(f"\nRun status: {status}", flush=True)
        raise
    finally:
        finished_at = datetime.now().astimezone()
        elapsed = time.perf_counter() - start_counter
        print("=" * 88, flush=True)
        print(f"Finished at: {finished_at.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
        print(f"Runtime: {format_runtime(elapsed)} ({elapsed:.2f} seconds)", flush=True)
        print(f"Status: {status}", flush=True)
        print("=" * 88, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()
