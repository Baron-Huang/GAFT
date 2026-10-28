#!/usr/bin/env python3
"""Run non-empty SOTA model scripts with a bounded parallel worker pool.

Run from the SOTAs_main_func directory, for example:
    python run_sotas_parallel.py

The default of two concurrent models is conservative for a 32 GB GPU. Increase
--max-parallel only after checking the per-model logs and GPU memory usage.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO


ROOT = Path(__file__).resolve().parent
EXCLUDED_SCRIPTS = frozenset((
    "ALOE.py",
    Path(__file__).name,
))
def discover_scripts() -> List[Path]:
    """Return runnable SOTA entry scripts in a stable order."""
    return sorted(
        path
        for path in ROOT.glob("*.py")
        if path.name not in EXCLUDED_SCRIPTS and path.stat().st_size > 0
    )


def get_free_gpu_memory_mb(gpu_index: int) -> Optional[int]:
    """Return currently free memory for one GPU, or None when nvidia-smi is unavailable."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def make_environment() -> Dict[str, str]:
    """Make SOTAs_main_func importable as a package for child processes."""
    environment = os.environ.copy()
    package_parent = str(ROOT.parent)
    prior = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        package_parent if not prior else f"{package_parent}{os.pathsep}{prior}"
    )
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum simultaneous model processes; default: 2 for a 32 GB GPU.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for model scripts; default: current interpreter.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index monitored with nvidia-smi; default: 0.",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=4096,
        help="Minimum GPU memory kept free after a launch; default: 4096 MB.",
    )
    parser.add_argument(
        "--reserve-mb-per-process",
        type=int,
        default=6000,
        help="Estimated GPU memory needed by one newly launched model; default: 6000 MB.",
    )
    parser.add_argument(
        "--launch-settle-seconds",
        type=float,
        default=30.0,
        help="Wait after each launch for CUDA memory to settle before another launch; default: 30 seconds.",
    )
    parser.add_argument(
        "--growth-buffer-mb",
        type=int,
        default=2048,
        help="Extra GPU memory reserved for training-time growth; default: 2048 MB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1")
    if args.gpu_index < 0:
        raise SystemExit("--gpu-index must be non-negative")
    if (args.min_free_mb < 0 or args.reserve_mb_per_process < 0
            or args.growth_buffer_mb < 0 or args.launch_settle_seconds < 0):
        raise SystemExit("GPU memory limits must be non-negative")

    scripts = discover_scripts()
    if not scripts:
        raise SystemExit("No eligible non-empty model scripts were found.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"sotas_run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    launcher_log = log_dir / "launcher.log"
    environment = make_environment()
    queued = list(scripts)
    active: Dict[Path, tuple[subprocess.Popen, TextIO, float]] = {}
    failures: List[Path] = []
    stopped = False
    last_wait_reason: Optional[str] = None
    last_launch_at: Optional[float] = None

    def terminate_process_tree(process: subprocess.Popen) -> None:
        """Terminate the model process and its DataLoader worker processes."""
        if process.poll() is not None:
            return
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)

    def write(message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        print(line, flush=True)
        with launcher_log.open("a", encoding="utf-8") as output:
            output.write(line + "\n")

    def stop_children(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True
        write("Interrupt received; terminating active model processes.")
        for process, _, _ in active.values():
            terminate_process_tree(process)

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)

    write(
        f"START scripts={len(scripts)} max_parallel={args.max_parallel} "
        f"python={args.python} gpu={args.gpu_index} "
        f"min_free_mb={args.min_free_mb} growth_buffer_mb={args.growth_buffer_mb} "
        f"reserve_mb_per_process={args.reserve_mb_per_process} "
        f"launch_settle_seconds={args.launch_settle_seconds}"
    )

    while queued or active:
        while queued and len(active) < args.max_parallel and not stopped:
            if last_launch_at is not None:
                elapsed_since_launch = time.monotonic() - last_launch_at
                if elapsed_since_launch < args.launch_settle_seconds:
                    wait_reason = (
                        f"WAIT CUDA settling elapsed_seconds={elapsed_since_launch:.0f} "
                        f"required_seconds={args.launch_settle_seconds:.0f} "
                        f"next={queued[0].name}"
                    )
                    if wait_reason != last_wait_reason:
                        write(wait_reason)
                        last_wait_reason = wait_reason
                    break

            free_memory_mb = get_free_gpu_memory_mb(args.gpu_index)
            if free_memory_mb is None:
                write("WARNING nvidia-smi is unavailable; launching by --max-parallel only")
            else:
                # Active processes have already allocated their measured memory.
                # Reserve only the expected allocation of the process about to start.
                projected_free_mb = free_memory_mb - args.reserve_mb_per_process
                required_free_mb = args.min_free_mb + args.growth_buffer_mb
                if projected_free_mb < required_free_mb:
                    wait_reason = (
                        f"WAIT GPU free_mb={free_memory_mb} projected_free_mb={projected_free_mb} "
                        f"required_mb={required_free_mb} (minimum={args.min_free_mb} "
                        f"+ growth_buffer={args.growth_buffer_mb}) active={len(active)} "
                        f"next={queued[0].name}"
                    )
                    if wait_reason != last_wait_reason:
                        write(wait_reason)
                        last_wait_reason = wait_reason
                    break
            last_wait_reason = None
            script = queued.pop(0)
            log_handle = (log_dir / f"{script.stem}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                [args.python, "-u", str(script)],
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
            started_at = time.monotonic()
            active[script] = (process, log_handle, started_at)
            last_launch_at = started_at
            free_memory_mb = get_free_gpu_memory_mb(args.gpu_index)
            memory_note = "unknown" if free_memory_mb is None else f"{free_memory_mb}MB"
            write(f"START {script.name} pid={process.pid} gpu_free_mb={memory_note}")

        for script, (process, log_handle, started) in list(active.items()):
            result = process.poll()
            if result is None:
                continue
            log_handle.close()
            elapsed = time.monotonic() - started
            write(f"EXIT {script.name} status={result} elapsed_seconds={elapsed:.1f}")
            if result != 0:
                failures.append(script)
            del active[script]

        if active:
            time.sleep(1)
        elif stopped:
            break

    for process, log_handle, _ in active.values():
        if process.poll() is None:
            terminate_process_tree(process)
            process.wait()
        log_handle.close()

    if stopped:
        write("STOPPED")
        return 130
    if failures:
        write("COMPLETE failures=" + ", ".join(path.name for path in failures))
        return 1
    write("COMPLETE all scripts exited successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


