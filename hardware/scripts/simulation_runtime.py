#!/usr/bin/env python3
"""Shared command execution and resource checks for the public test runners."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time

from artifact_paths import artifact_root

RTL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERILATOR = Path(os.environ.get("SUPRA_VERILATOR") or shutil.which("verilator") or "verilator")
DRAMSIM3_SOURCES = [f"third_party/dramsim3/src/{name}.cc" for name in (
    "bankstate", "channel_state", "command_queue", "common", "configuration",
    "controller", "dram_system", "hmc", "memory_system", "refresh", "simple_stats", "timing")]
INCLUDE_FLAGS = [f"-I{RTL_ROOT / name}" for name in (
    "tb/dramsim3", "third_party/dramsim3/src", "third_party/dramsim3/ext/headers",
    "third_party/dramsim3/ext/fmt/include")]


def _external(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(artifact_root())
    if resolved == RTL_ROOT or RTL_ROOT in resolved.parents:
        raise ValueError(f"run outputs must be outside the source directory: {resolved}")
    return resolved


def runtime_resource_check(run_dir: Path, log_path: Path,
                           runtime_threads: int, requested_workers: int) -> dict:
    if runtime_threads < 1 or requested_workers < 1:
        raise ValueError("runtime threads and workers must be positive")
    directory = _external(run_dir)
    _external(log_path)
    if not directory.is_dir() or not os.access(directory, os.W_OK | os.X_OK):
        raise ValueError(f"run directory must already exist and be writable: {directory}")
    disk = shutil.disk_usage(directory)
    free_inodes = os.statvfs(directory).f_favail
    cpus = len(os.sched_getaffinity(0))
    memory = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith("MemAvailable:"))
    errors = []
    if disk.free < 1024**3:
        errors.append("artifact filesystem requires at least 1 GiB free")
    if free_inodes < 4096:
        errors.append("artifact filesystem requires at least 4096 free inodes")
    if cpus < runtime_threads:
        errors.append(f"CPU affinity exposes {cpus} CPUs, but runtime requests {runtime_threads}")
    if memory < 1024**3:
        errors.append("runtime requires at least 1 GiB MemAvailable")
    record = dict(status="FAIL" if errors else "PASS", errors=errors,
                  artifact_free_bytes=disk.free, artifact_free_inodes=free_inodes,
                  affinity_cpus=cpus, mem_available_bytes=memory,
                  load_average_1_5_15=list(os.getloadavg()),
                  runtime_threads_per_case=runtime_threads,
                  requested_parallel_cases=requested_workers,
                  parallel_case_workers=max(1, min(requested_workers, cpus//runtime_threads,
                                                    memory//(1024**3))))
    with log_path.open("w") as log:
        for command in (["test", "-d", str(directory)], ["test", "-w", str(directory)],
                        ["df", "-h", str(directory)], ["df", "-i", str(directory)]):
            log.write("COMMAND " + shlex.join(command) + "\n")
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(result.stdout)
            if result.returncode:
                raise RuntimeError(f"resource command failed: {shlex.join(command)}")
        snapshot = subprocess.run(["ps", "-eo", "pid,etime,pcpu,rss,args", "--sort=-pcpu"],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        log.write("\n".join(snapshot.stdout.splitlines()[:11]) + "\n")
        log.write("RESOURCE_RECORD " + json.dumps(record) + "\n")
    return record


def pin_runtime_threads(pid: int, runtime_threads: int) -> list[int]:
    """Use only an explicitly assigned CPU set, with one CPU per model thread."""
    try:
        cpus = sorted(os.sched_getaffinity(pid))
        tids = sorted(int(path.name) for path in Path(f"/proc/{pid}/task").iterdir())
        if len(cpus) != runtime_threads or len(tids) != runtime_threads:
            return []
        for tid, cpu in zip(tids, cpus):
            os.sched_setaffinity(tid, {cpu})
        return cpus
    except (ProcessLookupError, FileNotFoundError):
        return []  # A very short process may already have exited.


def _stop_process_group(process: subprocess.Popen, grace_seconds: float = 2.0) -> None:
    """Stop only the session created for this command, including its children."""
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    send(signal.SIGKILL)
    process.wait(timeout=grace_seconds)


def run_command(name: str, command: list[str], log_path: Path,
                cwd: Path = RTL_ROOT, required_text: str | None = None,
                environment: dict[str, str] | None = None, stream_output: bool = False,
                thread_settings: dict[str, int] | None = None,
                expected_runtime_threads: int | None = None) -> dict:
    _external(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ if environment is None else environment)
    environment.setdefault("SUPRA_ARTIFACT_ROOT", str(artifact_root()))
    environment["CCACHE_DISABLE"] = "1"
    environment.setdefault("VERILATOR_NUMA_STRATEGY", "none")
    if expected_runtime_threads is not None:
        environment["SUPRA_RUNTIME_THREADS"] = str(expected_runtime_threads)
    command = list(map(str, command))
    started = time.monotonic()
    found = not required_text
    threads = []
    tail: deque[str] = deque(maxlen=30)
    with log_path.open("w", buffering=1) as log:
        log.write("COMMAND " + shlex.join(command) + "\n")
        log.write("VERILATOR_NUMA_STRATEGY " + environment["VERILATOR_NUMA_STRATEGY"] + "\n")
        if thread_settings is not None:
            log.write("THREAD_SETTINGS " + json.dumps(thread_settings) + "\n")
        previous_term = None
        if threading.current_thread() is threading.main_thread():
            previous_term = signal.getsignal(signal.SIGTERM)
            def terminated(signum, frame):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, terminated)
        process = None
        try:
            process = subprocess.Popen(command, cwd=cwd, env=environment, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       bufsize=1, start_new_session=True)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                tail.append(line)
                found |= bool(required_text and required_text in line)
                match = re.fullmatch(r"VERILATOR_MODEL_THREADS ([0-9]+)\s*", line)
                if match:
                    threads.append(int(match[1]))
                    if expected_runtime_threads == int(match[1]):
                        cpus = pin_runtime_threads(process.pid, expected_runtime_threads)
                        if cpus:
                            affinity = "VERILATOR_CPU_AFFINITY " + ",".join(map(str, cpus)) + "\n"
                            log.write(affinity)
                            if stream_output:
                                print(affinity, end="", flush=True)
                if stream_output:
                    print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            if previous_term is not None:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if process is not None:
                try:
                    _stop_process_group(process)
                except (OSError, subprocess.TimeoutExpired) as error:
                    log.write(f"PROCESS_CLEANUP_FAILED {error}\n")
            raise
        finally:
            if previous_term is not None:
                signal.signal(signal.SIGTERM, previous_term)
            if process is not None and process.stdout is not None:
                process.stdout.close()
        failures = []
        if code:
            failures.append(f"command exit status {code}")
        if not found:
            failures.append(f"missing required output: {required_text!r}")
        if expected_runtime_threads is not None and threads != [expected_runtime_threads]:
            failures.append(f"expected runtime threads {[expected_runtime_threads]}, observed {threads}")
        log.write(f"COMMAND_EXIT_STATUS {code}\n")
        for failure in failures:
            log.write("RUNNER_CHECK_FAILED " + failure + "\n")
    result = dict(name=name, status="FAIL" if failures else "PASS", exit_code=code,
                  failure_reasons=failures, elapsed_seconds=round(time.monotonic()-started, 3),
                  command=shlex.join(command), log=str(log_path.resolve()), log_bytes=log_path.stat().st_size)
    if expected_runtime_threads is not None:
        result["actual_runtime_threads"] = threads[0] if len(threads) == 1 else None
    if failures:
        raise RuntimeError(f"{name}: {'; '.join(failures)}; see {log_path}\n{''.join(tail)}")
    print(f"PASS {name} elapsed_seconds={result['elapsed_seconds']}", flush=True)
    return result
