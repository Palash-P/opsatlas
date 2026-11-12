"""Small dependency-free process/resource sampler for local load tests.

The sampler records per-process CPU time and working set without importing
``psutil``. On Windows it uses the Win32 process APIs through ``ctypes``. On
other platforms it returns an empty sample instead of adding a dependency.

Windows Python is launched through a small trampoline process on some managed
machines, so a single logical service can appear as a parent/child pair with
the same command line. The sampler resolves the whole ``python`` process family
for each root PID and reports the aggregate CPU and memory, which is what a
capacity measurement needs.

This is a local measurement helper, not an APM agent. It samples at a coarse
interval and must not be used as a production metrics source.
"""

from __future__ import annotations

import ctypes
import platform
import subprocess
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessSnapshot:
    pid: int
    label: str
    cpu_seconds: float
    working_set_bytes: int
    peak_working_set_bytes: int
    timestamp: float


@dataclass
class SamplerResult:
    processes: list[dict[str, Any]] = field(default_factory=list)
    containers: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_char * 260),
    ]


def _windows_snapshot(pid: int, label: str) -> ProcessSnapshot | None:
    if platform.system() != "Windows":
        return None

    process_query_limited_information = 0x1000
    process_vm_read = 0x0010
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]

    handle = kernel32.OpenProcess(
        process_query_limited_information | process_vm_read,
        False,
        pid,
    )
    if not handle:
        return None

    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        cpu_seconds = (
            (kernel.dwHighDateTime << 32 | kernel.dwLowDateTime)
            + (user.dwHighDateTime << 32 | user.dwLowDateTime)
        ) / 10_000_000

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            ctypes.sizeof(counters),
        ):
            return None
        return ProcessSnapshot(
            pid=pid,
            label=label,
            cpu_seconds=round(cpu_seconds, 4),
            working_set_bytes=int(counters.WorkingSetSize),
            peak_working_set_bytes=int(counters.PeakWorkingSetSize),
            timestamp=time.time(),
        )
    finally:
        kernel32.CloseHandle(handle)


def _process_table() -> list[tuple[int, int, str]]:
    """Return ``(pid, parent_pid, exe_name)`` for every visible process."""

    if platform.system() != "Windows":
        return []

    th32cs_snapprocess = 0x00000002
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32First.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32),
    ]
    kernel32.Process32Next.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32),
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    snapshot = kernel32.CreateToolhelp32Snapshot(th32cs_snapprocess, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or snapshot == invalid_handle:
        return []

    entry = _PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(entry)
    rows: list[tuple[int, int, str]] = []
    try:
        present = kernel32.Process32First(snapshot, ctypes.byref(entry))
        while present:
            name = entry.szExeFile.decode("mbcs", errors="replace")
            rows.append(
                (int(entry.th32ProcessID), int(entry.th32ParentProcessID), name)
            )
            present = kernel32.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return rows


def process_family(root_pid: int) -> list[int]:
    """Return ``root_pid`` plus every descendant ``python`` process."""

    children: dict[int, list[tuple[int, str]]] = {}
    for pid, parent_pid, name in _process_table():
        children.setdefault(parent_pid, []).append((pid, name))

    family = [root_pid]
    queue = [root_pid]
    while queue:
        current = queue.pop()
        for pid, name in children.get(current, []):
            if name.lower().startswith("python") and pid not in family:
                family.append(pid)
                queue.append(pid)
    return family


def _docker_stats(container: str) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}|{{.NetIO}}|{{.BlockIO}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split("|")
    if len(parts) != 5:
        return None
    return {
        "container": container,
        "cpu_percent": parts[0],
        "memory_usage": parts[1],
        "memory_percent": parts[2],
        "network_io": parts[3],
        "block_io": parts[4],
    }


class ResourceSampler:
    """Sample selected PIDs (and their Python children) and containers."""

    def __init__(
        self,
        *,
        processes: dict[str, int],
        containers: list[str],
        interval_seconds: float = 1.0,
    ) -> None:
        self.processes = dict(processes)
        self.containers = list(containers)
        self.interval_seconds = interval_seconds
        self.samples: list[ProcessSnapshot] = []
        self.container_samples: list[dict[str, Any]] = []
        self.families: dict[str, list[int]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._finished_at = 0.0
        self._error: str | None = None

    def _sample_once(self) -> None:
        for label, root_pid in self.processes.items():
            pids = self.families.get(label)
            if not pids:
                pids = process_family(root_pid)
                self.families[label] = pids
            for pid in pids:
                snapshot = _windows_snapshot(pid, label)
                if snapshot is not None:
                    self.samples.append(snapshot)

    def _run(self) -> None:
        try:
            self._sample_once()
            for container in self.containers:
                stats = _docker_stats(container)
                if stats is not None:
                    self.container_samples.append(stats)
            while not self._stop.wait(self.interval_seconds):
                self._sample_once()
        except Exception as exc:  # noqa: BLE001 - sampler must not fail a load test
            self._error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> SamplerResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._finished_at = time.time()
        return self.result()

    def result(self) -> SamplerResult:
        duration = max(0.0, self._finished_at - self._started_at)
        by_label_pid: dict[str, dict[int, list[ProcessSnapshot]]] = {}
        for sample in self.samples:
            by_label_pid.setdefault(sample.label, {}).setdefault(
                sample.pid, []
            ).append(sample)

        processes: list[dict[str, Any]] = []
        for label, by_pid in by_label_pid.items():
            cpu_seconds = 0.0
            working_set = 0
            peak_working_set = 0
            per_process: list[dict[str, Any]] = []
            for pid, samples in by_pid.items():
                ordered = sorted(samples, key=lambda item: item.timestamp)
                first, last = ordered[0], ordered[-1]
                delta = max(0.0, last.cpu_seconds - first.cpu_seconds)
                cpu_seconds += delta
                working_set += last.working_set_bytes
                peak_working_set += max(
                    sample.peak_working_set_bytes for sample in ordered
                )
                per_process.append(
                    {
                        "pid": pid,
                        "cpu_seconds": round(delta, 3),
                        "final_working_set_bytes": last.working_set_bytes,
                    }
                )
            processes.append(
                {
                    "label": label,
                    "process_count": len(by_pid),
                    "pids": sorted(by_pid),
                    "sample_count": sum(len(items) for items in by_pid.values()),
                    "cpu_seconds": round(cpu_seconds, 3),
                    "average_cpu_percent_of_one_core": round(
                        cpu_seconds / duration * 100 if duration else 0.0,
                        2,
                    ),
                    "peak_working_set_bytes": peak_working_set,
                    "final_working_set_bytes": working_set,
                    "per_process": per_process,
                }
            )

        return SamplerResult(
            processes=processes,
            containers=self.container_samples,
            duration_seconds=round(duration, 3),
            error=self._error,
        )


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Sample local process/container resources.")
    parser.add_argument("--pid", action="append", default=[], metavar="LABEL=PID")
    parser.add_argument("--container", action="append", default=[])
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    processes = {}
    for item in args.pid:
        label, _, value = item.partition("=")
        processes[label or f"pid-{value}"] = int(value)

    sampler = ResourceSampler(
        processes=processes,
        containers=args.container,
        interval_seconds=args.interval,
    )
    sampler.start()
    time.sleep(max(0.0, args.seconds))
    result = sampler.stop()
    print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
