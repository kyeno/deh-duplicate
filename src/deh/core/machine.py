# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Machine-state service: capture the box once, plan per-stage resource budgets.

Why this module exists:

- Stages used to guess their own parallelism (``min(cpu_count, 16)`` literals,
  a hard-coded worker pool of 2) and, worse, some materialised every image's
  buffers in RAM before doing any work - measured at 25.6 MB per 1920x1080
  image, i.e. ~25 GB for a 1001-image corpus.
- One service now captures what the machine *is* (cores, total RAM, GPU, tool
  versions) eagerly at startup, and answers what a *stage* may use given the
  workload it is about to run: ``plan(workload) -> Budget``.
- Live values (free RAM, free VRAM) change while a run proceeds, so they are
  re-probed behind a short TTL and shared with the stage threads.

``describe()`` produces the one-line ``[Machine] ...`` banner; ``dump()`` a
JSON-able snapshot for ``--machine`` / bug reports.

Process-wide access: :func:`machine` returns the single instance; tests use
:func:`set_machine` / :func:`reset_machine`. This is one of the two sanctioned
process-wide services (see CONTRIBUTING.md; the other is ``deh.config.DEFAULT``,
which is immutable). Everything else stays injected by the composition root.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Mapping, Optional

from .imageio import sample_image_pixels

_MB = 1024 * 1024

#: How long a live reading (free RAM / free VRAM) may be reused.
DEFAULT_TTL_S = 2.0
#: Never let stages commit more than this share of the machine's RAM.
RAM_RESERVE_FRAC = 0.25
#: ... and always keep at least this much free for the OS / other stages.
RAM_RESERVE_MIN_MB = 4096
#: CPU lanes left free for the main thread, heartbeats and the GUI.
CPU_RESERVE = 2
#: Per-worker-process cost on top of the stage buffers it holds (Python + PIL).
WORKER_OVERHEAD_MB = 64
#: Conservative assumptions when a probe cannot answer (documented, reported).
FALLBACK_RAM_MB = 8192
FALLBACK_CPUS = 4
#: Sanity bound for any stage pool - more workers than this never help.
MAX_WORKERS = 64


@dataclass(frozen=True)
class CpuState:
    """CPU capability: logical/affined cores, physical cores, 1-min load."""

    logical: int
    affinity: int
    physical: int | None = None
    load1: float | None = None


@dataclass(frozen=True)
class RamState:
    """RAM readings in MB; ``probed`` is False when the OS could not answer."""

    total_mb: int
    available_mb: int
    swap_total_mb: int = 0
    swap_free_mb: int = 0
    probed: bool = True


@dataclass(frozen=True)
class MachineState:
    """One consistent view of the machine (static facts + current free memory)."""

    captured_at: float
    host: str
    platform: str
    python: str
    cpu: CpuState
    ram: RamState
    gpu: object                      # core.device.GpuMemory (kept torch-free here)
    torch_version: str | None = None
    cuda: bool = False
    binaries: Optional[dict] = None     # {"exiftool": "12.70", ...}


# --- probes (injectable; module-level because they are pure readers) ---------

def _run(cmd, timeout: float = 10.0) -> str | None:
    """Run a probe command; returns decoded stdout/stderr text or None."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout or proc.stderr
    return out.decode("utf-8", "replace") if out else ""


def _meminfo_mb() -> tuple[int, int, int, int] | None:
    """``(total, available, swap_total, swap_free)`` MB, or None when unknown.

    Linux reads ``/proc/meminfo``, where ``MemAvailable`` is the kernel's own
    "what can be handed out without swapping" estimate. Other POSIX systems
    fall back to ``os.sysconf`` physical pages, which has no availability
    notion - the total is reported for both figures so callers treat it as an
    upper bound rather than a promise.
    """
    try:
        values = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                toks = rest.split()
                if toks:
                    values[key.strip()] = int(toks[0])      # kB
        total = values.get("MemTotal", 0) // 1024
        if total:
            avail = (values.get("MemAvailable")
                     or values.get("MemFree", 0)) // 1024
            return (total, avail, values.get("SwapTotal", 0) // 1024,
                    values.get("SwapFree", 0) // 1024)
    except (OSError, ValueError):
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page // _MB
        if total:
            return (int(total), int(total), 0, 0)
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _physical_cores() -> int | None:
    """Unique (physical id, core id) pairs from ``/proc/cpuinfo`` (else None)."""
    try:
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                if key == "physical id":
                    phys = value.strip()
                elif key == "core id":
                    core = value.strip()
        return len(pairs) or None
    except OSError:
        return None


def _cpu_state() -> CpuState:
    """Logical/affined/physical core counts plus the 1-minute load average."""
    logical = os.cpu_count() or 0
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    try:
        load1 = float(os.getloadavg()[0])
    except (OSError, AttributeError):
        load1 = None
    return CpuState(logical=logical, affinity=affinity,
                    physical=_physical_cores(), load1=load1)


#: How each tool reports its version (exiftool wants ``-ver``).
_VERSION_FLAGS = {"exiftool": ("-ver",)}


def _binary_version(name: str) -> str | None:
    """First line of ``<tool> --version`` (the bare name when it is silent;
    None when the tool is not installed). Documents the per-machine EXIF
    toolchain in ``--machine`` output and bug reports."""
    if shutil.which(name) is None:
        return None
    text = _run([name, *_VERSION_FLAGS.get(name, ("--version",))])
    if text is None:
        return name
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return name


def _default_gpu_probe():
    """``core.device.gpu_memory`` (imported lazily so this module stays light)."""
    from .device import gpu_memory
    return gpu_memory()


# --- workload contract -------------------------------------------------------

@dataclass(frozen=True)
class Workload:
    """What a stage is about to do, expressed in machine terms.

    ``item_bytes_worker`` is what one in-flight item costs *inside* a worker
    process (0 for thread pools), ``item_bytes_parent`` what the parent holds
    per in-flight item; ``processes`` selects the accounting model (worker
    processes pay ``WORKER_OVERHEAD_MB`` each). ``env_prefix`` makes the
    planner honour ``<PREFIX>_WORKERS`` / ``<PREFIX>_CHUNK`` overrides.
    """

    name: str
    n_items: int = 0
    item_bytes_parent: float = 0.0
    item_bytes_worker: float = 0.0
    processes: bool = False
    cpu_lanes_per_item: float = 1.0
    in_flight_per_worker: int = 2
    min_workers: int = 1
    cap_workers: int | None = None
    env_prefix: str = ""


@dataclass(frozen=True)
class Budget:
    """The planner's answer: how many workers, how much work in flight, why."""

    jobs: int
    chunk_items: int
    in_flight: int
    ram_budget_mb: int
    reserve_mb: int
    reasons: tuple = ()

    def describe(self) -> str:
        """One-line human summary, e.g. ``12 workers, chunk 24, RAM 12600 MB``."""
        return (f"{self.jobs} workers, chunk {self.chunk_items} items, "
                f"in-flight {self.in_flight}, RAM budget "
                f"{self.ram_budget_mb} MB (reserve {self.reserve_mb} MB)")

    def explain(self) -> str:
        """Multi-line trace of the planning decisions (verbose/--machine)."""
        return "\n".join(f"    - {r}" for r in self.reasons)


# --- the service -------------------------------------------------------------

class MachineService:
    """Captures the machine once and plans per-stage resource budgets.

    ``state`` is probed eagerly in ``__init__`` so a run can print what it
    assumed *before* doing any work; :meth:`live` re-reads the volatile parts
    (free RAM / free VRAM) behind ``ttl_s`` and is safe to call from stage
    threads. Every probe is injectable, which is what makes the planning math
    testable on fictional machines.
    """

    def __init__(self, *, probe_gpu: Callable = _default_gpu_probe,
                 meminfo: Callable = _meminfo_mb,
                 cpu: Callable = _cpu_state,
                 binary: Callable = _binary_version,
                 env: Mapping | None = None,
                 ttl_s: float = DEFAULT_TTL_S,
                 ram_reserve_frac: float = RAM_RESERVE_FRAC,
                 ram_reserve_min_mb: int = RAM_RESERVE_MIN_MB,
                 cpu_reserve: int = CPU_RESERVE,
                 binaries: tuple = ("exiftool",)):
        self._probe_gpu = probe_gpu
        self._meminfo = meminfo
        self._cpu_probe = cpu
        self._binary = binary
        self._env = env if env is not None else os.environ
        self._binaries = tuple(binaries)
        self.ttl_s = float(ttl_s)
        self.ram_reserve_frac = float(ram_reserve_frac)
        self.ram_reserve_min_mb = int(ram_reserve_min_mb)
        self.cpu_reserve = int(cpu_reserve)
        self._lock = threading.RLock()
        self._live_at = 0.0
        self._live_state = None
        self.state = self.capture()          # eager - "starts immediately"

    # --- capture ---------------------------------------------------------
    def capture(self) -> MachineState:
        """Probe everything now: cores, RAM, GPU, torch + EXIF versions."""
        mem = self._meminfo()
        if mem is None:
            ram = RamState(total_mb=FALLBACK_RAM_MB, available_mb=FALLBACK_RAM_MB,
                           probed=False)
        else:
            ram = RamState(total_mb=int(mem[0]), available_mb=int(mem[1]),
                           swap_total_mb=int(mem[2]), swap_free_mb=int(mem[3]))
        torch_version, cuda = None, False
        try:                                  # torch may not be importable
            import torch
            torch_version = torch.__version__
            cuda = bool(torch.cuda.is_available())
        except (ImportError, RuntimeError):   # not installed, or CUDA init fails
            pass
        try:
            gpu = self._probe_gpu()
        except (RuntimeError, OSError, ValueError):   # a broken probe never aborts a run
            gpu = None
        return MachineState(
            captured_at=time.time(),
            host=socket.gethostname(),
            platform=f"{platform.system()} {platform.release()}",
            python=platform.python_version(),
            cpu=self._cpu_probe(), ram=ram, gpu=gpu,
            torch_version=torch_version, cuda=cuda,
            binaries={name: self._binary(name) for name in self._binaries},
        )

    def live(self, refresh: bool = False) -> MachineState:
        """State with fresh free-RAM/free-VRAM readings (TTL-cached).

        Stages call this right before they allocate (back-pressure checks,
        banners). The cache keeps a burst of heartbeat calls from shelling out
        to ``nvidia-smi`` every tick; ``refresh=True`` forces a re-probe.
        """
        with self._lock:
            now = time.time()
            if (not refresh and self._live_state is not None
                    and now - self._live_at < self.ttl_s):
                return self._live_state
            mem = self._meminfo()
            ram = self.state.ram
            if mem is not None:
                ram = replace(ram, available_mb=int(mem[1]),
                              swap_total_mb=int(mem[2]),
                              swap_free_mb=int(mem[3]), probed=True)
            try:
                gpu = self._probe_gpu()
            except (RuntimeError, OSError, ValueError):
                gpu = self.state.gpu
            self._live_state = replace(self.state, captured_at=now,
                                       ram=ram, gpu=gpu)
            self._live_at = now
            return self._live_state

    def _env_int(self, key: str, default=None):
        """Read an integer env override (blank/malformed -> *default*)."""
        raw = str(self._env.get(key) or "").strip()
        if not raw:
            return default
        try:
            return int(float(raw))
        except ValueError:
            return default

    # --- planning --------------------------------------------------------
    def plan(self, workload: Workload) -> Budget:
        """How much parallelism *workload* may use on this machine right now.

        Invariants: stay inside the free-RAM
        budget, never occupy more than the usable CPU lanes, honour
        ``<PREFIX>_WORKERS`` / ``<PREFIX>_CHUNK`` overrides but clamp them to
        both ceilings, and always return at least ``min_workers`` so a caller
        never builds a zero-worker pool.
        """
        st = self.live()
        reasons = []

        # 1. CPU ceiling
        cores = st.cpu.affinity or st.cpu.logical or FALLBACK_CPUS
        cpu_reserve = max(0, self._env_int("DEH_CPU_RESERVE", self.cpu_reserve))

        usable = max(1, cores - cpu_reserve)
        lanes = max(0.05, float(workload.cpu_lanes_per_item or 1.0))
        by_cpu = max(1, int(usable / lanes))
        if st.cpu.affinity and st.cpu.affinity < st.cpu.logical:
            reasons.append(f"CPU: {st.cpu.affinity} of {st.cpu.logical} logical "
                           f"cores usable (affinity)")
        reasons.append(f"CPU: {usable} usable - {cpu_reserve} reserved -> "
                       f"{by_cpu} workers at {lanes:g} lane(s)/item")

        # 2. RAM ceiling
        total = st.ram.total_mb or FALLBACK_RAM_MB
        reserve_mb = max(int(total * self.ram_reserve_frac),
                         self.ram_reserve_min_mb)
        if st.ram.probed:
            avail = st.ram.available_mb
            reasons.append(f"RAM: {avail} MB free of {total} MB, "
                           f"{reserve_mb} MB reserve")
        else:
            avail = FALLBACK_RAM_MB
            reasons.append(f"RAM: probe unavailable - assuming {avail} MB free")
        budget = max(0, avail - reserve_mb)
        cap_env = self._env_int("DEH_MAX_RAM_MB")
        if cap_env is not None and cap_env < budget:
            reasons.append(f"RAM: DEH_MAX_RAM_MB={cap_env} caps {budget} MB")

            budget = max(0, cap_env)

        # 3. per-worker cost -> workers by RAM
        per_item = float(workload.item_bytes_worker if workload.processes
                         else workload.item_bytes_parent)
        in_flight = max(1, int(workload.in_flight_per_worker or 1))
        per_worker_mb = max(1.0, per_item * in_flight / _MB
                            + (WORKER_OVERHEAD_MB if workload.processes else 0))
        by_ram = max(1, int(budget // per_worker_mb))
        buffer_s = ("process buffers" if workload.processes
                    else "parent buffers")
        reasons.append(f"RAM: {per_worker_mb:.0f} MB/worker "
                       f"({buffer_s} x{in_flight} in flight) -> {by_ram} workers")

        # 4. combine, then apply the stage/hard caps
        jobs = max(int(workload.min_workers or 1), min(by_ram, by_cpu))
        cap = min(int(workload.cap_workers or MAX_WORKERS), MAX_WORKERS)
        if jobs > cap:
            reasons.append(f"cap: stage limit {cap} workers")
            jobs = cap

        # 5. explicit override wins inside the ceilings (never outside them)
        prefix = (workload.env_prefix or "").strip().upper()
        if prefix:
            wanted = self._env_int(f"{prefix}_WORKERS")
            if wanted is not None:
                jobs = max(1, min(int(wanted), by_ram, by_cpu, cap))
                if jobs != wanted:
                    reasons.append(f"{prefix}_WORKERS={wanted} clamped to "
                                   f"{jobs} (RAM/CPU ceiling)")
                else:
                    reasons.append(f"{prefix}_WORKERS={wanted} (override)")

        # 6. chunk = how many items one wave holds (bounds parent + worker RAM)
        default_chunk = min(64, max(8, jobs * 2))
        parent_cap = None
        if workload.item_bytes_parent > 0:
            parent_ceiling = max(64.0 * _MB, 0.25 * budget * _MB)
            parent_cap = max(1, int(parent_ceiling // workload.item_bytes_parent))
        chunk = default_chunk
        if workload.n_items:
            chunk = min(chunk, max(1, int(workload.n_items)))
        if parent_cap:
            chunk = min(chunk, parent_cap)
        if prefix:
            wanted = self._env_int(f"{prefix}_CHUNK")
            if wanted and wanted > 0:
                chunk = max(1, min(int(wanted), parent_cap or 64, 64))
                if chunk != wanted:
                    reasons.append(f"{prefix}_CHUNK={wanted} clamped to {chunk}")
                else:
                    reasons.append(f"{prefix}_CHUNK={wanted} (override)")
        if chunk != default_chunk:
            reasons.append(f"chunk: {default_chunk} -> {chunk} items "
                           f"(item count / memory ceiling)")
        reasons.append(f"chunk: {chunk} items, {jobs * in_flight} in flight")

        return Budget(jobs=jobs, chunk_items=chunk, in_flight=jobs * in_flight,
                      ram_budget_mb=int(budget), reserve_mb=int(reserve_mb),
                      reasons=tuple(reasons))

    def plan_images(self, name: str, paths, *, parent_bytes_per_px: float = 3.0,
                    parent_fixed_bytes: float = 0.0,
                    worker_bytes_per_px: float = 0.0,
                    worker_fixed_bytes: float = 0.0, processes: bool = False,
                    in_flight_per_worker: int = 2, cap_workers: Optional[int] = None,
                    env_prefix: str = "", sample=None) -> Budget:
        """``plan()`` for an image corpus, sizing items from a pixel sample.

        ``*_bytes_per_px`` turn the sampled image area into the bytes one item
        holds (3.0 for a decoded RGB copy),
        ``*_fixed_bytes`` add per-item overheads (crop tensors, model inputs).
        The parent's average item uses the **median** area while the workers'
        buffer uses the **p90** one: the tail must not blow up a process, but the
        chunk should be sized for the typical image. Stages therefore never need
        to know corpus dimensions before asking for a budget.
        """
        sample = sample if sample is not None else sample_image_pixels(paths)
        median_px = sample.median_px * 1e6 or 1.5e6
        p90_px = sample.p90_px * 1e6 or median_px
        return self.plan(Workload(
            name=name, n_items=len(paths),
            item_bytes_parent=parent_bytes_per_px * median_px + parent_fixed_bytes,
            item_bytes_worker=worker_bytes_per_px * p90_px + worker_fixed_bytes,
            processes=processes, in_flight_per_worker=in_flight_per_worker,
            cap_workers=cap_workers, env_prefix=env_prefix))

    # --- reporting -------------------------------------------------------
    def describe(self, live: bool = True) -> str:
        """One-line ``cores | RAM | GPU`` summary for the startup banner."""
        st = self.live() if live else self.state
        cores = st.cpu.affinity or st.cpu.logical or 0
        load = f" (load {st.cpu.load1:.1f})" if st.cpu.load1 is not None else ""
        if st.ram.probed:
            ram = (f"RAM {st.ram.available_mb / 1024:.1f} GB free of "
                   f"{st.ram.total_mb / 1024:.0f} GB")
        else:
            ram = f"RAM {st.ram.total_mb / 1024:.0f} GB (free unknown)"
        gpu = st.gpu
        if getattr(gpu, "vendor", None):
            name = getattr(gpu, "name", None) or gpu.vendor
            free = getattr(gpu, "free_mb", None)
            total = getattr(gpu, "total_mb", None)
            if free is not None and total:
                vram = f" {free / 1024:.1f}/{total / 1024:.0f} GB"
            elif free is not None:
                vram = f" {free / 1024:.1f} GB free"
            else:
                vram = ""
            gpu_s = f"{name}{vram}"
        else:
            gpu_s = "no GPU"
        return f"{cores} cores{load} | {ram} | {gpu_s}"

    def dump(self) -> dict:
        """JSON-able snapshot of the machine state (``--machine``, bug reports)."""
        st = self.live()
        gpu = st.gpu
        return {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime(st.captured_at)),
            "host": st.host,
            "platform": st.platform,
            "python": st.python,
            "cpu": asdict(st.cpu),
            "ram": asdict(st.ram),
            "gpu": {key: getattr(gpu, key, None)
                    for key in ("vendor", "source", "name", "total_mb", "free_mb", "error")},
            "torch": st.torch_version,
            "cuda": st.cuda,
            "binaries": dict(st.binaries or {}),
        }


# --- process-wide instance (sanctioned; see the module docstring) ------------

_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def machine() -> MachineService:
    """The process-wide :class:`MachineService`, captured on first use."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = MachineService()
    return _INSTANCE


def set_machine(service) -> None:
    """Replace (or clear) the process-wide instance - tests and embedders."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = service


def reset_machine() -> None:
    """Drop the cached instance so the next :func:`machine` re-probes."""
    set_machine(None)
