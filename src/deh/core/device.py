# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Device resolution — the single place for CUDA/GPU decisions.

Two distinct concepts, kept separate on purpose:

- ``resolve_device()`` — the *stable* preferred device: CUDA if
  ``torch.cuda.is_available()``, else CPU. No VRAM check. Used only for the
  *engine tag* stored in cached payloads, so a transient CPU fallback never
  invalidates the cache.
- ``pick_device(min_free_mb)`` — the *dynamic* device for an actual
  allocation/processing step: CUDA if available **and** free VRAM is at least
  ``min_free_mb`` (default ``GPU_STAGE_MIN_FREE_MB``), else CPU with a big red
  warning. Call it at every allocation/processing point so the decision tracks
  the live GPU state, not just the state at init.

The *warning* is a phase-level fact, not a per-call one: ``pick_device`` runs
per hash batch, per distance matrix, per extractor, so on a GPU held by an LLM
server it used to scream once per batch. Pipelines call ``begin_phase()`` when
a phase starts, and the red banner is then printed once per phase per reason.
The device decision itself stays live — every call still probes the GPU, so
freeing the VRAM mid-phase is picked up on the next call.

The GPU is a shared resource: LLM servers (llama-server, vLLM, ...) and other
GPU users regularly hold most of it, so a stage that cannot fit falls back to
CPU deliberately (with a loud warning) instead of triggering a CUBLAS/OOM
failure mid-run. Cheap signals that benefit from the GPU: batched pHash/dHash
(torch DCT) and Hamming distance matrices. Everything this project does with
metadata (``core.machine`` version probes) is CPU-only.
"""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from .terminal import notice, red_banner

# Minimum free VRAM (MB) the GPU-local torch stages need to run on the GPU.
# Below this, a stage falls back to CPU deliberately (the shared GPU is usually
# held by another process).
GPU_STAGE_MIN_FREE_MB = 2048

_MB = 1024 * 1024


@dataclass(frozen=True)
class GpuMemory:
    """Device memory snapshot from the first probe that could answer.

    ``vendor``: 'nvidia' | 'amd' | 'intel' | 'apple' | None.
    ``source``: probe name (``torch``, ``nvidia-smi``, ``rocm-smi``,
        ``amd-smi``, ``sysfs``, ``xpu-smi``, ``mps``, ``none``).
    ``free_mb``: ``None`` when the device is known but free memory is not
        reportable (Apple MPS, some Intel setups) -- an unknown value never
        blocks a stage.
    ``error``: set when *no* probe could answer and at least one of them
        raised. A machine with no GPU and a machine whose probes all exploded
        are different facts, and the planning code must be able to tell them
        apart (``--machine`` shows it).
    """

    vendor: str | None
    source: str
    name: str | None = None
    total_mb: float | None = None
    free_mb: float | None = None
    error: str | None = None


def _run(cmd, timeout: float = 10.0) -> str | None:
    """Run a probe command; returns stdout text or None when unavailable."""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def _first_float(text: str, pattern: str) -> float | None:
    """First float captured by *pattern* in *text* (None when absent)."""
    m = re.search(pattern, text or "", re.IGNORECASE)
    return float(m.group(1)) if m else None


def _find_key_float(obj, *needles) -> float | None:
    """Recursively find a numeric value whose JSON key contains all *needles*."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            kl = str(key).lower()
            if all(n in kl for n in needles) and isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, (dict, list)):
                got = _find_key_float(value, *needles)
                if got is not None:
                    return got
    elif isinstance(obj, list):
        for value in obj:
            got = _find_key_float(value, *needles)
            if got is not None:
                return got
    return None


def _first_not_none(*values):
    """First non-None argument (JSON probes get 0.0 legitimately)."""
    for value in values:
        if value is not None:
            return value
    return None


class GpuProbe:
    """Probe the GPU vendor and memory via the first tool that can answer.

    The seven vendor probes are tried in order (``_PROBES``); the first
    that returns a ``GpuMemory`` wins. Grouped in a class because the probe
    chain is a genuine unit of encapsulation (a dispatch table plus the
    orchestration), not a bag of unrelated free functions.
    """

    #: Probe order: native torch first (most accurate), then vendor tooling and
    #: Linux sysfs, then Apple MPS. The first probe that answers wins.
    _PROBES = (
        "_probe_torch",
        "_probe_nvidia_smi",
        "_probe_rocm_smi",
        "_probe_amd_smi",
        "_probe_sysfs_amdgpu",
        "_probe_xpu_smi",
        "_probe_mps",
    )

    def probe(self) -> GpuMemory:
        """First probe that can answer wins; ``GpuMemory(None, "none")`` when none can.

        A probe that *raised* is not the same as a probe with nothing to say: the
        names and messages are collected and reported in the ``error`` field of
        the fallback result, so a broken probe chain is visible in ``--machine``
        instead of looking like a machine without a GPU.
        """
        failures = []
        for name in self._PROBES:
            try:
                info = getattr(self, name)()
            except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
                failures.append(f"{name[7:]}: {type(exc).__name__}: {exc}")
                continue
            if info is not None:
                return info
        return GpuMemory(vendor=None, source="none",
                         error="; ".join(failures) if failures else None)


    def _probe_torch(self) -> GpuMemory | None:
        """Native torch probe: CUDA (NVIDIA or ROCm build) then Intel XPU.

        Most accurate and always tried first; needs no external tool. Returns None
        when torch sees no usable device, so CLI/sysfs fallbacks can take over.
        """
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vendor = "amd" if getattr(torch.version, "hip", None) else "nvidia"
                return GpuMemory(vendor, "torch", torch.cuda.get_device_name(),
                                 total / _MB, free / _MB)
        except (RuntimeError, OSError):        # CUDA/ROCm initialised badly
            pass
        try:
            xpu = getattr(torch, "xpu", None)
            if (xpu is not None and hasattr(xpu, "is_available")
                    and xpu.is_available() and hasattr(xpu, "mem_get_info")):
                free, total = xpu.mem_get_info()
                name = (xpu.get_device_name()
                        if hasattr(xpu, "get_device_name") else None)
                return GpuMemory("intel", "torch", name, total / _MB, free / _MB)
        except (RuntimeError, OSError):        # XPU initialised badly
            pass
        return None


    def _probe_nvidia_smi(self) -> GpuMemory | None:
        """NVIDIA via ``nvidia-smi`` (picks the GPU with the most free memory)."""
        text = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                     "--format=csv,noheader,nounits"])
        if not text:
            return None
        best = None
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                total, free = float(parts[-2]), float(parts[-1])
            except ValueError:
                continue
            info = GpuMemory("nvidia", "nvidia-smi", ",".join(parts[:-2]) or None,
                             total, free)
            if best is None or (info.free_mb or 0) > (best.free_mb or 0):
                best = info
        return best


    def _probe_rocm_smi(self) -> GpuMemory | None:
        """AMD via ``rocm-smi`` (CSV first, then the legacy human format)."""
        text = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        if text:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if lines:
                header = [c.strip().lower() for c in lines[0].split(",")]

                def col(*needles):
                    for i, h in enumerate(header):
                        if all(n in h for n in needles):
                            return i
                    return None

                i_total, i_used = col("vram", "total"), col("vram", "used")
                if i_total is not None and i_used is not None:
                    best = None
                    for line in lines[1:]:
                        parts = [p.strip() for p in line.split(",")]
                        try:
                            total, used = float(parts[i_total]), float(parts[i_used])
                        except (IndexError, ValueError):
                            continue
                        info = GpuMemory("amd", "rocm-smi", parts[0] or None,
                                         total / _MB, max(0.0, total - used) / _MB)
                        if best is None or (info.free_mb or 0) > (best.free_mb or 0):
                            best = info
                    if best is not None:
                        return best
        text = _run(["rocm-smi"])
        total = _first_float(text, r"VRAM Total Memory \(B\):\s*([\d.]+)")
        if total is not None:
            used = _first_float(text, r"VRAM Total Used Memory \(B\):\s*([\d.]+)")
            return GpuMemory("amd", "rocm-smi", None, total / _MB,
                             max(0.0, total - (used or 0.0)) / _MB)
        return None


    def _probe_amd_smi(self) -> GpuMemory | None:
        """AMD via the newer ``amd-smi`` (JSON when available, regex fallback)."""
        text = _run(["amd-smi", "static", "--vram", "--json"])
        if not text:
            text = _run(["amd-smi", "static", "--vram"])
        if not text:
            return None
        size = _first_float(text, r'"size"\s*:\s*([\d.]+)')
        used = _first_float(text, r'"used"\s*:\s*([\d.]+)')
        if size is not None:
            # amd-smi reports bytes when the numbers are huge, MB otherwise
            total_mb = size / _MB if size > (1 << 20) else size
            used_mb = (used or 0.0) / _MB if (used or 0.0) > (1 << 20) else \
                (used or 0.0)
            return GpuMemory("amd", "amd-smi", None, total_mb,
                             max(0.0, total_mb - used_mb))
        m = re.search(r"VRAM[:\s]+([\d.]+)\s*/\s*([\d.]+)\s*(GB|GiB)", text, re.I)
        if m:  # "<used> / <total> GB"
            return GpuMemory("amd", "amd-smi", None, float(m.group(2)) * 1024,
                             float(m.group(1)) * 1024)
        return None

    def _probe_sysfs_amdgpu(self, base: Path | None = None) -> GpuMemory | None:
        """AMD via Linux sysfs (amdgpu ``mem_info_vram_*``; works with no tools)."""
        base = base if base is not None else Path("/sys/class/drm")
        total_files = sorted(base.glob("card*/device/mem_info_vram_total"))
        if not total_files:
            total_files = sorted(base.glob("card*/mem_info_vram_total"))
        for total_f in total_files:
            try:
                total = int(total_f.read_text().strip())
                used = int(total_f.with_name("mem_info_vram_used")
                           .read_text().strip())
            except (OSError, ValueError):
                continue
            card = (total_f.parent if total_f.parent.name.startswith("card")
                    else total_f.parents[1])
            return GpuMemory("amd", "sysfs", card.name or None,
                             total / _MB, max(0.0, total - used) / _MB)
        return None


    def _probe_xpu_smi(self) -> GpuMemory | None:
        """Intel via ``xpu-smi`` (JSON when available, plain-text fallback)."""
        text = _run(["xpu-smi", "stats", "-d", "0", "--json"])
        if not text:
            text = _run(["xpu-smi", "stats", "-d", "0"])
        if not text:
            return None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            data = None
        used = total = None
        if isinstance(data, (dict, list)):
            used = _first_not_none(_find_key_float(data, "memory", "used"),
                                   _find_key_float(data, "mem", "used"))
            total = _first_not_none(_find_key_float(data, "memory", "total"),
                                    _find_key_float(data, "mem", "total"))
        if used is None:
            used = _first_float(text, r"memory[ _]used[^0-9]*([\d.]+)")
        if total is None:
            total = _first_float(text, r"memory[ _]total[^0-9]*([\d.]+)")
        if total is None:
            return None
        free = None if used is None else max(0.0, total - used)
        # xpu-smi reports MiB
        return GpuMemory("intel", "xpu-smi", None, total, free)


    def _probe_mps(self) -> GpuMemory | None:
        """Apple Silicon/MPS: the device is known, free memory is not reportable."""
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return GpuMemory("apple", "mps", "Apple MPS")
        except (RuntimeError, OSError):        # MPS backend initialised badly
            pass
        return None


def gpu_memory() -> GpuMemory:
    """First probe that can answer wins; ``GpuMemory(None, "none")`` when none can."""
    return GpuProbe().probe()


def free_vram_mb() -> float | None:
    """Free VRAM in MB, or None when the GPU cannot report it."""
    return GpuProbe().probe().free_mb


def _amd_hog_name(text: str) -> str | None:
    """Best-effort top process name from ``rocm-smi --showpids`` text.

    Tolerant of column-order differences across rocm-smi versions: a row is
    treated as a pid row when it starts with a bare integer; the first
    non-numeric token becomes the name and the largest trailing number its
    used memory. Returns ``None`` when no pid row is found.
    """
    best_name, best_mb = None, -1.0
    for line in text.splitlines():
        toks = line.split()
        if not toks or not toks[0].isdigit():
            continue
        name = None
        mb = 0.0
        for tok in toks[1:]:
            if tok.replace(".", "", 1).isdigit():
                mb = float(tok)
            elif name is None:
                name = tok
        if name is not None and mb > best_mb:
            best_name, best_mb = name, mb
    return best_name


def gpu_hog_process() -> str | None:
    """Name of the process using the most device memory (best-effort).

    Used in VRAM warnings to tell the user *what* to stop. NVIDIA via
    ``nvidia-smi --query-compute-apps``; AMD via ``rocm-smi --showpids``
    (column-tolerant). Returns ``None`` when no tool answers or no compute
    process is listed -- the VRAM figure alone is always enough to warn.
    """
    text = _run(["nvidia-smi", "--query-compute-apps=process_name,used_memory",
                 "--format=csv,noheader,nounits"])
    if text:
        rows = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                rows.append((int(parts[1]), parts[0]))
        if rows:
            rows.sort(reverse=True)
            return rows[0][1]
    text = _run(["rocm-smi", "--showpids"])
    if text:
        name = _amd_hog_name(text)
        if name is not None:
            return name
    return None


# --------------------------------------------------------------------------- #
# Fallback warnings: one alarm per pipeline phase                             #
# --------------------------------------------------------------------------- #

#: Label of the pipeline phase the current ``pick_device`` call belongs to.
_phase: str = "run"

#: Fallback reasons already announced in the current phase (``"low-vram"``,
#: ``"no-gpu"``). The device *decision* is never cached here — only the alarm.
_announced: set = set()

#: True while the current phase is in a CPU fallback (drives the recovery line).
_fell_back: bool = False


def begin_phase(name: str) -> None:
    """Start a pipeline phase: the next CPU fallback is announced again.

    :func:`pick_device` is called at every allocation point — per hash batch,
    per Hamming-distance matrix, per feature extractor — so on a GPU held by an
    LLM server it printed the red fallback banner on *every* call. The fallback
    is a fact about the phase, not about the call: pipelines call this at each
    phase boundary and the banner appears once per phase per reason.

    Freeing the GPU mid-phase is picked up by the next call (with a single
    recovery notice), so this never freezes a device decision.
    """
    global _phase, _fell_back
    _phase = str(name)
    _announced.clear()
    _fell_back = False


def current_phase() -> str:
    """The phase label last set by :func:`begin_phase` (``"run"`` before any)."""
    return _phase


def reset_device_warnings() -> None:
    """Forget the announced fallbacks (test helper; phases do this themselves)."""
    global _fell_back
    _announced.clear()
    _fell_back = False


def _announce_fallback(free_mb: float | None, min_free_mb: float) -> None:
    """Print the CPU-fallback banner at most once per phase (per reason)."""
    global _fell_back
    reason = "no-gpu" if free_mb is None else "low-vram"
    if reason in _announced:
        return
    _announced.add(reason)
    _fell_back = True
    if free_mb is None:
        red_banner([
            f"GPU FALLBACK [{_phase}] - no usable GPU - running on CPU.",
            "This will be slower; a GPU is recommended.",
        ])
        return
    hog = gpu_hog_process()
    hog_s = f" ({hog} is holding it)" if hog else ""
    red_banner([
        f"GPU FALLBACK [{_phase}] - only {free_mb / 1024:.1f} GB free "
        f"(need ~{min_free_mb / 1024:.0f} GB){hog_s} - running on CPU.",
        "Free the GPU and re-run for GPU speed.",
    ])


def _announce_recovery(free_mb: float | None) -> None:
    """One line when the GPU becomes usable again inside the same phase."""
    global _fell_back
    if not _fell_back:
        return
    _fell_back = False
    free_s = f"{free_mb / 1024:.1f} GB free" if free_mb is not None else "VRAM n/a"
    notice(f"[Device] GPU usable again ({free_s}) - resuming on cuda "
           f"(phase {_phase}).")


def resolve_device() -> torch.device:
    """The *stable* preferred device: CUDA if available, else CPU.

    No VRAM check — the engine tag is a property of the environment, not the
    transient free-VRAM state, so a CPU fallback never invalidates a cached
    payload. Use :func:`pick_device` for the actual allocation/processing
    device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_device(min_free_mb: float | None = None) -> torch.device:
    """The *dynamic* device for an allocation/processing step.

    Always prefers CUDA. Falls back to CPU when the GPU is unavailable or has
    less than ``min_free_mb`` free VRAM (default ``GPU_STAGE_MIN_FREE_MB``).
    Call this at every allocation/processing point so the decision tracks the
    live GPU state, not just the state at init.

    The red warning is deliberately not printed on every fallback: it is
    announced once per pipeline phase (see :func:`begin_phase`), so a run whose
    GPU is held by an LLM server says it once per phase, not once per batch.
    """
    if min_free_mb is None:
        min_free_mb = GPU_STAGE_MIN_FREE_MB
    if torch.cuda.is_available():
        free = free_vram_mb()
        if free is not None and free < min_free_mb:
            _announce_fallback(free, min_free_mb)
            return torch.device("cpu")
        _announce_recovery(free)
        return torch.device("cuda")
    _announce_fallback(None, min_free_mb)
    return torch.device("cpu")


def banner(device: torch.device) -> str:
    """One-line human-readable device status for pipeline banners."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        free = free_vram_mb()
        free_s = f"{free / 1024:.1f} GB free" if free is not None else "VRAM n/a"
        return f"GPU: {name} ({free_s}) | hashing+distances on cuda"
    return "GPU: none | hashing+distances on cpu"
