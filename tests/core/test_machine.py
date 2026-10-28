"""Tests for deh.core.machine: the machine-state service and its planner.

Every probe is injected, so the planning math is exercised on fictional
machines (an 8 GB box, a 2-core box, a box whose probes answer nothing)
without touching the real ``/proc`` or GPU.
"""

import json

from deh.core import machine as machine_mod
from deh.core.machine import Budget, CpuState, MachineService, Workload


def make_machine(available_mb=52000, total_mb=64000, cpus=32, env=None,
                 ttl_s=2.0):
    """A :class:`MachineService` on a fictional box (no /proc, GPU, binaries)."""
    return MachineService(
        probe_gpu=lambda: None,
        meminfo=lambda: (total_mb, available_mb, 0, 0),
        cpu=lambda: CpuState(logical=cpus, affinity=cpus, physical=None,
                             load1=None),
        binary=lambda name: None,
        env=env if env is not None else {},
        binaries=(),
        ttl_s=ttl_s)


def decode_workload(n=100, **kw):
    """The image-decode workload shape a duplicate stage uses (~26 MB/img buffers, 2 in flight)."""
    kw.setdefault("item_bytes_worker", 26e6)
    kw.setdefault("item_bytes_parent", 1e6)
    kw.setdefault("processes", True)
    kw.setdefault("in_flight_per_worker", 2)
    kw.setdefault("cap_workers", 16)
    kw.setdefault("env_prefix", "DECODE")
    return Workload(name="duplicate-decode", n_items=n, **kw)


class TestEagerCapture:

    def test_state_captured_eagerly(self):
        """``state`` is probed in ``__init__`` - no ``live()`` call needed."""
        svc = make_machine()
        assert svc.state.captured_at > 0
        assert svc.state.cpu.affinity == 32
        assert svc.state.ram.total_mb == 64000
        assert svc.state.ram.available_mb == 52000
        assert svc.state.ram.probed is True

    def test_live_ttl_caches_probe(self):
        """``live()`` reuses the reading within ``ttl_s``; refresh re-probes."""
        calls = []

        def meminfo():
            calls.append(1)
            return (64000, 52000, 0, 0)

        svc = MachineService(
            probe_gpu=lambda: None, meminfo=meminfo,
            cpu=lambda: CpuState(logical=32, affinity=32),
            binary=lambda name: None, env={}, binaries=(), ttl_s=60.0)
        svc.live()                                  # populates the TTL cache
        after_init = len(calls)
        svc.live()
        svc.live()
        assert len(calls) == after_init             # TTL cache: no re-probe
        svc.live(refresh=True)
        assert len(calls) == after_init + 1

    def test_dump_json_serializable(self):
        d = make_machine().dump()
        assert json.loads(json.dumps(d)) == d
        assert d["cpu"]["affinity"] == 32
        assert d["ram"]["available_mb"] == 52000
        assert isinstance(d["binaries"], dict)       # binaries=() -> no probing


class TestPlanRamCeiling:

    def test_low_ram_fewer_workers_never_zero(self):
        """An 8 GB box with 2 GB free: the reserve eats the budget, but the
        planner still returns ``min_workers`` - never a zero-worker pool."""
        svc = make_machine(available_mb=2048, total_mb=8192)
        plan = svc.plan(decode_workload())
        assert plan.jobs == 1
        assert plan.ram_budget_mb == 0

    def test_ram_budget_leaves_the_reserve(self):
        """budget = MemAvailable - max(25% of total, 4 GB floor)."""
        svc = make_machine(available_mb=52000, total_mb=64000)
        plan = svc.plan(decode_workload())
        assert plan.reserve_mb == 16000
        assert plan.ram_budget_mb == 52000 - 16000

    def test_max_ram_mb_caps_the_budget(self):
        svc = make_machine(env={"DEH_MAX_RAM_MB": "1000"})
        plan = svc.plan(decode_workload())
        assert plan.ram_budget_mb == 1000


class TestPlanCpuCeiling:

    def test_two_core_box(self):
        """2 cores - 2 reserved = 1 usable lane -> a single worker."""
        svc = make_machine(cpus=2)
        plan = svc.plan(decode_workload())
        assert plan.jobs == 1

    def test_stage_cap_workers(self):
        svc = make_machine(cpus=32)
        plan = svc.plan(decode_workload(cap_workers=4))
        assert plan.jobs == 4
        assert any("cap" in r for r in plan.reasons)


class TestPlanEnvOverrides:

    def test_override_clamped_to_ceilings_and_reported(self):
        """``DECODE_WORKERS=64`` on a box with no RAM budget is clamped
        down to the ceiling - and the clamp is printed as a reason."""
        svc = make_machine(available_mb=2048, total_mb=8192,
                           env={"DECODE_WORKERS": "64"})
        plan = svc.plan(decode_workload())
        assert plan.jobs == 1
        assert any("clamped" in r for r in plan.reasons)

    def test_override_honored_inside_ceilings(self):
        svc = make_machine(env={"DECODE_WORKERS": "4",
                                "DECODE_CHUNK": "12"})
        plan = svc.plan(decode_workload())
        assert plan.jobs == 4
        assert plan.chunk_items == 12
        assert any("DECODE_WORKERS=4" in r for r in plan.reasons)

    def test_malformed_override_ignored(self):
        svc = make_machine(env={"DECODE_WORKERS": "not-a-number"})
        plan = svc.plan(decode_workload())
        assert plan.jobs >= 1
        assert not any("DECODE_WORKERS" in r for r in plan.reasons)


class TestPlanUnknownProbes:

    def test_unknown_probes_conservative_never_zero(self):
        """No RAM probe, no cores: conservative fallbacks, still >= 1 worker."""
        svc = MachineService(
            probe_gpu=lambda: None, meminfo=lambda: None,
            cpu=lambda: CpuState(logical=0, affinity=0),
            binary=lambda name: None, env={}, binaries=())
        assert svc.state.ram.probed is False
        plan = svc.plan(decode_workload())
        assert 1 <= plan.jobs <= 4                  # FALLBACK_CPUS-bound
        assert "probe unavailable" in " ".join(plan.reasons)

    def test_broken_gpu_probe_never_aborts_capture(self):
        def boom():
            raise RuntimeError("nvidia-smi exploded")
        svc = MachineService(probe_gpu=boom,
                             meminfo=lambda: (64000, 52000, 0, 0),
                             cpu=lambda: CpuState(logical=4, affinity=4),
                             binary=lambda name: None, env={}, binaries=())
        assert svc.state.gpu is None


class TestPlanImages:

    def test_uses_corpus_sample(self, tmp_path):
        """``plan_images`` samples the corpus and sizes the decode pool."""
        from PIL import Image
        paths = []
        for i in range(4):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (160, 120)).save(p)
            paths.append(str(p))
        svc = make_machine()
        plan = svc.plan_images("test-decode", paths, parent_bytes_per_px=3.0,
                               in_flight_per_worker=4, cap_workers=8)
        assert isinstance(plan, Budget)
        assert plan.jobs == 8                       # CPU-capped
        assert plan.in_flight == plan.jobs * 4
        assert plan.chunk_items == 4                # bounded by the item count

    def test_huge_images_shrink_the_chunk(self, tmp_path):
        """12 MP images on a small budget: the parent-memory ceiling cuts the
        wave below the default ``2 x jobs`` chunk."""
        from PIL import Image
        paths = []
        for i in range(4):
            p = tmp_path / f"big_{i}.png"
            Image.new("RGB", (4000, 3000)).save(p)
            paths.append(str(p))
        svc = make_machine(available_mb=5000, total_mb=16000, cpus=8)
        plan = svc.plan_images("test-decode", paths, parent_bytes_per_px=3.0,
                               in_flight_per_worker=4, cap_workers=8)
        assert plan.jobs == 6                       # 8 cores - 2 reserved
        assert plan.chunk_items < 12                # default would be 2 x 6


class TestProcessWideAccessor:

    def test_set_and_reset_machine(self):
        try:
            sentinel = object()
            machine_mod.set_machine(sentinel)
            assert machine_mod.machine() is sentinel
        finally:
            machine_mod.reset_machine()
        assert machine_mod.machine() is not sentinel      # the cache was dropped
        assert isinstance(machine_mod.machine(), MachineService)

    def test_machine_returns_the_same_instance(self):
        assert machine_mod.machine() is machine_mod.machine()


class TestBudgetReporting:

    def test_describe_and_explain(self):
        plan = make_machine().plan(decode_workload())
        assert "workers" in plan.describe()
        assert "RAM budget" in plan.describe()
        assert plan.explain().startswith("    - ")
