"""Tests for deh.core.device: the GpuProbe dispatch table.

``GpuProbe.probe()`` resolves its probe chain by *string* name
(``_PROBES`` -> ``getattr``), so an entry that does not match a method name is
invisible to every static check and raises ``AttributeError`` at runtime. The
sysfs probe is the one the class advertises as the no-tools fallback ("the
first probe that answers wins"), and it is the one that was misnamed: it fires
exactly on a host with no usable CUDA torch build and no ``nvidia-smi``,
``rocm-smi`` or ``amd-smi``. These tests pin the table against the class and
exercise that fallback on a fictional ``/sys`` tree.
"""

import pytest

from deh.core import device
from deh.core.device import GpuMemory, GpuProbe

_MB = 1024 * 1024

#: The documented probe order (torch first, vendor tooling, sysfs, MPS).
PROBE_ORDER = ("_probe_torch", "_probe_nvidia_smi", "_probe_rocm_smi",
               "_probe_amd_smi", "_probe_sysfs_amdgpu", "_probe_xpu_smi",
               "_probe_mps")


def _probe_methods(cls):
    """Names of the ``_probe_*`` methods a probe class actually defines."""
    return {name for name, value in vars(cls).items()
            if name.startswith("_probe_") and callable(value)}


class TestProbeTable:
    """The string table and the class must not drift apart."""

    def test_every_table_entry_resolves_to_a_method(self):
        missing = [name for name in GpuProbe._PROBES
                   if not callable(getattr(GpuProbe, name, None))]
        assert missing == [], f"_PROBES names non-existent methods: {missing}"

    def test_no_probe_method_is_left_out_of_the_table(self):
        """``probe()`` walks ``_PROBES`` only: an unlisted probe is unreachable."""
        assert _probe_methods(GpuProbe) == set(GpuProbe._PROBES)

    def test_table_is_the_documented_order(self):
        assert tuple(GpuProbe._PROBES) == PROBE_ORDER


class TestFallbackWarningDedup:
    """One CPU-fallback alarm per pipeline phase, not per ``pick_device`` call.

    ``pick_device`` runs at every allocation point (per hash batch, per Hamming
    matrix, per extractor), so on a GPU held by an LLM server it used to print
    the red banner on *every* call. The alarm is a phase fact: ``begin_phase``
    re-arms it at each phase boundary. The device *decision* is never cached.
    """

    @pytest.fixture(autouse=True)
    def _fresh_phase(self):
        device.reset_device_warnings()
        device.begin_phase("test")
        yield
        device.reset_device_warnings()

    @staticmethod
    def _held_gpu(monkeypatch, free_mb: float = 100.0):
        """A CUDA-capable torch whose VRAM is almost entirely held by a process."""
        monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(device, "free_vram_mb", lambda: free_mb)
        monkeypatch.setattr(device, "gpu_hog_process", lambda: "strata")

    def test_one_banner_for_many_calls(self, monkeypatch, capsys):
        self._held_gpu(monkeypatch)
        picked = [device.pick_device() for _ in range(25)]
        assert {d.type for d in picked} == {"cpu"}
        out = capsys.readouterr().out
        assert out.count("GPU FALLBACK") == 1
        assert "[test]" in out
        assert "strata is holding it" in out

    def test_each_phase_alarms_once(self, monkeypatch, capsys):
        self._held_gpu(monkeypatch)
        device.pick_device()
        device.pick_device()
        device.begin_phase("hashing")
        assert device.current_phase() == "hashing"
        device.pick_device()
        out = capsys.readouterr().out
        assert out.count("GPU FALLBACK") == 2
        assert out.count("[hashing]") == 1

    def test_decision_is_live_only_the_alarm_is_deduped(self, monkeypatch):
        self._held_gpu(monkeypatch)
        assert device.pick_device().type == "cpu"
        monkeypatch.setattr(device, "free_vram_mb", lambda: 9000.0)
        assert device.pick_device().type == "cuda"

    def test_freed_gpu_is_picked_up_mid_phase_with_one_notice(self, monkeypatch, capsys):
        monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
        free = [100.0, 100.0, 8000.0, 8000.0]
        monkeypatch.setattr(device, "free_vram_mb", lambda: free.pop(0))
        monkeypatch.setattr(device, "gpu_hog_process", lambda: None)
        picked = [device.pick_device().type for _ in range(4)]
        assert picked == ["cpu", "cpu", "cuda", "cuda"]
        out = capsys.readouterr().out
        assert out.count("GPU FALLBACK") == 1
        assert out.count("GPU usable again") == 1

    def test_no_usable_gpu_is_a_separate_alarm(self, monkeypatch, capsys):
        monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
        assert {device.pick_device().type for _ in range(5)} == {"cpu"}
        out = capsys.readouterr().out
        assert out.count("no usable GPU") == 1

    def test_healthy_gpu_prints_nothing(self, monkeypatch, capsys):
        self._held_gpu(monkeypatch, free_mb=9000.0)
        for _ in range(3):
            device.begin_phase("clean")
            assert device.pick_device().type == "cuda"
        assert capsys.readouterr().out == ""


class TestProbeDispatch:

    def test_first_probe_that_answers_wins(self, monkeypatch):
        monkeypatch.setattr(GpuProbe, "_probe_torch", lambda self: None)
        monkeypatch.setattr(GpuProbe, "_probe_nvidia_smi", lambda self:
                            GpuMemory("nvidia", "nvidia-smi", "GeForce", 8192.0, 4096.0))
        info = GpuProbe().probe()
        assert (info.vendor, info.source) == ("nvidia", "nvidia-smi")
        assert info.free_mb == 4096.0

    def test_all_probes_silent_degrades_to_none(self, monkeypatch):
        """No GPU and no vendor tool is a degraded answer, not a crash."""
        for name in GpuProbe._PROBES:
            monkeypatch.setattr(GpuProbe, name, lambda self: None)
        assert GpuProbe().probe() == GpuMemory(vendor=None, source="none")

    def test_silent_chain_reports_no_error(self, monkeypatch):
        """'Nothing to say' and 'exploded' must stay distinguishable."""
        for name in GpuProbe._PROBES:
            monkeypatch.setattr(GpuProbe, name, lambda self: None)
        assert GpuProbe().probe().error is None

    def test_all_probes_raising_is_reported_not_hidden(self, monkeypatch):
        """A chain that exploded must not look like a machine with no GPU."""
        def boom(self):
            raise RuntimeError("nvidia-smi exploded")
        for name in GpuProbe._PROBES:
            monkeypatch.setattr(GpuProbe, name, boom)
        info = GpuProbe().probe()
        assert (info.vendor, info.source) == (None, "none")
        assert info.error is not None
        assert "torch" in info.error and "RuntimeError" in info.error
        assert "nvidia_smi" in info.error

    def test_raising_probe_does_not_stop_the_chain(self, monkeypatch):
        """One exploding probe is skipped; the next one that answers still wins."""
        def boom(self):
            raise RuntimeError("torch init exploded")
        monkeypatch.setattr(GpuProbe, "_probe_torch", boom)
        monkeypatch.setattr(GpuProbe, "_probe_nvidia_smi", lambda self:
                            GpuMemory("nvidia", "nvidia-smi", "GeForce", 8192.0, 4096.0))
        info = GpuProbe().probe()
        assert info.source == "nvidia-smi"
        assert info.error is None


class TestSysfsFallback:
    """The advertised no-tools fallback: amdgpu VRAM straight from sysfs."""

    @staticmethod
    def _card(base, total_bytes, used_bytes, name="card0", nested=True):
        """A fake ``/sys/class/drm`` tree (``device/`` layout and flat layout)."""
        leaf = base / name / "device" if nested else base / name
        leaf.mkdir(parents=True)
        (leaf / "mem_info_vram_total").write_text(str(total_bytes))
        (leaf / "mem_info_vram_used").write_text(str(used_bytes))
        return leaf

    def test_reports_amd_vram_from_device_layout(self, tmp_path):
        self._card(tmp_path, 8 * 1024 * _MB, 2 * 1024 * _MB)
        info = GpuProbe()._probe_sysfs_amdgpu(tmp_path)
        assert (info.vendor, info.source, info.name) == ("amd", "sysfs", "card0")
        assert info.total_mb == 8192.0
        assert info.free_mb == 6144.0

    def test_reports_amd_vram_from_flat_layout(self, tmp_path):
        self._card(tmp_path, 16 * 1024 * _MB, 4 * 1024 * _MB, nested=False)
        info = GpuProbe()._probe_sysfs_amdgpu(tmp_path)
        assert (info.vendor, info.name) == ("amd", "card0")
        assert info.free_mb == 12288.0

    def test_no_amdgpu_sysfs_entries_returns_none(self, tmp_path):
        assert GpuProbe()._probe_sysfs_amdgpu(tmp_path) is None

    def test_unreadable_vram_file_is_skipped(self, tmp_path):
        leaf = self._card(tmp_path, 8 * 1024 * _MB, 0)
        (leaf / "mem_info_vram_used").write_text("not-a-number")
        assert GpuProbe()._probe_sysfs_amdgpu(tmp_path) is None

    def test_sysfs_probe_is_reachable_from_probe(self, tmp_path, monkeypatch):
        """The regression this file exists for.

        Every vendor probe is silent (the host the sysfs probe exists for), so
        ``probe()`` must reach ``_probe_sysfs_amdgpu`` through the table and
        degrade to a real answer instead of raising ``AttributeError``.
        """
        self._card(tmp_path, 8 * 1024 * _MB, 2 * 1024 * _MB)
        for name in ("_probe_torch", "_probe_nvidia_smi", "_probe_rocm_smi",
                     "_probe_amd_smi", "_probe_xpu_smi", "_probe_mps"):
            monkeypatch.setattr(GpuProbe, name, lambda self: None)
        real = GpuProbe._probe_sysfs_amdgpu
        monkeypatch.setattr(GpuProbe, "_probe_sysfs_amdgpu",
                            lambda self, base=None: real(self, tmp_path))
        info = GpuProbe().probe()
        assert (info.vendor, info.source) == ("amd", "sysfs")


class TestFirstNotNone:
    """``_first_not_none`` is called as a bare name by ``_probe_xpu_smi``.

    It used to be a ``@staticmethod`` only, so the call resolved to a global
    that did not exist: any Intel host whose ``xpu-smi`` answers with JSON hit
    ``NameError`` inside the probe chain. The helper is module-level now, and
    it must return 0.0 - a JSON probe that legitimately reports zero used VRAM
    must not be skipped as if it were missing.
    """

    def test_module_level_helper_exists(self):
        from deh.core.device import _first_not_none
        assert _first_not_none(None, None, 3) == 3

    def test_zero_is_a_real_answer(self):
        from deh.core.device import _first_not_none
        assert _first_not_none(None, 0.0, 5.0) == 0.0

    def test_all_none(self):
        from deh.core.device import _first_not_none
        assert _first_not_none(None, None) is None

    def test_json_xpu_smi_output_does_not_raise(self, monkeypatch):
        import json
        import deh.core.device as dev
        # Intel's flat key shape: the probe searches for one key containing
        # both "memory" and "used" (see _find_key_float needles).
        payload = json.dumps({"device": [{"memory_used": 1024,
                                          "memory_total": 24576}]})
        monkeypatch.setattr(dev, "_run", lambda cmd, timeout=10.0: payload)
        info = GpuProbe()._probe_xpu_smi()
        assert (info.vendor, info.source) == ("intel", "xpu-smi")
        assert (info.total_mb, info.free_mb) == (24576.0, 23552.0)

    def test_zero_used_vram_is_reported_not_dropped(self, monkeypatch):
        import json
        import deh.core.device as dev
        payload = json.dumps({"device": [{"memory_used": 0,
                                          "memory_total": 8192}]})
        monkeypatch.setattr(dev, "_run", lambda cmd, timeout=10.0: payload)
        info = GpuProbe()._probe_xpu_smi()
        assert info.free_mb == 8192.0

    def test_plain_text_output_still_parses(self, monkeypatch):
        import deh.core.device as dev
        text = "xpu-smi stats -d 0\nmemory used 2048 MiB\nmemory total 8192 MiB\n"
        monkeypatch.setattr(dev, "_run", lambda cmd, timeout=10.0: text)
        info = GpuProbe()._probe_xpu_smi()
        assert (info.total_mb, info.free_mb) == (8192.0, 6144.0)

    def test_no_xpu_smi_returns_none(self, monkeypatch):
        import deh.core.device as dev
        monkeypatch.setattr(dev, "_run", lambda cmd, timeout=10.0: None)
        assert GpuProbe()._probe_xpu_smi() is None
