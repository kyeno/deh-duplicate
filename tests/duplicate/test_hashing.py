"""Tests for deh.duplicate.hashing perceptual-hash grouping."""
import numpy as np
from PIL import Image

import pytest

from deh.core.device import resolve_device
from deh.core.hashes import HashEngine
from deh.duplicate.hashing import HashComparator


def _save(path, arr):
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path.resolve())


class TestHashGrouping:
    def test_identical_copies_are_grouped(self, tmp_path):
        rng = np.random.default_rng(42)
        base = rng.integers(0, 256, (128, 128, 3)).astype(np.uint8)
        p1 = _save(tmp_path / "dup1.png", base)
        p2 = _save(tmp_path / "dup2.png", base)          # identical pixel content
        other = _save(tmp_path / "other.png",
                      np.full((128, 128, 3), 200, dtype=np.uint8))

        groups = HashComparator(threshold=0.9).find_similar_groups([p1, p2, other])
        assert any({p1, p2}.issubset(g) for g in groups), \
            f"expected the duplicate pair to be grouped together; got {groups}"

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            HashComparator(threshold=1.5)


class TestComparisonAnnouncement:
    def test_announces_start_and_progress_milestones(self, tmp_path, capsys):
        rng = np.random.default_rng(7)
        paths = []
        for k in range(8):   # 8 images -> 28 pairwise comparisons; crosses 25/50/75%
            arr = rng.integers(0, 256, (32, 32, 3)).astype(np.uint8)
            p = tmp_path / f"img_{k}.png"
            Image.fromarray(arr, "RGB").save(str(p))
            paths.append(str(p.resolve()))

        HashComparator(threshold=0.9).find_similar_groups(paths)
        out = capsys.readouterr().out
        assert "[Hash] Comparing 8 images pairwise (28 comparisons)" in out
        assert "25% compared" in out
        assert "50% compared" in out
        assert "75% compared" in out


class TestOutputRouting:
    """Library output goes to the caller's callbacks, not to stdout (review 5.3)."""

    @staticmethod
    def _images(tmp_path, n=8, seed=7):
        rng = np.random.default_rng(seed)
        paths = []
        for k in range(n):   # 8 images -> 28 pairwise comparisons
            arr = rng.integers(0, 256, (32, 32, 3)).astype(np.uint8)
            p = tmp_path / f"img_{k}.png"
            Image.fromarray(arr, "RGB").save(str(p))
            paths.append(str(p.resolve()))
        return paths

    def test_log_cb_captures_every_line_and_stdout_stays_clean(self, tmp_path, capsys):
        """The module's own ``[Hash]`` prose follows ``log_cb``.

        Device alarms (``core.terminal.red_banner`` from ``pick_device``) stay
        on stdout by design: an embedder capturing the run's prose must not
        lose the "your GPU is unusable" shout with it.
        """
        lines = []
        comp = HashComparator(threshold=0.9, log_cb=lines.append)
        capsys.readouterr()                       # drop the construction line
        comp.find_similar_groups(self._images(tmp_path))

        out = capsys.readouterr().out
        leaked = [ln for ln in out.splitlines() if ln.startswith("[Hash]")]
        assert leaked == [], f"[Hash] output must go to log_cb, not stdout: {leaked}"
        assert any("Comparing 8 images pairwise" in ln for ln in lines)
        assert any("Found" in ln for ln in lines)

    def test_progress_cb_receives_structured_milestones(self, tmp_path):
        events = []
        comp = HashComparator(threshold=0.9, log_cb=lambda _text: None,
                              progress_cb=lambda done, total, phase: events.append(
                                  (done, total, phase)))
        comp.find_similar_groups(self._images(tmp_path))

        assert {"hash", "compare"} <= {phase for _, _, phase in events}
        assert all(done <= total for done, total, _ in events)
        assert max(done for done, _t, phase in events if phase == "compare") == 28

    def test_no_callbacks_keeps_today_terminal_channel(self, tmp_path, capsys, monkeypatch):
        """Default behaviour is unchanged: the CLI still sees every line."""
        monkeypatch.delenv("DEH_QUIET", raising=False)
        comp = HashComparator(threshold=0.9)
        capsys.readouterr()
        comp.find_similar_groups(self._images(tmp_path))
        assert "[Hash] Comparing 8 images pairwise" in capsys.readouterr().out


class TestHashEngine:
    def test_torch_engine_matches_imagehash(self, tmp_path):
        """The shared engine must stay bit-identical to imagehash (reference)."""
        import imagehash

        rng = np.random.default_rng(11)
        paths = []
        for k in range(4):
            arr = rng.integers(0, 256, (160, 120, 3)).astype(np.uint8)
            p = tmp_path / f"img{k}.png"
            Image.fromarray(arr, "RGB").save(str(p))
            paths.append(str(p.resolve()))

        got = HashEngine().compute_batch(paths, resolve_device())
        for p in paths:
            img = Image.open(p).convert("RGB")
            assert got[p]["phash"] == str(imagehash.phash(img))
            assert got[p]["dhash"] == str(imagehash.dhash(img))

    def test_parallel_decode_matches_serial(self, tmp_path):
        rng = np.random.default_rng(13)
        paths = []
        for k in range(6):
            arr = rng.integers(0, 256, (96, 96, 3)).astype(np.uint8)
            p = tmp_path / f"p{k}.png"
            Image.fromarray(arr, "RGB").save(str(p))
            paths.append(str(p.resolve()))

        engine = HashEngine()
        device = resolve_device()
        serial = engine.compute_batch(paths, device, decode_workers=1)
        parallel = engine.compute_batch(paths, device, decode_workers=4)
        assert serial == parallel

    def test_unreadable_image_is_skipped(self, tmp_path):
        rng = np.random.default_rng(17)
        good = tmp_path / "good.png"
        Image.fromarray(rng.integers(0, 256, (64, 64, 3)).astype(np.uint8),
                        "RGB").save(str(good))
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")

        got = HashEngine().compute_batch([str(good), str(bad)], resolve_device())
        assert str(good) in got and str(bad) not in got
        # An all-unreadable input is empty output, not an exception.
        assert HashEngine().compute_batch([str(bad)], resolve_device()) == {}


class TestDevicePaths:
    def test_cpu_and_auto_device_group_identically(self, tmp_path):
        """CPU and auto-picked (GPU) comparisons must produce identical groups."""
        rng = np.random.default_rng(23)
        base = rng.integers(0, 256, (96, 96, 3)).astype(np.uint8)
        paths = []
        for k in range(6):
            frame = base if k < 2 else rng.integers(0, 256, (96, 96, 3)).astype(np.uint8)
            p = tmp_path / f"img{k}.png"
            Image.fromarray(frame, "RGB").save(str(p))
            paths.append(str(p.resolve()))

        cpu = HashComparator(threshold=0.9, device="cpu").find_similar_groups(paths)
        auto = HashComparator(threshold=0.9).find_similar_groups(paths)
        assert sorted(map(sorted, cpu)) == sorted(map(sorted, auto))
