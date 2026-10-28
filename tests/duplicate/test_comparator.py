"""Regression tests for the duplicate-quality fixes.

These lock in two silent failures that shipped in deh-duplicate:

1. ``list(model.children())[:-1]`` stopped including MobileNetV2's pooling
   once torchvision folded avgpool into ``classifier``, so the extractor
   emitted an unpooled ``(1280, 7, 7)`` map instead of a 1280-d vector.
2. That 3-D map then made the comparator's row-block matmul
   ``F[start:end] @ F.mT`` broadcast into a wrong-shaped result, so the CUDA
   path returned ZERO groups while looking perfectly healthy.


Hybrid mode compares every extracted feature pair, then verifies deep groups
with perceptual hashes before reporting them.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from deh.duplicate.comparator import DeepSimilarityComparator
from deh.duplicate.hashing import HashComparator
from deh.duplicate.pipeline import DuplicateConfig, DuplicatePipeline


def _unit(v):
    """L2-normalize a vector and return it as float32."""
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_library_config_uses_mode_specific_threshold_defaults():
    assert DuplicateConfig(directory="/images", mode="hash").threshold == 0.75
    assert DuplicateConfig(directory="/images", mode="hybrid").threshold == 0.75
    assert DuplicateConfig(directory="/images", mode="deep").threshold == 0.82
    assert DuplicateConfig(directory="/images", mode="deep", threshold=0.75).threshold == 0.75
    assert DeepSimilarityComparator().threshold == 0.82


class TestFeatureExtractorPooling:
    """The extractor's advertised ``feature_dim`` must match reality."""

    def test_extract_returns_1d_1280_vector(self, tmp_path):
        from PIL import Image
        from deh.duplicate.features import FeatureExtractor

        p = tmp_path / "a.png"
        Image.new("RGB", (300, 260), (120, 30, 200)).save(p)

        ex = FeatureExtractor()
        feat = ex.extract(p)
        arr = feat.cpu().numpy() if isinstance(feat, torch.Tensor) else np.asarray(feat)

        assert arr.ndim == 1, (
            "feature must be a single pooled vector, got shape "
            f"{arr.shape} - the model trunk is missing its pooling layer"
        )
        assert arr.shape == (ex.feature_dim,), (
            f"expected {ex.feature_dim} dims, got {arr.shape}"
        )

    def test_model_output_is_pooled(self):
        """Guard the trunk itself, independent of any input image."""
        from deh.duplicate.features import FeatureExtractor

        ex = FeatureExtractor()
        with torch.no_grad():
            out = ex.model(torch.zeros(2, 3, 224, 224, device=ex.device))
        assert out.ndim == 2 and out.shape[1] == ex.feature_dim, (
            f"model trunk must emit [B, {ex.feature_dim}], got {tuple(out.shape)}"
        )

    def test_extract_batch_is_1d(self, tmp_path):
        from PIL import Image
        from deh.duplicate.features import FeatureExtractor

        paths = []
        for i, color in enumerate([(200, 10, 10), (10, 200, 10), (10, 10, 200)]):
            p = tmp_path / f"{i}.png"
            Image.new("RGB", (240, 200), color).save(p)
            paths.append(str(p))

        feats = FeatureExtractor().extract_batch(paths, batch_size=2)
        assert len(feats) == 3
        for path, feat in feats.items():
            arr = feat.cpu().numpy() if isinstance(feat, torch.Tensor) else np.asarray(feat)
            assert arr.ndim == 1, f"{path}: expected 1-D feature, got {arr.shape}"


class TestComparatorShapeGuard:
    def test_rejects_3d_feature_matrix(self):
        """An unpooled map must raise loudly, not silently find nothing."""
        bad = {f"/i/{i}.png": np.random.rand(1280, 7, 7).astype(np.float32)
               for i in range(3)}
        with pytest.raises(ValueError, match="2-D"):
            DeepSimilarityComparator(threshold=0.75).find_similar_groups(bad)

    def test_assert_2d_checks_row_count(self):
        with pytest.raises(ValueError, match="rows"):
            DeepSimilarityComparator._assert_2d(np.zeros((2, 1280)), 5)

    def test_reports_pairwise_progress(self):
        features = {str(i): _unit([1.0, i * 0.01, 0.0]) for i in range(5)}
        updates = []
        DeepSimilarityComparator(progress_cb=lambda done, total: updates.append(
            (done, total))).find_similar_groups(features)
        assert updates[0] == (0, 10)
        assert updates[-1] == (10, 10)
        assert all(earlier <= later for (earlier, _), (later, _) in
                   zip(updates, updates[1:]))


class TestComparatorGpuCpuParity:
    """The CUDA path must agree with the numpy path on identical features."""

    @pytest.fixture
    def features(self):
        rng = np.random.default_rng(1234)
        clusters = []
        # three tight clusters -> a correct comparator finds exactly 3 groups
        for _ in range(3):
            centre = _unit(rng.normal(size=64))
            for _ in range(4):
                clusters.append(_unit(centre + rng.normal(scale=0.01, size=64)))
        for _ in range(3):   # unrelated singletons
            clusters.append(_unit(rng.normal(size=64)))
        return {f"/img/{i:02d}.jpg": f for i, f in enumerate(clusters)}

    def test_cuda_path_is_not_silently_empty(self, features):
        """Regression test for the exact symptom: 0 groups on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        cuda_feats = {k: torch.from_numpy(np.ascontiguousarray(v)).cuda()
                      for k, v in features.items()}
        groups = DeepSimilarityComparator(threshold=0.9).find_similar_groups(cuda_feats)
        assert groups, "CUDA comparator returned zero groups on a clustered corpus"

    def test_numpy_and_cuda_agree(self, features):
        cpu = DeepSimilarityComparator(threshold=0.9).find_similar_groups(features)
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cuda_feats = {k: torch.from_numpy(np.ascontiguousarray(v)).cuda()
                      for k, v in features.items()}
        gpu = DeepSimilarityComparator(threshold=0.9).find_similar_groups(cuda_feats)

        assert {frozenset(g) for g in gpu} == {frozenset(g) for g in cpu}, (
            "CUDA and numpy comparison diverged; the unpooled-feature regression "
            "silently zeroes out the GPU path"
        )

        assert len(gpu) == 3, f"expected the 3 planted clusters, got {len(gpu)} groups"


# --------------------------------------------------------------------- #
# Hybrid mode: deep candidates must also pass perceptual hashes
# --------------------------------------------------------------------- #

class _StubHash:
    def __init__(self, groups):
        self._groups = groups

    def find_similar_groups(self, pool):
        available = set(pool)
        return [[p for p in group if p in available]
                for group in self._groups
                if len(available.intersection(group)) > 1]


class _StubExtractor:
    """Returns canned features keyed by path."""

    def __init__(self, features):
        self._features = features

    def extract_batch(self, paths, **kwargs):
        results = {p: self._features[p] for p in paths if p in self._features}
        if kwargs.get("progress_cb") is not None:
            kwargs["progress_cb"](len(paths), len(paths))
        return results


class TestHybridDeepVerification:
    def _pipeline(self, root, hash_groups, features):
        lines = []

        class FakeScanner:
            def __init__(self, directory): pass
            def scan(self): return list(features.keys())

        class FakeResolver:
            def __init__(self, directory, log_cb=None): pass
            def resolve(self, image_paths): return (list(image_paths), [])

        cfg = DuplicateConfig(
            directory=str(root), mode="hybrid", use_gui=False,
            identical_dedupe=True, action="move", dry_run=True,
            threshold=0.75, deep_threshold=0.90,
        )
        pipe = DuplicatePipeline(
            cfg, scanner_cls=FakeScanner, resolver_cls=FakeResolver,
            hash_cls=lambda threshold=None, log_cb=None: _StubHash(hash_groups),
            extractor_cls=lambda log_cb=None: _StubExtractor(features),
            out=lines.append,
        )
        return pipe, lines

    def test_unrelated_image_is_excluded_by_deep_threshold(self, tmp_path):
        features = {
            str(tmp_path / "a.jpg"): _unit([1.0] + [0.0] * 63),
            str(tmp_path / "b.jpg"): _unit([1.0, 0.01] + [0.0] * 62),
            str(tmp_path / "c.jpg"): _unit([0.0] * 63 + [1.0]),
        }
        pipe, lines = self._pipeline(tmp_path, [list(features)], features)
        groups = pipe._find_similar_groups(list(features))

        assert len(groups) == 1, f"expected one surviving group, got {groups}"
        assert {Path(g).name for g in groups[0]} == {"a.jpg", "b.jpg"}, (
            f"stray 'c.jpg' survived deep verification: {groups}"
        )
        assert any("Pairwise similarity comparison" in line for line in lines)

    def test_all_images_are_deep_scanned_before_hash_verification(self, tmp_path):
        features = {
            str(tmp_path / "x.jpg"): _unit([1.0] + [0.0] * 63),
            str(tmp_path / "y.jpg"): _unit([1.0, 0.01] + [0.0] * 62),
            str(tmp_path / "z.jpg"): _unit([0.0] * 63 + [1.0]),
        }
        pipe, _ = self._pipeline(tmp_path, [list(features)[:2]], features)
        groups = pipe._find_similar_groups(list(features))

        assert len(groups) == 1
        assert {Path(g).name for g in groups[0]} == {"x.jpg", "y.jpg"}

    def test_deep_match_without_hash_match_is_rejected(self, tmp_path):
        """Semantic similarity alone must not send an image to quarantine."""
        a, b, c = (str(tmp_path / f"{n}.jpg") for n in "abc")
        features = {
            a: _unit([1.0, 0.0, 0.0]),
            b: _unit([0.0, 1.0, 0.0]),
            c: _unit([0.0, 1.0, 0.01]),
        }
        pipe, _ = self._pipeline(tmp_path, [[a, b]], features)
        groups = pipe._find_similar_groups([a, b, c])
        assert groups == []

    def test_default_hybrid_rejects_large_semantic_false_group(self, tmp_path):
        """A shared deep embedding must not turn unrelated images into losers."""
        from PIL import Image

        rng = np.random.default_rng(619)
        base = rng.integers(16, 240, (128, 128, 3), dtype=np.uint8)
        images = [base, np.clip(base.astype(np.int16) + 2, 0, 255).astype(np.uint8)]
        images.extend(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
                      for _ in range(14))
        paths = []
        for i, pixels in enumerate(images):
            path = tmp_path / f"image_{i:02}.png"
            Image.fromarray(pixels).save(path)
            paths.append(str(path))

        # Simulate the failure mode: MobileNet rates all 16 images as the
        # same content, so deep comparison alone produces one large group.
        features = {path: _unit([1.0, 0.0, 0.0]) for path in paths}
        config = DuplicateConfig(directory=str(tmp_path), identical_dedupe=False)
        assert config.mode == "hybrid"
        updates = []
        pipe = DuplicatePipeline(
            config,
            extractor_cls=lambda log_cb=None: _StubExtractor(features),
            hash_cls=lambda threshold=None, log_cb=None, progress_cb=None: HashComparator(
                threshold=threshold, log_cb=log_cb, progress_cb=progress_cb,
                device=torch.device("cpu")
            ),
            out=lambda *_: None,
            progress_cb=lambda stage, done, total: updates.append((stage, done, total)),
        )

        groups = pipe._find_similar_groups(paths)
        assert {frozenset(group) for group in groups} == {
            frozenset(paths[:2])
        }
        for stage in ("features", "deep_pairs", "hash_images", "hash_pairs"):
            assert any(current == stage and done == total and total > 0
                       for current, done, total in updates), stage

    def test_no_image_lands_in_two_groups(self, tmp_path):
        names = "abcdef"
        features = {
            str(tmp_path / f"{n}.jpg"): _unit(
                [1.0] * 64 if n in "ab" else
                [0.0] * 32 + [1.0] * 32 if n in "cd" else
                [i / 64 for i in range(64)]
            )
            for i, n in enumerate(names)
        }
        pool = [str(tmp_path / f"{n}.jpg") for n in names]
        pipe, _ = self._pipeline(tmp_path, [pool[:2], pool[2:4]], features)
        groups = pipe._find_similar_groups(pool)

        flat = [g for group in groups for g in group]
        assert len(flat) == len(set(flat)), "an image landed in more than one group"

    @pytest.mark.parametrize("available", [0, 1])
    def test_missing_features_abort_instead_of_using_hash_groups(self, tmp_path, available):
        paths = [str(tmp_path / f"{n}.jpg") for n in "abc"]
        features = {p: _unit([1, 0, 0]) for p in paths[:available]}
        pipe, _ = self._pipeline(tmp_path, [paths], features)
        with pytest.raises(RuntimeError, match="features|Feature extraction"):
            pipe._find_similar_groups(paths)


class TestMixedDeviceBatchStacking:
    """One batch can hold nvJPEG (GPU) AND PIL (CPU) tensors at once.

    ``_decode_and_transform_gpu`` returns ``None`` when nvJPEG rejects a file
    and ``prepare()`` silently substitutes the CPU PIL tensor for that slot. A
    single corrupt JPEG is therefore enough to make the whole batch
    mixed-device, and ``torch.stack`` raises "Expected all tensors to be on the
    same device" -- the whole run died there. The tensors must be moved to the
    device BEFORE the stack; normalising the stacked result is too late.
    """

    def test_stack_normalises_every_tensor_before_stacking(self):
        """Device-agnostic: guards the ordering without needing CUDA.

        Uses recording stubs instead of real tensors so the assertion is about
        *when* ``.to()`` happens and *what* it targets, not about hardware.
        """
        from deh.duplicate.features import FeatureExtractor

        ex = FeatureExtractor.__new__(FeatureExtractor)  # skip model download
        ex.device = torch.device("cpu")

        seen = []

        class _Recorder:
            def __init__(self, tag):
                self.tag = tag

            def to(self, device):
                seen.append((self.tag, str(device)))
                return torch.zeros(3, 224, 224)

        out = ex._stack_on_device([_Recorder("a"), _Recorder("b")])

        assert [tag for tag, _ in seen] == ["a", "b"], (
            f"every tensor must be moved before the stack, saw {seen}"
        )
        assert all(dev == str(ex.device) for _, dev in seen), (
            f"each tensor must target the extractor device, saw {seen}"
        )
        assert tuple(out.shape) == (2, 3, 224, 224)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_corrupt_jpeg_in_batch_does_not_kill_the_run(self, tmp_path, monkeypatch):
        """End-to-end shape of the real crash: one bad file, whole batch dies."""
        from PIL import Image
        from deh.duplicate.features import FeatureExtractor

        paths = []
        for i in range(3):
            p = tmp_path / f"{i}.jpg"
            Image.new("RGB", (240, 200), (i * 40, 20, 30)).save(p)
            paths.append(str(p))

        ex = FeatureExtractor()
        if ex.device.type != "cuda":
            pytest.skip("pick_device() chose CPU; mixed-device batch needs CUDA")

        real_gpu_decode = ex._decode_and_transform_gpu

        def flaky(path):
            # Exactly what a corrupt JPEG does today: nvJPEG fails, returns
            # None, and the CPU PIL tensor takes its place in the batch.
            return None if path.name == "1.jpg" else real_gpu_decode(path)

        monkeypatch.setattr(ex, "_decode_and_transform_gpu", flaky)

        feats = ex.extract_batch(paths, batch_size=3)

        assert len(feats) == 3, (
            f"a fallback in one slot must not cost the other images: {list(feats)}"
        )
        for path, feat in feats.items():
            assert feat.device.type == "cuda", f"{path}: features left on {feat.device}"
            arr = feat.cpu().numpy()
            assert arr.shape == (ex.feature_dim,), f"{path}: got {arr.shape}"
