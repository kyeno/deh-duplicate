"""Tests for deh.duplicate.quality scoring and winner/loser selection."""
from pathlib import Path

from PIL import Image

from deh.duplicate.quality import QualityAssessor


def _checkerboard(path, size=512):
    """High-detail image: strong edges -> high sharpness & larger lossless file."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    px = img.load()
    for y in range(size):
        for x in range(size):
            if (x // 8 + y // 8) % 2 == 0:
                px[x, y] = (255, 255, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)


def _flat(path, color=(128, 128, 128), size=512):
    """Low-detail image: uniform -> zero edge variance, tiny compressed file."""
    img = Image.new("RGB", (size, size), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)


class TestScoreImage:
    def test_score_in_unit_range(self, tmp_path):
        qa = QualityAssessor()
        for p in (_checkerboard(tmp_path / "sharp.png"), _flat(tmp_path / "flat.png")):
            s = qa.score_image(p)
            assert 0.0 <= s <= 1.0

    def test_sharper_scores_higher_than_flat(self, tmp_path):
        sharp = _checkerboard(tmp_path / "sharp.png")
        flat = _flat(tmp_path / "flat.png")
        qa = QualityAssessor()
        assert qa.score_image(sharp) > qa.score_image(flat)


class TestEvaluateGroup:
    def test_partition_and_winner_exists(self, tmp_path):
        a = _checkerboard(tmp_path / "a.png")   # clearly better
        b = _flat(tmp_path / "b.png")           # clearly worse
        winners, losers = QualityAssessor().evaluate_group([a, b])

        # Winners and losers must partition the input with no overlap.
        assert set(winners) | set(losers) == {a, b}
        assert not (set(winners) & set(losers))
        assert len(winners) >= 1
        # The sharper image must be kept as a winner.
        assert a in winners
        assert b in losers

    def test_duplicates_collapse_to_single_winner(self, tmp_path):
        # Two copies of the same image now yield exactly one keeper + one loser.
        p = _checkerboard(tmp_path / "same.png")
        winners, losers = QualityAssessor().evaluate_group([p, p])
        assert len(winners) == 1
        assert len(losers) == 1
        assert winners[0] == p

    def test_always_exactly_one_winner(self, tmp_path):
        qa = QualityAssessor()
        paths = [
            _flat(tmp_path / "f0.png", size=128),
            _flat(tmp_path / "f1.png", size=256),
            _checkerboard(tmp_path / "c0.png", size=384),
            _flat(tmp_path / "f2.png", size=512),
            _checkerboard(tmp_path / "c1.png", size=640),
        ]
        winners, losers = qa.evaluate_group(paths)
        assert len(winners) == 1                       # always a single keeper
        assert len(losers) == len(paths) - 1
        assert set(winners).isdisjoint(set(losers))
        assert (set(winners) | set(losers)) == set(paths)

    def test_tie_breaks_by_larger_file(self, tmp_path):
        small = tmp_path / "small.bin"; medium = tmp_path / "medium.bin"
        large = tmp_path / "large.bin"
        for f, n in ((small, 100), (medium, 500), (large, 900)):
            f.write_bytes(b"x" * n)
        qa = QualityAssessor()
        qa.score_image = lambda p: 0.7                 # force an exact score tie
        winners, losers = qa.evaluate_group([str(small), str(medium), str(large)])
        assert winners == [str(large)]                 # larger file wins the tie
        assert sorted(losers) == sorted([str(small), str(medium)])

# ---------------------------------------------------------------------------
# New behaviors: sharpness-first scoring with JPEG-artifact & grain penalties,
# and the "always pick the non-cropped image" rule.
# ---------------------------------------------------------------------------
import numpy as np
from PIL import ImageFilter


def _texture(path, w=320, h=240, seed=7):
    """Non-repeating textured image -- unique enough for template matching."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)


def _smooth_image(path, w=320, h=320):
    """Photo-like smooth content (gradient + solid shapes): low noise, real edges."""
    yy, xx = np.mgrid[0:h, 0:w]
    arr = (yy / h * 160.0 + xx / w * 95.0).copy()
    for cx, cy, r in ((0.3, 0.4, 0.18), (0.7, 0.65, 0.12)):
        mask = ((xx - cx * w) ** 2 + (yy - cy * h) ** 2) < (r * min(w, h)) ** 2
        arr[mask] += 90
    out = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(str(path))
    return str(path)


class TestCropRule:
    def test_cropped_version_loses_to_full_image(self, tmp_path):
        full = _texture(tmp_path / "full.png")
        crop_box = (40, 30, 280, 210)   # strictly interior region of the full image
        crop_p = str(tmp_path / "crop.png")
        Image.open(full).crop(crop_box).save(crop_p)

        qa = QualityAssessor()
        winners, losers = qa.evaluate_group([crop_p, full])
        assert winners == [full]
        assert set(losers) == {crop_p}

    def test_crop_rule_independent_of_input_order(self, tmp_path):
        full = _smooth_image(tmp_path / "full.png")
        crop_p = str(tmp_path / "crop.png")
        Image.open(full).crop((60, 40, 260, 280)).save(crop_p)

        w1, l1 = QualityAssessor().evaluate_group([full, crop_p])
        w2, l2 = QualityAssessor().evaluate_group([crop_p, full])
        assert w1 == w2 == [full]
        assert set(l1) == set(l2) == {crop_p}

    def test_resized_crop_still_loses_to_full_image(self, tmp_path):
        # A crop that was upscaled after cropping.
        full = _smooth_image(tmp_path / "full.png")
        crop_p = str(tmp_path / "resized_crop.png")
        Image.open(full).crop((60, 40, 260, 200)).resize((300, 240), Image.BICUBIC).save(crop_p)

        winners, losers = QualityAssessor().evaluate_group([crop_p, full])
        assert winners == [full]
        assert set(losers) == {crop_p}

    def test_unrelated_images_not_flagged_as_crops(self, tmp_path):
        a = _texture(tmp_path / "a.png", seed=1)
        b = _texture(tmp_path / "b.png", seed=2)
        qa = QualityAssessor()
        assert qa._find_cropped_versions([a, b]) == {}
        winners, losers = qa.evaluate_group([a, b])   # still resolves to one winner
        assert len(winners) == 1
        assert set(winners) | set(losers) == {a, b}
        assert not (set(winners) & set(losers))


class TestJpegArtifactPenalty:
    def test_compressed_jpeg_scores_below_clean_png(self, tmp_path):
        clean = _texture(tmp_path / "clean.png", seed=3)
        jpg = str(tmp_path / "compressed.jpg")
        Image.open(clean).convert("RGB").save(jpg, quality=10)

        qa = QualityAssessor()
        assert qa.score_image(clean) > qa.score_image(jpg)

    def test_artifact_score_higher_for_heavily_compressed(self, tmp_path):
        clean = _texture(tmp_path / "clean.png", seed=4)
        jpg = str(tmp_path / "compressed.jpg")
        Image.open(clean).convert("RGB").save(jpg, quality=10)

        qa = QualityAssessor()
        assert qa._jpeg_artifact_score(Path(jpg)) > qa._jpeg_artifact_score(Path(clean))


class TestGrainCheck:
    @staticmethod
    def _grain_pair(tmp_path):
        base_p = _smooth_image(tmp_path / "base.png")
        arr = np.array(Image.open(base_p), dtype=np.float64)
        rng = np.random.default_rng(3)
        noisy = np.clip(arr + rng.normal(0.0, 8.0, size=arr.shape), 0, 255).astype(np.uint8)
        noise_p = str(tmp_path / "noisy.png")
        Image.fromarray(noisy, mode="L").save(noise_p)
        return base_p, noise_p

    def test_grain_level_higher_for_noisy_copy(self, tmp_path):
        base_p, noise_p = self._grain_pair(tmp_path)
        qa = QualityAssessor()
        assert qa._grain_level(Path(noise_p)) > qa._grain_level(Path(base_p))

    def test_clean_image_beats_heavily_grained_copy(self, tmp_path):
        base_p, noise_p = self._grain_pair(tmp_path)
        qa = QualityAssessor()
        assert qa.score_image(base_p) > qa.score_image(noise_p)


class TestReencodePreference:
    """Regression: a downscaled + sharpened re-encode must not beat its own source.

    Web pipelines typically take an original, halve it, apply unsharp masking to
    "compensate", and re-compress -- producing a smaller copy whose halo-inflated
    edge energy used to outscore the larger original in group selection.
    """

    @staticmethod
    def _photo_pair(tmp_path):
        w, h = 1280, 960
        yy, xx = np.mgrid[0:h, 0:w]
        arr = (yy / h * 170.0 + xx / w * 90.0).copy()
        rng = np.random.default_rng(7)
        for _ in range(60):
            cx, cy = int(rng.integers(60, w - 60)), int(rng.integers(60, h - 60))
            r = int(rng.integers(35, 110))
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
            arr[mask] += int(rng.integers(-60, 90))
        arr = arr + rng.normal(0.0, 3.0, size=arr.shape)
        base = np.clip(arr, 0, 255).astype(np.uint8)

        orig_p = str(tmp_path / "original.jpg")
        Image.fromarray(base, mode="L").convert("RGB").save(orig_p, quality=92)

        small = Image.fromarray(base, mode="L").resize((w // 2, h // 2), Image.BILINEAR)
        sharpened = small.filter(ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=2))
        copy_p = str(tmp_path / "copy_sharpened_small.jpg")
        sharpened.convert("RGB").save(copy_p, quality=78)
        return orig_p, copy_p

    def test_original_beats_downscaled_sharpened_copy(self, tmp_path):
        orig_p, copy_p = self._photo_pair(tmp_path)
        qa = QualityAssessor()
        assert qa.score_image(orig_p) > qa.score_image(copy_p)

    def test_group_keeps_original_over_reencode(self, tmp_path):
        orig_p, copy_p = self._photo_pair(tmp_path)
        winners, losers = QualityAssessor().evaluate_group([copy_p, orig_p])
        assert winners == [orig_p]
        assert set(losers) == {copy_p}


class TestNearTieResolutionPreference:
    """Within TIE_BAND the higher-pixel-count original wins -- even when it is
    stored in a SMALLER file (which would have won under the old filesize-only
    tiebreak)."""

    def test_higher_pixels_wins_within_tie_band_even_if_file_smaller(self, tmp_path):
        big_p = str(tmp_path / "big.png")
        Image.new("L", (640, 640), 128).save(big_p)      # solid color -> ~1KB file
        small_arr = np.random.default_rng(3).integers(0, 256, size=(320, 320), dtype=np.uint8)
        small_p = str(tmp_path / "small.png")             # noise -> much larger file
        Image.fromarray(small_arr, mode="L").save(small_p)

        qa = QualityAssessor()
        qa.score_image = lambda p: 0.7                    # force an exact score tie
        winners, losers = qa.evaluate_group([str(small_p), str(big_p)])
        assert winners == [big_p]                         # more pixels wins the near-tie


class TestDecisionNotes:
    """evaluate_with_notes must explain WHY each member won or lost -- this is what
    lets the GUI justify framing a lower-scoring image over a higher-scoring one."""

    def test_partition_invariants_preserved(self, tmp_path):
        a = _checkerboard(tmp_path / "a.png")   # clearly better
        b = _flat(tmp_path / "b.png")           # clearly worse
        w1, l1 = QualityAssessor().evaluate_group([a, b])
        w2, l2, notes = QualityAssessor().evaluate_with_notes([a, b])
        assert (w2, l2) == (w1, l1)             # identical decision to before
        assert set(notes) == {a, b}             # every input path gets an entry
        assert not (set(w2) & set(l2))          # still a disjoint partition

    def test_crop_demotion_note_names_the_source(self, tmp_path):
        full = _texture(tmp_path / "full.png")
        crop_p = str(tmp_path / "crop.png")
        Image.open(full).crop((40, 30, 280, 210)).save(crop_p)

        winners, losers, notes = QualityAssessor().evaluate_with_notes([crop_p, full])
        assert winners == [full] and set(losers) == {crop_p}
        assert Path(full).name in notes[crop_p]            # names the image it lost to
        text = notes[crop_p].lower()
        assert ("variant" in text or "cropped" in text) and "preferred" in text
        assert notes[full] == ""                           # plain score win: no note needed

    def test_near_tie_pixel_preference_is_explained(self, tmp_path):
        small = tmp_path / "small.bin"; large = tmp_path / "large.bin"
        small.write_bytes(b"x" * 10_000)          # plain blobs: no pixels, filesize differs
        large.write_bytes(b"x" * 50_000)
        qa = QualityAssessor()
        # The larger file scores LOWER but within TIE_BAND -> pixel/file preference
        # wins; that override of raw score order must be explained in its note.
        qa.score_image = lambda p: 0.76 if Path(p).name == "large.bin" else 0.80
        winners, losers, notes = qa.evaluate_with_notes([str(small), str(large)])
        assert winners == [str(large)]
        assert notes[str(large)].startswith("Near-tie")
        assert Path(small).name in notes[str(large)]
        assert notes[str(small)] == ""

    def test_equal_resolution_note_does_not_claim_higher_resolution(self, tmp_path):
        # D1 regression: both images have identical w×h, so the pixel preference
        # cannot decide -- the winner was picked by FILE SIZE and the note must say
        # so instead of claiming a higher resolution.
        solid_p = str(tmp_path / "solid.png")
        Image.new("L", (256, 256), 128).save(solid_p)          # same pixels, tiny file
        rng = np.random.default_rng(11)
        noise_arr = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
        noise_p = str(tmp_path / "noise.png")                   # same pixels, large file
        Image.fromarray(noise_arr, mode="L").save(noise_p)

        qa = QualityAssessor()
        qa.score_image = lambda p: 0.76 if Path(p).name == "noise.png" else 0.80
        winners, losers, notes = qa.evaluate_with_notes([solid_p, noise_p])
        assert winners == [noise_p]                             # filesize breaks the tie
        text = notes[noise_p].lower()
        assert text.startswith("near-tie") and "solid.png" in text
        assert "resolution" not in text                         # no false claim
        assert "file" in text                                   # names the real tiebreak

    def test_higher_resolution_note_claimed_when_pixels_differ(self, tmp_path):
        small_p = str(tmp_path / "small.png")
        Image.new("L", (320, 320), 128).save(small_p)
        big_p = str(tmp_path / "big.png")
        rng = np.random.default_rng(12)
        Image.fromarray(rng.integers(0, 256, size=(640, 640), dtype=np.uint8),
                        mode="L").save(big_p)

        qa = QualityAssessor()
        # The BIGGER image scores lower but within TIE_BAND -> pixel preference wins;
        # resolutions genuinely differ here, so the note may claim higher resolution.
        qa.score_image = lambda p: 0.76 if Path(p).name == "big.png" else 0.80
        winners, losers, notes = qa.evaluate_with_notes([small_p, big_p])
        assert winners == [big_p]
        text = notes[big_p].lower()
        assert "higher resolution preferred" in text and "small.png" in text


class TestGrainySharpBeatsSmoothJpegTwin:
    """Quality-preference check: a grainy+sharp original must beat its
    smooth, blurry, JPEG-artifact-heavy twin -- the JPEG penalty fires before
    the grain penalty, and among equal-resolution twins sharpness dominates."""

    @staticmethod
    def _twin_pair(tmp_path):
        w, h = 960, 720
        yy, xx = np.mgrid[0:h, 0:w]
        arr = (yy / h * 170.0 + xx / w * 90.0).copy()
        rng = np.random.default_rng(42)
        for _ in range(80):
            cx, cy = int(rng.integers(60, w - 60)), int(rng.integers(60, h - 60))
            r = int(rng.integers(35, 110))
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
            arr[mask] += int(rng.integers(-60, 90))
        base = np.clip(arr, 0, 255).astype(np.uint8)

        # Twin A: sensor grain on top of crisp content, high-quality encode.
        a_arr = np.clip(base.astype(float) + rng.normal(0.0, 8.0, size=base.shape),
                        0, 255).astype(np.uint8)
        a_p = str(tmp_path / "grainy_sharp.jpg")
        Image.fromarray(a_arr, mode="L").convert("RGB").save(a_p, quality=92)

        # Twin B: blurred smooth re-encode with heavy JPEG artifacts.
        b_p = str(tmp_path / "smooth_blurry_jpeg.jpg")
        (Image.fromarray(base, mode="L").convert("RGB")
         .filter(ImageFilter.GaussianBlur(radius=4.0)).save(b_p, quality=35))
        return a_p, b_p

    def test_grainy_sharp_scores_higher(self, tmp_path):
        a_p, b_p = self._twin_pair(tmp_path)
        qa = QualityAssessor()
        assert qa.score_image(a_p) > qa.score_image(b_p)

    def test_group_keeps_grainy_sharp_original(self, tmp_path):
        a_p, b_p = self._twin_pair(tmp_path)
        winners, losers = QualityAssessor().evaluate_group([b_p, a_p])
        assert winners == [a_p]
        assert set(losers) == {b_p}


class TestMetricMemoization:
    """D6: the metric set must run once per image per run -- and stay exact."""

    def test_repeated_scoring_reuses_cached_metrics(self, tmp_path, monkeypatch):
        qa = QualityAssessor()
        p = _checkerboard(tmp_path / "once.png", size=256)
        calls = []
        original = qa._grain_level
        monkeypatch.setattr(qa, "_grain_level",
                            lambda path: (calls.append(path), original(path))[1])

        first = qa.score_image(p)
        second = qa.score_image(p)

        assert first == second
        assert len(calls) == 1              # the second call is a cache hit

    def test_decision_and_score_map_share_a_single_metric_pass(self, tmp_path, monkeypatch):
        a = _checkerboard(tmp_path / "a.png", size=256)
        b = _flat(tmp_path / "b.png", size=256)
        qa = QualityAssessor()
        calls = []
        original = qa._compute_metrics

        def counting(path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(qa, "_compute_metrics", counting)
        scores = {}
        qa.evaluate_with_notes([a, b], scores_out=scores)

        assert set(scores) == {a, b}        # every member gets a display score
        assert len(calls) == 2              # ...from one metric pass per image

    def test_rewritten_file_is_re_measured(self, tmp_path):
        qa = QualityAssessor()
        p = str(tmp_path / "change.png")
        Image.new("L", (320, 320), 128).save(p)
        before = qa.score_image(p)

        Image.new("L", (640, 640), 128).save(p)     # same path, new size & mtime
        after = qa.score_image(p)

        assert after == QualityAssessor().score_image(p)   # fresh computation
        assert after != before                             # resolution moved the score


class _FixedMachine:
    """Minimal MachineService stand-in: fixed worker count, no probing."""

    def __init__(self, jobs):
        self.jobs = jobs

    def plan_images(self, *args, **kwargs):
        return self


class TestIntraGroupParallelism:
    """The pool is an implementation detail: decisions and scores must not move."""

    @staticmethod
    def _group(tmp_path):
        full = _texture(tmp_path / "full.png")
        crop_p = str(tmp_path / "crop.png")
        Image.open(full).crop((40, 30, 280, 210)).save(crop_p)
        other = _flat(tmp_path / "other.png", size=128)
        return [crop_p, full, other]

    def test_parallel_plan_decides_identically_to_serial(self, tmp_path):
        group = self._group(tmp_path)
        serial = QualityAssessor(machine=_FixedMachine(1))
        parallel = QualityAssessor(machine=_FixedMachine(4))

        serial_scores, parallel_scores = {}, {}
        serial_result = serial.evaluate_with_notes(group, scores_out=serial_scores)
        parallel_result = parallel.evaluate_with_notes(group, scores_out=parallel_scores)

        assert parallel_result == serial_result
        assert parallel_scores == serial_scores          # float-exact scores
        assert set(parallel_scores) == set(group)        # crops get display scores too

        winners, losers, notes = parallel_result
        assert winners == [group[1]]                     # full frame beats its crop
        assert "variant" in notes[group[0]].lower()

    def test_duplicate_entries_keep_first_match_semantics(self, tmp_path):
        full = _texture(tmp_path / "full.png")
        crop_p = str(tmp_path / "crop.png")
        Image.open(full).crop((40, 30, 280, 210)).save(crop_p)

        qa = QualityAssessor(machine=_FixedMachine(4))
        crop_of = qa._find_cropped_versions([crop_p, crop_p, full])

        assert list(crop_of) == [crop_p]                 # duplicate collapses
        assert crop_of[crop_p][0] == full                # ...to the real source
