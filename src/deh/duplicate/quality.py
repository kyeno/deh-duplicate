# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Quality Assessment Module

Evaluates image quality using multiple metrics to rank images within similar
groups and determine which should be preserved versus quarantined.

Scoring priorities (highest first):

1. **Resolution / carried data** -- within a near-duplicate group the larger
   original almost always beats a downscaled + sharpened re-encode of itself;
   human review on web-photo corpora confirms this is the signal users care
   about most. Calibration anchors are set for small images (~0.25MP floor).
2. **Edge sharpness** -- crisp real edges matter second. Measured on a
   median-filtered copy capped to a fixed working size, so sensor grain does not
   masquerade as detail and an unsharp-mask halo cannot win simply because the
   image carries more pixels.
3. **JPEG artifact penalty** -- coherent 8x8 block-boundary periodicity and
   very low bits-per-pixel reduce the score of re-compressed copies.
4. **Grain check** -- high-frequency noise in flat regions applies a final
   penalty; kept moderate because re-compression flattens natural grain, so a
   maximal penalty would favor processed copies over originals. Filesize (bpp)
   carries the smallest weight: it also rises with plain sensor noise.

Near-tie rule: when two group candidates score within ``TIE_BAND`` of each
other, the higher-pixel-count original is preferred -- a downscaled + sharpened
re-encode must not beat its own source. File size remains the fallback tiebreak.

Crop rule: if one image is detected as a cropped version of another member of
the group, the non-cropped (fuller) image ALWAYS wins regardless of scores.
Detection uses normalized cross-correlation template matching with a small
scale sweep so crops that were resized after cropping are caught too. A trim
gate (CROP_MIN_TRIM_FRACTION, ~1.5%/side for symmetric trims) keeps tiny
reframes from counting as crops.

Decision transparency: ``evaluate_with_notes`` returns human-readable reasons
per path alongside the winners/losers split (e.g. "cropped variant of X",
"near-tie resolved toward higher resolution") so GUIs can explain why a
lower-scoring image was selected over a higher-scoring one.

Performance: metric components are memoized per ``(path, mtime_ns, size)`` for
the lifetime of one assessor, so an image is never measured twice in a run --
the pipeline needs each score both for the decision and for the GUI display
map, and ``evaluate_with_notes`` can hand that map back via ``scores_out``. The
independent per-image metrics and the crop-pair checks run on an intra-group
thread pool sized by the machine service; groups are never parallelised
against each other because the GUI dialog for group N waits on group N.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

from ..core.machine import machine as default_machine
from ..core.terminal import notice


class QualityAssessor:
    """Ranks images by combined quality score."""

    # Weights for different quality factors (must sum to 1.0). On near-duplicate
    # web-photo groups the "more data" signal dominates: human review shows the
    # larger original consistently beats its downscaled + sharpened re-encode.
    # Sharpness stays second; it is measured on a capped working size precisely
    # because raw edge energy rewards unsharp-mask halos. Filesize carries the
    # smallest weight -- bits-per-pixel also rises with plain sensor noise.
    WEIGHT_SHARPNESS = 0.40
    WEIGHT_RESOLUTION = 0.50
    WEIGHT_FILESIZE = 0.10

    # Maximum fractional reduction applied on top of the base score. JPEG
    # artifacts are penalized first; grain is checked only afterwards.
    MAX_PENALTY_JPEG_ARTIFACTS = 0.60
    MAX_PENALTY_GRAIN = 0.60

    # --- Calibration anchors for the "more data" components -----------------
    # Web-scraped corpora are full of sub-megapixel photos; the old 5-50MP log
    # scale and 2-10bpp range scored them ALL as zero, which let halo-inflated
    # sharpness decide near-duplicate groups on its own. These anchors keep
    # resolution/bpp meaningful down to web-photo territory while still
    # saturating at camera-original levels.
    RES_MIN_PIXELS = 250_000
    RES_MAX_PIXELS = 25_000_000
    BPP_MIN = 0.15
    BPP_MAX = 1.5

    # Sharpness is measured after capping the working size (longest side), so a
    # huge image cannot win on raw edge-pixel count and unsharp-mask halos
    # (fixed pixel width) contribute less relative weight in large images.
    SHARP_WORK_SIDE = 768

    # Near-tie band: when two group candidates score within this margin, prefer
    # the higher-pixel-count original over a downscaled re-encode; file size
    # remains the fallback tiebreak (see evaluate_group).
    TIE_BAND = 0.05

    # --- Crop detection (template matching via normalized cross-correlation) ---
    CROP_MATCH_THRESHOLD = 0.90      # NCC peak required to call it "same content"
    # A matched region must leave >= 3% total margin on at least one axis -- i.e.
    # roughly >1.5% trimmed PER SIDE for symmetric crops; smaller reframes are not
    # treated as crops. Chosen from corpus review: gates below ~2%/side start
    # demoting tight-crop compositions users actually preferred over full frames.
    CROP_MIN_TRIM_FRACTION = 0.03
    CROP_MIN_TEMPLATE_AREA = 0.02    # template must cover >= 2% of search image area
    CROP_WORK_SIDE = 768             # cap on working-image longest side (px)
    CROP_SCALE_SWEEP = (1.0, 0.8, 0.66, 0.5, 1.25, 1.5)   # resized-crop tolerance (~x1.25 steps over [0.5..1.5])

    # Crop prescreen levels, coarse -> fine: (thumbnail side px, NCC threshold).
    # The O(k^2) pair gate runs the same scale sweep on these thumbnails and
    # only a pair that stays above BOTH thresholds reaches the exact working-
    # resolution check. Thresholds sit safely below the tuned exact gates
    # (0.90 native / 0.85 resized): corpus calibration showed true crops peak
    # >= 0.84 at 192 px, so the gate only ever *adds* verification work for
    # ambiguous near-dupes; the winner/loser dump diff on the sample corpus is
    # the regression guard for the corpus-tuned crop rule.
    CROP_SCREEN_LEVELS = ((192, 0.80), (384, 0.84))

    # --- Intra-group parallel planning (machine service) ----------------------
    # One metrics worker holds a decoded grayscale copy plus the temporaries of
    # a single metric set (median filter + gradients, or the two FFT passes);
    # one crop row furthermore decodes two (mostly downscaled) working copies.
    METRIC_WORKER_BYTES_PER_PX = 40.0
    CROP_WORKER_BYTES_PER_PX = 64.0
    MAX_PARALLEL_WORKERS = 16        # same cap as the phash decode pool

    def __init__(self, machine=None, log_cb: Optional[Callable[[str], None]] = None):
        self.machine = machine or default_machine()
        # (path, mtime_ns, size) -> {sharpness, resolution, filesize, jpeg, grain}
        self._metric_cache = {}
        self._log_cb = log_cb

    def _log(self, text: str) -> None:
        """Send one warning line to the caller's ``log_cb`` (default: terminal)."""
        if self._log_cb is not None:
            self._log_cb(text)
        else:
            notice(text)

    def evaluate_group(self, paths: List[str]) -> Tuple[List[str], List[str]]:
        """Evaluate all images in a similar group and split into winners/losers.

        Thin compatibility wrapper around :meth:`_decide`; callers that also want
        the human-readable reason behind each selection should call
        :meth:`evaluate_with_notes` instead.
        """
        return self._decide(paths)[:2]

    def evaluate_with_notes(self, paths: List[str], scores_out: Optional[dict] = None):
        """Like :meth:`evaluate_group`, but also returns per-path decision notes.

        Returns ``(winners, losers, notes)`` where ``notes`` maps EVERY input path
        to a short human-readable explanation of its outcome -- e.g. why a
        lower-scoring image was kept over a higher-scoring one ("cropped variant
        of ...", "near-tie resolved toward higher resolution"). Paths with an
        empty-string note simply won on plain score order.

        When *scores_out* is given it is filled with the final display score of
        EVERY input path (0.0 for files that cannot be scored), so the caller
        never needs a second, full metric pass to build a GUI score map.
        """
        return self._decide(paths, scores_out)

    def _decide(self, paths: List[str], scores_out: Optional[dict] = None):
        """Core group decision: pick ONE winner and explain every member's fate.

        Selection rules, in order of precedence:
          1. **Crop rule** -- an image that is a cropped version of another
             member can never win; the non-cropped (fuller) image always wins,
             regardless of scores.
          2. Among the remaining candidates the highest-scoring image wins. When
             top candidates fall within ``TIE_BAND`` of each other (common for
             same-content pairs where one is a downscaled re-encode), the
             higher-pixel-count original is preferred; on an exact tie the
             larger file (more data) is preferred, then original scan order.

        Returns ``(winner_path, losers, notes)``. ``losers`` contains every input
        entry except the first occurrence of the winner (a duplicated path still
        collapses to one keeper + one loser). If no image could be scored at all,
        everything is kept (no losers) rather than guessing which to quarantine.
        With *scores_out* the decision additionally fills it with every path's
        score (see :meth:`evaluate_with_notes`).
        """
        if not paths:
            return ([], [], {})

        # Rule 1: demote any member that is a cropped version of another member.
        crop_of = self._find_cropped_versions(paths)   # {path: (source_path, trim_fraction)}
        candidates = [p for p in paths if p not in crop_of]
        if not candidates:
            # Every member was flagged as a crop of some other member (circular);
            # fall back to plain scoring rather than quarantining everything.
            candidates = list(paths)
            crop_of = {}

        # Warm the metric cache before combining scores: candidates always, crop
        # members too when the caller wants the GUI score map (they are scored by
        # _fill_scores). Already-cached images are skipped, and the one metric
        # pass per image is what the pipeline's double-scoring used to cost.
        self._precompute(paths if scores_out is not None else candidates)

        # Rule 2: best quality first among the full-frame candidates. Candidates
        # that land within TIE_BAND of the top score are treated as near-ties and
        # resolved toward the higher-pixel-count original -- a downscaled +
        # sharpened re-encode must not beat its own source. File size remains the
        # fallback tiebreak and stable ordering keeps the result deterministic.
        scored = []
        best_score = 0.0
        for path in candidates:
            try:
                score = self.score_image(path)
            except Exception as e:
                self._log(f"[Quality] Warning: Could not score {path}: {e}")
                score = 0.0
            try:
                filesize = os.path.getsize(path)
            except OSError:
                filesize = 0
            pixels = 0
            try:
                with Image.open(path) as im:
                    pixels = int(im.width * im.height)
            except (OSError, ValueError):     # unreadable image: rank on score alone
                pass
            scored.append((path, float(score), int(pixels), int(filesize)))
            best_score = max(best_score, float(score))

        notes = {p: "" for p in paths}

        # Safety edge: if no image could be scored at all, keep everything rather than
        # guessing which to quarantine.
        if best_score <= 0.0:
            self._fill_scores(paths, scores_out)
            return (list(paths), [], notes)

        top_score = max(t[1] for t in scored)
        band = [t for t in scored if t[1] >= top_score - self.TIE_BAND]
        # Within the near-tie band prefer more pixels (the original), then larger
        # file; max() keeps the first maximal entry so scan order breaks full ties.
        winner_path = max(band, key=lambda t: (t[2], t[3]))[0]

        # Explain every crop demotion (this is what lets a LOWER-scoring image win).
        for p, (src, frac) in crop_of.items():
            pct = round(100.0 * frac)
            suffix = f" (~{pct}% smaller framing)" if pct else ""
            notes[p] = (f"Cropped/smaller variant of {Path(src).name}{suffix}"
                        " -- full-frame image preferred")

        # If the banded winner's raw score sits below another member's, the pixel/
        # file preference overrode plain score order -- say so explicitly, naming
        # whichever tiebreak actually decided it (never "higher resolution" when
        # both images carry identical pixel counts).
        winner_entry = next((t for t in scored if t[0] == winner_path), None)
        if winner_entry is not None and len(band) > 1 \
                and winner_entry[1] < top_score - 1e-9:
            others = [t for t in band if t[0] != winner_path]
            other = max(others, key=lambda t: (t[1], t[0]))   # highest scorer; stable on ties
            if winner_entry[2] > other[2]:
                reason = "higher resolution preferred"
            elif winner_entry[3] > other[3]:
                reason = "larger file wins the tiebreak"
            else:
                reason = "resolved by stable scan order"
            notes[winner_path] = (f"Near-tie within +/-{self.TIE_BAND} of "
                                  f"{Path(other[0]).name} ({other[1]:.3f}) -- {reason}")

        losers = []
        skipped_winner = False
        for q in paths:
            if not skipped_winner and q == winner_path:
                skipped_winner = True
                continue
            losers.append(q)
        self._fill_scores(paths, scores_out)
        return ([winner_path], losers, notes)

    def score_image(self, path: str) -> float:
        """Compute combined quality score for an image (higher is better).

        Base score is a weighted mix of resolution, sharpness and file-size
        efficiency -- all normalized to [0, 1], each component memoized per
        ``(path, mtime_ns, size)`` so repeats within a run are cache hits. Two
        penalties are then applied IN ORDER:
          1. JPEG artifacts (block-boundary periodicity / very low bpp),
          2. grain (high-frequency noise in flat regions).
        The result is clamped back into [0, 1].
        """
        comp = self._metrics(path)

        # Component 1: Resolution (pixel count). Component 2: Sharpness (edge
        # strength on a grain-free copy). Component 3: file-size efficiency
        # (bits per pixel). Keep the exact combination order -- memoized and
        # freshly computed components must score bit-equal.
        base = (comp["resolution"] * self.WEIGHT_RESOLUTION +
                comp["sharpness"] * self.WEIGHT_SHARPNESS +
                comp["filesize"] * self.WEIGHT_FILESIZE)

        # Penalty stage 1: JPEG artifacts (blocking / heavy compression).
        base *= (1.0 - self.MAX_PENALTY_JPEG_ARTIFACTS * comp["jpeg"])

        # Penalty stage 2 (only then): grain / sensor noise in flat regions.
        base *= (1.0 - self.MAX_PENALTY_GRAIN * comp["grain"])

        return float(min(max(base, 0.0), 1.0))

    # ------------------------------------------------------------------ #
    # Metric memoization & intra-group parallelism                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(path) -> tuple:
        """File identity behind the cached metrics: ``(path, mtime_ns, size)``."""
        p = str(path)
        try:
            st = os.stat(p)
            return (p, st.st_mtime_ns, st.st_size)
        except OSError:
            return (p, None, None)

    def _metrics(self, path: str) -> dict:
        """Metric components of *path*, measured once per (path, mtime, size).

        The cache holds five floats per image (nothing heavy), and the key
        includes mtime+size so a file rewritten mid-run is re-measured instead
        of being served stale numbers.
        """
        key = self._cache_key(path)
        comp = self._metric_cache.get(key)
        if comp is None:
            comp = self._compute_metrics(str(path))
            self._metric_cache[key] = comp
        return comp

    def _compute_metrics(self, path: str) -> dict:
        """Run the independent per-image metric set once.

        Components are the raw [0, 1] signals (each metric catches its own I/O
        errors and falls back to 0.0); :meth:`score_image` combines them with
        the calibrated weights.
        """
        p = Path(path)
        return {
            "sharpness": self._sharpness_score(p),
            "resolution": self._resolution_score(p),
            "filesize": self._filesize_score(p),
            "jpeg": self._jpeg_artifact_score(p),
            "grain": self._grain_level(p),
        }

    def _precompute(self, paths) -> None:
        """Warm the metric cache for *paths* on an intra-group thread pool.

        Decode/median-filter/FFT kernels release the GIL, so threads scale the
        metric set without process overhead; sizing follows
        ``MachineService.plan_images`` -- the same DI pattern as
        ``HashEngine.compute_batch``. Groups are never parallelised against each
        other: the dialog for group N waits on group N.
        """
        pending, seen = [], set()
        for path in paths:
            key = self._cache_key(path)
            if key in self._metric_cache or key in seen:
                continue
            seen.add(key)
            pending.append((key, str(path)))
        if not pending:
            return

        def compute(path):
            try:
                return self._compute_metrics(path)
            except Exception as e:      # never abort the whole group
                self._log(f"[Quality] Warning: Could not score {path}: {e}")
                return None

        if len(pending) == 1:
            workers = 1
        else:
            workers = min(self._plan_workers(
                "quality-metrics", [p for _, p in pending],
                bytes_per_px=self.METRIC_WORKER_BYTES_PER_PX,
                env_prefix="QUALITY_METRICS"), len(pending))

        if workers <= 1:
            results = [compute(path) for _, path in pending]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(compute, [path for _, path in pending]))
        for (key, _), comp in zip(pending, results, strict=True):
            if comp is not None:
                self._metric_cache[key] = comp

    def _plan_workers(self, name: str, paths, *, bytes_per_px: float,
                      env_prefix: str) -> int:
        """Machine-sized worker count for one intra-group pool."""
        plan = self.machine.plan_images(
            name, paths,
            parent_bytes_per_px=0.1,        # parent caches five floats per image
            worker_bytes_per_px=bytes_per_px,
            in_flight_per_worker=1,         # each task holds one image
            cap_workers=self.MAX_PARALLEL_WORKERS,
            env_prefix=env_prefix)
        return max(1, int(plan.jobs))

    def _fill_scores(self, paths, scores_out: Optional[dict]) -> None:
        """Fill *scores_out* with the final score of EVERY path (failures 0.0).

        Paths already scored during the decision come straight from the metric
        cache, so this is not a second metric pass.
        """
        if scores_out is None:
            return
        for path in paths:
            if path in scores_out:
                continue
            try:
                scores_out[path] = self.score_image(path)
            except Exception:
                # score_image spans PIL, numpy, SciPy and (optionally) OpenCV;
                # the exact failure set is the union of all of them. An
                # unscoreable image degrades to 0.0 -- the same value the
                # per-image path in rank_group() reports -- rather than
                # aborting the group.
                scores_out[path] = 0.0

    def _resolution_score(self, path: Path) -> float:
        """Score based on pixel dimensions. Higher resolution wins."""
        try:
            with Image.open(path) as img:
                width, height = img.size
                pixels = width * height
        except (OSError, ValueError):      # unreadable image: no resolution score
            return 0.0

        # Log-scale normalization calibrated for web-photo corpora (~0.25MP up to
        # ~25MP). The old 5-50MP anchors scored every sub-megapixel image as zero,
        # which let sharpening artifacts decide near-duplicate groups on their own.
        if pixels <= 0:
            return 0.0
        lo = math.log(self.RES_MIN_PIXELS)
        hi = math.log(self.RES_MAX_PIXELS)
        score = min(max((math.log(pixels) - lo) / (hi - lo), 0.0), 1.0)
        return score

    def _sharpness_score(self, path: Path) -> float:
        """Score based on edge sharpness using gradient strength.

        Blurry or heavily compressed images have lower edge variance. The image
        is median-filtered first so that sensor grain / compression noise does
        not inflate the measurement -- only coherent real edges count (grain is
        handled separately by ``_grain_level``).
        Computes horizontal and vertical gradients then measures their combined variance.
        """
        try:
            img = Image.open(path).convert('L')  # Grayscale
            # Cap the working size so edge-pixel count cannot dominate and fixed-
            # width unsharp-mask halos contribute less relative weight in large images.
            side = max(img.width, img.height)
            if side > self.SHARP_WORK_SIDE:
                f = self.SHARP_WORK_SIDE / float(side)
                img = img.resize(
                    (max(1, int(round(img.width * f))), max(1, int(round(img.height * f)))),
                    Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)

            h, w = arr.shape
            if h < 3 or w < 3:
                return 0.5  # Too small to assess

            # Suppress isolated-pixel noise (grain) before measuring edges.
            # PIL 'L' keeps integer pixels, so the 8-bit median is identical
            # to filtering the float64 copy (see _median3).
            arr = self._median3(arr).astype(np.float64)

            # Compute gradients on interior pixels only (avoid boundary issues)
            # Horizontal difference: compare each pixel with its right neighbor
            grad_h = np.abs(arr[:-1, :-1] - arr[1:, :-1])
            # Vertical difference: compare each pixel with its bottom neighbor
            grad_v = np.abs(arr[:-1, :-1] - arr[:-1, 1:])

            # Combined edge strength at each interior point
            edges = grad_h + grad_v
            edge_variance = np.var(edges)

            # Normalize using log scale; typical values vary widely across images
            if edge_variance <= 0:
                return 0.0
            score = min(max((math.log(edge_variance + 1) - 2.0) / 6.0, 0.0), 1.0)
            return float(score)

        except Exception as e:
            self._log(f"[Quality] Sharpness error on {path}: {e}")
            return 0.0

    def _filesize_score(self, path: Path) -> float:
        """Score based on bits-per-pixel metric. Larger files generally indicate less compression."""
        try:
            file_size_bytes = os.path.getsize(path)

            with Image.open(path) as img:
                width, height = img.size
                pixels = width * height

            if pixels == 0 or file_size_bytes == 0:
                return 0.0

            bpp = (file_size_bytes * 8) / pixels  # Bits per pixel

            # Calibrated for web photos: re-encoded copies typically sit around
            # 0.3-1.5 bpp; the old 2-10bpp scale scored them all as zero.
            score = min(max((bpp - self.BPP_MIN) / (self.BPP_MAX - self.BPP_MIN), 0.0), 1.0)
            return score

        except Exception as e:
            self._log(f"[Quality] Filesize scoring error on {path}: {e}")
            return 0.0

    def _jpeg_artifact_score(self, path: Path) -> float:
        """Estimate JPEG artifact level in [0, 1] (higher = more artifacts).

        Two signals are combined:
          * **Block periodicity** -- JPEG blocking shows up as a coherent
            period-8 discontinuity pattern that is aligned across the whole
            image. The column/row-averaged difference signal is FFT-analyzed
            and the fraction of spectral energy at exactly period 8 is used.
            Natural edges do not coherently align to an 8px grid, so this stays
            near zero for clean images.
          * **Very low bits-per-pixel** -- only counted for actual JPEG files,
            since lossless formats can legitimately be tiny without artifacts.
        """
        try:
            with Image.open(path) as img:
                fmt = (img.format or "").upper()
                gray = np.array(img.convert("L"), dtype=np.float64)
        except (OSError, ValueError):      # unreadable image: no compression score
            return 0.0

        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.0  # too small to resolve an 8px block structure reliably

        d_h = np.abs(np.diff(gray, axis=1)).mean(axis=0)   # per-column diff strength
        d_v = np.abs(np.diff(gray, axis=0)).mean(axis=1)   # per-row diff strength
        periodicity = max(self._period8_fraction(d_h), self._period8_fraction(d_v))
        # ~6% of spectral energy at period-8 -> no penalty; >= 40% -> full penalty.
        blockiness = min(max((periodicity - 0.06) / 0.34, 0.0), 1.0)

        score = blockiness
        if fmt == "JPEG":
            try:
                file_size_bytes = os.path.getsize(path)
                pixels = h * w
                bpp = (file_size_bytes * 8) / max(pixels, 1)
                low_bpp = min(max((0.9 - bpp) / 0.9, 0.0), 1.0)  # bpp<=0 -> 1 ; bpp>=0.9 -> 0
                score = max(score, 0.5 * low_bpp)
            except OSError:
                pass
        return float(min(max(score, 0.0), 1.0))

    @staticmethod
    def _period8_fraction(signal: np.ndarray) -> float:
        """Fraction of a signal's FFT power concentrated at period 8 samples."""
        x = signal - signal.mean()
        if x.std() < 1e-6:
            return 0.0
        spec = np.fft.rfft(x)
        power = spec.real ** 2 + spec.imag ** 2
        n_bins = len(power)
        k = int(round(len(signal) / 8))   # bin corresponding to period 8
        lo = max(1, k - 1)
        hi = min(n_bins - 1, k + 1)
        total = float(power.sum())
        if total <= 0 or lo >= n_bins:
            return 0.0
        return float(power[lo:hi + 1].sum() / total)

    # ------------------------------------------------------------------ #
    # Fast kernels (cv2 bit-identical to the SciPy reference)             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _median3(gray_u8: np.ndarray) -> np.ndarray:
        """3x3 median of an 8-bit image.

        cv2.medianBlur is bit-identical to ``ndimage.median_filter(size=3)``
        on integer data (verified per-pixel on real and random corpora) and
        ~3x faster on full-resolution images, which dominates the grain
        metric. Falls back to SciPy when OpenCV is unavailable.
        """
        try:
            import cv2
        except ImportError:
            return ndimage.median_filter(gray_u8, size=3)
        try:
            return cv2.medianBlur(np.ascontiguousarray(gray_u8), 3)
        except cv2.error:                  # OpenCV refuses this input; SciPy can
            return ndimage.median_filter(gray_u8, size=3)

    def _grain_level(self, path: Path) -> float:
        """Estimate grain/noise level in [0, 1] (higher = more grain).

        Measured as the residual between the image and its median-filtered copy,
        restricted to FLAT regions (where the denoised image is smooth). This
        isolates sensor grain from real texture/edges.
        """
        try:
            with Image.open(path) as img:
                gray_u8 = np.array(img.convert("L"), dtype=np.uint8)
        except (OSError, ValueError):      # unreadable image: assumed clean
            return 0.0

        h, w = gray_u8.shape
        if h < 32 or w < 32:
            return 0.0

        gray = gray_u8.astype(np.float64)
        med = self._median3(gray_u8).astype(np.float64)
        residual = np.abs(gray - med)

        gy, gx = np.gradient(med)
        grad = np.hypot(gx, gy)
        flat = grad < 4.0   # smooth areas only (8-bit scale)
        if int(flat.sum()) < max(64, int(0.01 * h * w)):
            return 0.0      # fully detailed image: no flat area to measure grain in

        sigma = float(residual[flat].std())
        # ~1.5 levels -> negligible; >= 6.5 levels of residual noise -> full penalty.
        score = min(max((sigma - 1.5) / 5.0, 0.0), 1.0)
        return float(score)

    # ------------------------------------------------------------------ crops --

    def _find_cropped_versions(self, paths: List[str]) -> dict:
        """Map each cropped member to ``(source_path, max_trim_fraction)``.

        Only members that are genuine crops/smaller variants of ANOTHER group
        member appear as keys; the value records which source they were matched
        against plus how much smaller their framing is (for UI messaging).

        Candidate rows are independent, so they run on an intra-group thread
        pool (sized by the machine service); merging the rows in scan order
        reproduces the sequential "first matching source wins" rule exactly,
        duplicate path entries included.
        """
        crop_of = {}
        n = len(paths)
        if n < 2:
            return crop_of

        workers = min(self._plan_workers(
            "quality-crops", paths,
            bytes_per_px=self.CROP_WORKER_BYTES_PER_PX,
            env_prefix="QUALITY_CROPS"), n)

        # Source working sets are candidate-independent (the shared downscale
        # factor comes from the source's own size), so decode them once per
        # source instead of once per ordered pair; each candidate row then
        # decodes its own image once. Both passes use the intra-group pool.
        if workers <= 1:
            sources = [self._crop_source_bundle(p) for p in paths]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                sources = list(pool.map(self._crop_source_bundle, paths))

        def scan_row(i: int):
            """First matching source for ``paths[i]``, or None."""
            a = paths[i]
            cand_full = cand_dims = None
            for j in range(n):
                if i == j:
                    continue
                bundle = sources[j]
                if bundle is None:
                    continue
                if cand_full is None:      # decode the candidate once per row
                    try:
                        with Image.open(a) as img:
                            cand_dims = img.size
                            cand_full = img.convert("L")
                    except (OSError, ValueError):
                        return None        # unreadable: no pair can match
                b = paths[j]
                try:
                    is_crop, frac = self._crop_check_pair(cand_full, cand_dims, bundle)
                except Exception as e:
                    self._log(f"[Quality] Crop check failed for {a} vs {b}: {e}")
                    continue
                if is_crop:
                    return (a, b, float(frac))
            return None

        if workers <= 1:
            rows = [scan_row(i) for i in range(n)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(scan_row, range(n)))

        for match in rows:
            if match is None:
                continue
            a, b, frac = match
            if a in crop_of:
                continue        # duplicate entry: the first processed row wins
            crop_of[a] = (b, frac)
        return crop_of

    # --- crop-check internals ---------------------------------------------
    # `_find_cropped_versions` uses a thumbnail prescreen before `_crop_sweep`
    # verifies each candidate pair at working resolution.

    def _crop_source_bundle(self, path):
        """Candidate-independent working set of one source image.

        Returns ``(dims, work_pil, ((thumb_pil, factor), ...))`` where *work_pil*
        is the L image downscaled so its longest side is <= CROP_WORK_SIDE, and each entry in
        *thumbs* is a prescreen thumbnail of that work image plus the factor
        applied. ``None`` when the file cannot be read.
        """
        try:
            with Image.open(path) as img:
                dims = img.size
                source = img.convert("L")
        except (OSError, ValueError):      # unreadable: caller skips this source
            return None
        f = min(1.0, self.CROP_WORK_SIDE / max(source.width, source.height))
        work = self._scale_pil(source, f)
        thumbs = []
        for side, _threshold in self.CROP_SCREEN_LEVELS:
            factor = min(1.0, side / max(work.width, work.height))
            thumbs.append((self._scale_pil(work, factor), factor))
        return (dims, work, tuple(thumbs))

    def _crop_check_pair(self, cand_full, cand_dims, source_bundle) -> Tuple[bool, float]:
        """Crop verdict from pre-decoded images.

        Applies the same size pre-filter, then the thumbnail prescreen gates;
        surviving pairs are decided by the exact `_crop_sweep`.
        """
        sw_, sh_ = source_bundle[0]
        cw, ch = cand_dims
        if cw <= 0 or ch <= 0 or sw_ <= 0 or sh_ <= 0 or (cw, ch) == (sw_, sh_):
            return (False, 0.0)
        fits_any_scale = any(
            max(1, int(round(cw * s))) <= sw_ and max(1, int(round(ch * s))) <= sh_
            for s in self.CROP_SCALE_SWEEP)
        if not fits_any_scale:
            return (False, 0.0)

        src_work = source_bundle[1]
        f = min(1.0, self.CROP_WORK_SIDE / max(sw_, sh_))
        cand_work = self._scale_pil(cand_full, f)
        for (_side, threshold), (thumb, factor) in zip(
                self.CROP_SCREEN_LEVELS, source_bundle[2], strict=True):
            cand_thumb = self._scale_pil(cand_work, factor)
            hay_thumb = np.asarray(thumb, dtype=np.float64) / 255.0
            if not self._crop_screen_sweep(cand_thumb, hay_thumb, threshold):
                return (False, 0.0)
        hay = np.asarray(src_work, dtype=np.float64) / 255.0
        return self._crop_sweep(cand_work, hay)

    @staticmethod
    def _scale_pil(pil, factor: float):
        """BILINEAR resize by *factor* (no-op at >= 1.0), floor 1 px per side."""
        if factor >= 1.0:
            return pil
        return pil.resize((max(1, int(round(pil.width * factor))),
                           max(1, int(round(pil.height * factor)))), Image.BILINEAR)

    def _crop_scale_fits(self, scale: float, tw: int, th: int, W: int, H: int):
        """Template dims for *scale*, or None when the scale is not usable."""
        t_w = max(1, int(round(tw * scale)))
        t_h = max(1, int(round(th * scale)))
        if t_w < 8 or t_h < 8 or t_w > W or t_h > H:
            return None
        if (t_w * t_h) < self.CROP_MIN_TEMPLATE_AREA * (W * H):
            return None   # template too small to be a meaningful crop match
        return (t_w, t_h)

    def _crop_sweep(self, cand_work, hay) -> Tuple[bool, float]:
        """Exact scale-sweep decision at working resolution (the tuned rule)."""
        H, W = hay.shape
        th, tw = cand_work.size[::-1]
        best = None   # (peak, rect, threshold, max_trim_fraction) so far
        for scale in self.CROP_SCALE_SWEEP:
            dims = self._crop_scale_fits(scale, tw, th, W, H)
            if dims is None:
                continue
            t_w, t_h = dims
            thr = self._crop_threshold(scale)
            tpl_img = cand_work if abs(scale - 1.0) < 1e-9 else cand_work.resize((t_w, t_h), Image.BICUBIC)
            tpl = np.asarray(tpl_img, dtype=np.float64) / 255.0
            peak, rect = self._ncc_match(hay, tpl)
            frac = max((W - t_w) / float(W), (H - t_h) / float(H)) \
                if rect is not None else 0.0
            if best is None or peak > best[0]:
                best = (peak, rect, thr, frac)
            if peak >= thr and rect is not None \
                    and self._is_meaningful_trim(rect, W, H):
                return (True, frac)   # early exit: confident crop found at this scale
        if best is None or best[0] < best[2] or best[1] is None:
            return (False, 0.0)
        if not self._is_meaningful_trim(best[1], W, H):
            return (False, 0.0)
        x0, y0, x1, y1 = best[1]
        matched_frac = max(
            (W - max(0, min(x1, W) - max(0, min(x0, W)))) / float(W),
            (H - max(0, min(y1, H) - max(0, min(y0, H)))) / float(H))
        return (True, max(matched_frac, best[3]))

    def _crop_screen_sweep(self, cand_thumb, hay_thumb, threshold: float) -> bool:
        """True when any screen scale stays above *threshold* with a real trim."""
        H, W = hay_thumb.shape
        th, tw = cand_thumb.size[::-1]
        for scale in self.CROP_SCALE_SWEEP:
            dims = self._crop_scale_fits(scale, tw, th, W, H)
            if dims is None:
                continue
            t_w, t_h = dims
            tpl_img = cand_thumb if abs(scale - 1.0) < 1e-9 else cand_thumb.resize((t_w, t_h), Image.BICUBIC)
            tpl = np.asarray(tpl_img, dtype=np.float64) / 255.0
            peak, rect = self._ncc_match(hay_thumb, tpl)
            if peak >= threshold and rect is not None \
                    and self._is_meaningful_trim(rect, W, H):
                return True
        return False

    def _crop_threshold(self, scale: float) -> float:
        """NCC threshold for a given template scale.

        Matching at non-native scales loses NCC through resampling of the
        template, so allow slightly lower confidence there while keeping the
        strict requirement for native-scale matches.
        """
        if abs(scale - 1.0) < 1e-9:
            return self.CROP_MATCH_THRESHOLD
        return max(self.CROP_MATCH_THRESHOLD - 0.05, 0.85)

    @staticmethod
    def _is_meaningful_trim(rect, hay_w: int, hay_h: int) -> bool:
        """True if the matched region leaves a real margin on at least one axis."""
        x0, y0, x1, y1 = rect
        trimmed_x = (hay_w - max(0, min(x1, hay_w) - max(0, min(x0, hay_w)))) / float(hay_w)
        trimmed_y = (hay_h - max(0, min(y1, hay_h) - max(0, min(y0, hay_h)))) / float(hay_h)
        return max(trimmed_x, trimmed_y) >= QualityAssessor.CROP_MIN_TRIM_FRACTION

    @staticmethod
    def _ncc_match(hay: np.ndarray, tpl: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """Normalized cross-correlation of ``tpl`` over ``hay`` via FFT.

        Returns (best_peak, best_rect) where rect is (x0, y0, x1, y1).

        Fast path: OpenCV ``TM_CCOEFF_NORMED`` on float32 finds the same peak
        as the float64 SciPy convolution below (agreement ~1e-6 on photo-like
        content, identical argmax) roughly 10x faster, which matters because
        the O(k^2) pair gate calls this up to 6 scales per pair. The constant-
        template guard runs FIRST so degenerate inputs behave exactly like the
        SciPy path (cv2 would report 1.0 for a featureless template); on any
        OpenCV error the SciPy convolution is used instead.
        """
        t = tpl - tpl.mean()
        template_energy = float(np.sum(t * t))
        if template_energy < 1e-9:
            return (0.0, None)   # constant template: no information

        # --- Fast path: OpenCV NCC (see comment above) ------------------------
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            try:
                sm = cv2.matchTemplate(hay.astype(np.float32, copy=False),
                                       tpl.astype(np.float32, copy=False),
                                       cv2.TM_CCOEFF_NORMED)
                if sm.size:
                    # Flat windows can yield NaN/inf in the normalized map; treat
                    # them as "no correlation", matching the SciPy denom gate.
                    sm = np.nan_to_num(sm, nan=0.0, posinf=0.0, neginf=0.0)
                    iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
                    peak = float(sm[iy, ix])
                    rect = (int(ix), int(iy),
                            int(ix) + tpl.shape[1], int(iy) + tpl.shape[0])
                    return (peak, rect)
            except cv2.error:
                pass                       # OpenCV refuses this input: SciPy below

        # --- Fallback/reference: float64 FFT convolution -----------------------
        num = fftconvolve(hay, t[::-1, ::-1], mode="valid")
        ones = np.ones(tpl.shape)
        s1 = fftconvolve(hay, ones, mode="valid")
        s2 = fftconvolve(hay * hay, ones, mode="valid")
        n = tpl.size
        var = s2 - (s1 * s1) / n
        denom = np.sqrt(np.maximum(var, 0.0)) * math.sqrt(template_energy)
        with np.errstate(divide="ignore", invalid="ignore"):
            ncc = np.where(denom > 1e-9, num / denom, 0.0)
        idx = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
        peak = float(ncc[idx])
        rect = (idx[1], idx[0], idx[1] + tpl.shape[1], idx[0] + tpl.shape[0])
        return (peak, rect)
