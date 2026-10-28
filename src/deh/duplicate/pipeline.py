# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""DuplicatePipeline — the "find & quarantine similar images" flow.

The class owns the whole pipeline::

    scan → identical-dedupe → similarity (hash / deep / hybrid)
         → quality scoring → GUI approval loop → quarantine

Design notes
------------
* **Dependency injection:** every collaborator is a constructor argument with
  a sensible default; ``DuplicatePipeline(config)`` needs nothing else. Tests
  can stub any collaborator (e.g. ``approval_dialog``) without touching the
  filesystem or Qt.
* **Composition root:** the pipeline instantiates the collaborators it needs
  per phase; no singletons and no module-level mutable state.
* **Config:** :class:`DuplicateConfig` is a frozen dataclass assembled by the
  CLI from ``argparse`` + environment variables; the pipeline itself never
  reads ``os.environ``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from ..core.device import begin_phase
from ..core.scanner import ImageScanner
from .comparator import DeepSimilarityComparator
from .features import FeatureExtractor
from .hashing import HashComparator
from .identical import IdenticalFileResolver
from .quality import QualityAssessor
from .quarantine import QuarantineManager
from ..gui import Decision, show_group_approval


@dataclass(frozen=True)
class DuplicateConfig:
    """Everything the pipeline needs to run.

    Attributes
    ----------
    directory:
        Root directory to scan recursively for images.
    threshold:
        Similarity threshold as a decimal fraction. Defaults to 0.75 for
        perceptual hashes in hash/hybrid modes and 0.82 for deep mode.
        Explicit values override the default in every mode.
    mode:
        ``"hash"``, ``"deep"``, or ``"hybrid"``.
    action:
        What happens to losers once approved: ``"move"`` (default) quarantines
        them into _quarantine; ``"delete"`` removes them permanently.
    dry_run:
        When True nothing on disk changes regardless of ``action``.
    verbose:
        Detailed output flag (reserved for per-image scores).
    use_gui:
        Show the interactive approval dialog per group.
    identical_dedupe:
        Run the byte-for-byte identical pre-pass first.
    deep_threshold:
        Deep feature threshold in hybrid mode (default 0.82). The CLI uses
        the same default for deep mode, which stores it in ``threshold``.
    """

    directory: str
    threshold: float | None = None
    mode: str = "hybrid"
    action: str = "move"
    dry_run: bool = False
    verbose: bool = False
    use_gui: bool = True
    identical_dedupe: bool = True
    deep_threshold: float = 0.82

    def __post_init__(self) -> None:
        if self.threshold is None:
            object.__setattr__(self, "threshold",
                               0.82 if self.mode == "deep" else 0.75)


class DuplicatePipeline:
    """Run the duplicate-detection flow for a configured directory."""

    def __init__(
        self,
        config: DuplicateConfig,
        *,
        scanner_cls: type = ImageScanner,
        hash_cls: type = HashComparator,
        extractor_cls: type = FeatureExtractor,
        comparator_cls: type = DeepSimilarityComparator,
        assessor_cls: type = QualityAssessor,
        manager_cls: type = QuarantineManager,
        resolver_cls: type = IdenticalFileResolver,
        approval_dialog: Callable[..., Tuple] = show_group_approval,
        out: Callable[..., None] = print,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        self.config = config
        self.scanner_cls = scanner_cls
        self.hash_cls = hash_cls
        self.extractor_cls = extractor_cls
        self.comparator_cls = comparator_cls
        self.assessor_cls = assessor_cls
        self.manager_cls = manager_cls
        self.resolver_cls = resolver_cls
        self.approval_dialog = approval_dialog
        self._out = out
        self._progress_cb = progress_cb

    # ------------------------------------------------------------------ #
    # Orchestration                                                       #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Execute the full pipeline (output identical to the old ``main()``)."""
        cfg, out = self.config, self._out

        # Phase 1: Scan for images
        begin_phase("scan")
        out("[Phase 1] Scanning directory for images...")
        scanner = self.scanner_cls(cfg.directory)
        image_paths = scanner.scan()
        out(f"[Phase 1] Found {len(image_paths)} image(s)")

        if len(image_paths) < 2:
            out("\nNeed at least 2 images to compare. Nothing to do.")
            return

        # Shared state for both the identical-file pre-pass below and the
        # similarity phase later, so progress survives a mid-cycle quit (each
        # quarantine happens as soon as it's approved).
        # Library collaborators get the pipeline's own output sink: their
        # progress and notices arrive through the same channel the pipeline
        # uses (review 5.3), so an embedder that passed a custom ``out``
        # captures everything, and DEH_QUIET silences the default one.
        manager = self.manager_cls(cfg.directory, log_cb=out)
        total_moved = 0

        # Phase 1b — runs IN FRONT of every other comparison pass (enabled by
        # default; disable with --no-identical-dedupe / DEH_DUPLICATE_IDENTICAL_DEDUPE=false).
        # Collapses byte-for-byte identical files: the DEEPEST copy in each set
        # survives; shallower copies are moved/deleted per the configured action
        # (or simply dropped from the comparison pool in dry-run), so they never
        # consume hash/feature budget. comparison_pool is unchanged when the
        # pre-pass is off or finds no duplicates.
        comparison_pool = [str(p) for p in image_paths]
        if cfg.identical_dedupe:
            pool, moved, errors = self._identical_prepass(image_paths, manager)
            total_moved += moved
            self._raise_on_errors(errors)
            if pool is None:  # fewer than 2 images left → nothing more to do
                return
            comparison_pool = [str(p) for p in pool]

        similar_groups = self._find_similar_groups(comparison_pool)
        if not similar_groups:
            out("\nNo similar image groups found above the threshold.")
            out("Try lowering the threshold (e.g., --threshold 60).")
            return

        self._approve_and_quarantine(
            similar_groups, len(image_paths), manager, total_moved
        )

    # ------------------------------------------------------------------ #
    # Phases                                                              #
    # ------------------------------------------------------------------ #

    def _identical_prepass(
        self, image_paths: Sequence[str], manager: QuarantineManager
    ) -> Tuple[Optional[List[str]], int, List[str]]:
        """Collapse byte-for-byte identical files before any similarity work.

        Returns ``(comparison_pool, moved, errors)``. ``comparison_pool`` is
        ``None`` when fewer than 2 images remain and the run must stop.
        """
        cfg, out = self.config, self._out

        begin_phase("identical-dedupe")
        out("\n[Phase 1b] Checking for byte-for-byte identical duplicates...")
        resolver = self.resolver_cls(cfg.directory, log_cb=out)
        keepers, redundant = resolver.resolve(image_paths)
        if not redundant:
            out("[Phase 1b] No byte-for-byte identical duplicates found.")
            return list(image_paths), 0, []

        out(
            f"[Phase 1b] Found {len(redundant)} exact duplicate(s) across "
            f"{len(keepers)} group(s); keeping the deepest copy of each."
        )
        result = manager.quarantine(
            [str(p) for p in redundant], action=cfg.action, dry_run=cfg.dry_run
        )
        moved = result.get(manager.count_key(cfg.action), 0)
        errors = list(result.get("errors", []))

        drop = {Path(p).resolve() for p in redundant}
        pool = [p for p in image_paths if Path(p).resolve() not in drop]
        out(f"[Phase 1b] Comparison pool now contains {len(pool)} image(s)")
        if len(pool) < 2:
            out(
                "\nFewer than 2 images remain after removing exact duplicates; "
                "nothing more to compare."
            )
            return None, moved, errors
        return pool, moved, errors

    def _find_similar_groups(
        self, comparison_pool: Sequence[str]
    ) -> List[List[str]]:
        """Phase 2 & 3: find similar groups using the configured method.

        Returns the list of similar groups (possibly empty).
        """
        cfg, out = self.config, self._out
        mode = cfg.mode

        if mode == "hash":
            begin_phase("hashing")
            out("\n[Phase 2] Computing perceptual hashes and finding similar groups...")
            hash_kwargs = {}
            if self._progress_cb is not None:
                hash_kwargs["progress_cb"] = lambda done, total, phase: self._progress_cb(
                    "hash_images" if phase == "hash" else "hash_pairs", done, total
                )
            hash_comp = self.hash_cls(threshold=cfg.threshold, log_cb=out,
                                      **hash_kwargs)
            return self._timed_call(
                "Hash-based grouping", hash_comp.find_similar_groups, comparison_pool
            )

        if mode in ("deep", "hybrid"):
            begin_phase("deep-features")
            out("\n[Phase 2] Extracting deep learning features from images...")
            extractor = self.extractor_cls(log_cb=out)
            extract_kwargs = {}
            if self._progress_cb is not None:
                extract_kwargs["progress_cb"] = lambda done, total: self._progress_cb(
                    "features", done, total)
                self._progress_cb("features", 0, len(comparison_pool))
            features = self._timed_call(
                "Deep feature extraction",
                extractor.extract_batch,
                comparison_pool,
                batch_size=32,
                **extract_kwargs,
            )
            out(f"[Phase 2] Successfully extracted features for {len(features)} image(s)")

            if not features:
                raise RuntimeError("No image features were extracted; comparison could not run.")
            missing = set(comparison_pool) - set(features)
            if missing:
                raise RuntimeError(
                    f"Feature extraction failed for {len(missing)} image(s); "
                    "comparison is incomplete."
                )

            begin_phase("similarity")
            out("\n[Phase 3] Comparing images and finding similar groups...")
            compare_threshold = cfg.threshold if mode == "deep" else cfg.deep_threshold
            compare_kwargs = {}
            if self._progress_cb is not None:
                compare_kwargs["progress_cb"] = lambda done, total: self._progress_cb(
                    "deep_pairs", done, total)
            comparator = self.comparator_cls(threshold=compare_threshold, log_cb=out,
                                             **compare_kwargs)
            deep_groups = self._timed_call(
                "Pairwise similarity comparison", comparator.find_similar_groups, features
            )
            if mode == "deep" or not deep_groups:
                return deep_groups

            # CNN embeddings describe image content, so two different photos
            # of the same subject can score highly. In the default hybrid mode
            # require perceptual agreement as well. The deep groups are already
            # mutually similar and disjoint; refining each one with hashes
            # keeps both guarantees and never joins deep-unrelated images.
            begin_phase("hashing")
            out("\n[Phase 3b] Verifying deep matches with perceptual hashes...")
            verified = []
            total_images = sum(len(group) for group in deep_groups)
            total_pairs = sum(len(group) * (len(group) - 1) // 2
                              for group in deep_groups)
            done_images = done_pairs = 0
            for group in deep_groups:
                hash_kwargs = {}
                if self._progress_cb is not None:
                    def on_hash_progress(done, _total, phase,
                                         image_offset=done_images,
                                         pair_offset=done_pairs):
                        if phase == "hash":
                            self._progress_cb("hash_images", image_offset + done,
                                              total_images)
                        else:
                            self._progress_cb("hash_pairs", pair_offset + done,
                                              total_pairs)

                    hash_kwargs["progress_cb"] = on_hash_progress
                hash_comp = self.hash_cls(threshold=cfg.threshold, log_cb=out,
                                          **hash_kwargs)
                verified.extend(self._timed_call(
                    "Perceptual hash verification",
                    hash_comp.find_similar_groups, group,
                ))
                done_images += len(group)
                done_pairs += len(group) * (len(group) - 1) // 2
                if self._progress_cb is not None:
                    self._progress_cb("hash_images", done_images, total_images)
                    self._progress_cb("hash_pairs", done_pairs, total_pairs)
            return verified

        raise ValueError(f"Unknown comparison mode: {mode!r}")

    def _approve_and_quarantine(
        self,
        similar_groups: List[List[str]],
        image_count: int,
        manager: QuarantineManager,
        total_moved: int,
    ) -> None:
        """Phase 4: score quality, approve via GUI, quarantine immediately.

        Each group is quarantined the moment it is approved so progress
        survives a mid-cycle quit (closing any dialog exits the app).
        """
        cfg, out = self.config, self._out

        begin_phase("quality+review")
        out("\n[Phase 4] Evaluating quality within each similar group...")
        assessor = self.assessor_cls(log_cb=out)

        all_winners: List[str] = []
        all_losers: List[str] = []
        all_excluded: List[str] = []  # right-clicked out of groups: neither kept nor quarantined
        approved_count = 0
        nooped_count = 0

        for i, group in enumerate(similar_groups):
            # evaluate_with_notes also returns a per-path reason string (e.g. why a
            # lower-scoring image was kept over a higher-scoring one: crop rule or
            # near-tie resolution), which the GUI renders on each card. It fills
            # scores_out with the display score of every member as it scores, so
            # the metric set runs once per image instead of twice.
            scores = {}
            winners, losers, notes = assessor.evaluate_with_notes(group, scores_out=scores)

            # Fallback for assessors that do not fill scores_out (e.g. test
            # stubs): compute whatever is missing, 0.0 when it cannot be scored.
            for p in group:
                if p not in scores:
                    try:
                        scores[p] = assessor.score_image(p)
                    except Exception:
                        scores[p] = 0.0

            out(f"\n  Group {i + 1} ({len(group)} images):")
            out(f"    Keep:   {[Path(p).name for p in winners]}")
            if losers:
                out(f"    Quarantine: {[Path(p).name for p in losers]}")
            for p, note in notes.items():
                if note:
                    out(f"    Note:     {Path(p).name}: {note}")

            decision = Decision.APPROVED  # Default to approve (automatic mode)

            if cfg.use_gui and losers:
                # Show interactive dialog only when there are actual losers to
                # quarantine. Returns (decision, winners, losers, excluded); with
                # no mouse clicks the partition is exactly what QualityAssessor
                # produced above.
                decision, winners, losers, excluded = self.approval_dialog(
                    paths=group,
                    winners=winners,
                    losers=losers,
                    scores=scores,
                    group_index=i,
                    total_groups=len(similar_groups),
                    notes=notes,
                )
            else:
                excluded = []

            if decision == Decision.APPROVED:
                all_winners.extend(winners)
                all_losers.extend(losers)
                all_excluded.extend(excluded)
                approved_count += 1
                # Final outcome AFTER any mouse-driven re-selection (the Keep/Quarantine
                # lines above are only the scorer's pre-dialog suggestion).
                out("    → Approved")
                out(f"      Final keeper:       {[Path(x).name for x in winners] or []}")
                out(f"      Final quarantine:   {[Path(x).name for x in losers] or []}")
                if excluded:
                    out(f"      Excluded (in place): {[Path(x).name for x in excluded]}")
                # Act NOW (apply the configured action, or report in dry-run mode)
                # so this approval is durable even if a later dialog quits the app.
                if losers:
                    result = manager.quarantine(
                        list(losers), action=cfg.action, dry_run=cfg.dry_run
                    )
                    total_moved += result.get(manager.count_key(cfg.action), 0)
                    self._raise_on_errors(result.get("errors", []))
            else:
                nooped_count += 1
                out("    → Noop'd (all kept)")

        out("\n[Phase 4] Summary:")
        out(f"  Total similar groups: {len(similar_groups)}")
        out(f"  Groups approved:      {approved_count}")
        out(f"  Groups noop'd:        {nooped_count}")
        out(f"  Images to keep:       {len(all_winners)}")
        out(f"  Images to quarantine: {len(all_losers)}")
        if all_excluded:
            out(f"  Excluded from group:  {len(all_excluded)} (kept in place)")

        if not all_losers:
            out("\nNo images need quarantining. All best-quality versions are already retained.")
            return

        # (No separate Phase 5 anymore: each group was quarantined the moment it was approved.)

        verb = "Moved" if cfg.action == "move" else "Deleted"
        out("\n" + "=" * 60)
        out(" Operation Complete ")
        out("=" * 60)
        out(f"Images processed:          {image_count}")
        out(f"Similar groups found:      {len(similar_groups)}")
        out(f"Groups approved:           {approved_count}")
        out(f"Groups noop'd (kept all):  {nooped_count}")
        out(f"{verb + ' (or would be)':<25} {total_moved}")

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _raise_on_errors(errors: Sequence[str]) -> None:
        """Stop the run if a requested filesystem operation failed."""
        if errors:
            raise RuntimeError(
                f"{len(errors)} file operation(s) failed; first error: {errors[0]}"
            )

    def _timed_call(self, description: str, fn: Callable, *args, **kwargs):
        """Run a callable and report its wall-clock duration for transparency."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        self._out(f"      [timing] {description}: {dt:.2f}s")
        return result
