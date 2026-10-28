# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Perceptual Hashing Module

Uses perceptual hashes (pHash, dHash) to quickly identify near-duplicate images.
This approach is faster than deep learning feature extraction and works well for
detecting exact duplicates, crops, slight edits, and re-uploads of the same image.

Hashing runs through the shared :class:`deh.core.hashes.HashEngine` (parallel PIL
decode + batched DCT on the picked device, CUDA preferred) and the pairwise
comparison through :func:`deh.core.hamming.hamming_hits_uint64` (CUDA XOR +
popcount kernel that transfers only the passing pairs; numpy 16-bit-lane
fallback).

Grouping uses **complete-linkage** clustering: the threshold pass only seeds
candidate components (connected images), then each component is re-clustered
so that EVERY pair inside a final group stays within the Hamming threshold.
Single-linkage/union-find chaining (A~B, B~C while A!~C) used to glue whole
shoots into one oversized group; complete linkage breaks such chains into
mutually-similar groups.
"""

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..core.cluster import UnionFind, check_linkage_budget
from ..core.device import pick_device
from ..core.hamming import _pack_uint64, hamming_hits_uint64, hamming_matrix_uint64
from ..core.hashes import HASH_MIN_FREE_MB, HashEngine
from ..core.terminal import notice


class HashComparator:
    """Groups similar images based on perceptual hash proximity."""

    def __init__(self, threshold: float = 0.75, *, engine=None, machine=None,
                 device=None,
                 log_cb: Optional[Callable[[str], None]] = None,
                 progress_cb: Optional[Callable[[int, int, str], None]] = None):
        """Initialize comparator with similarity threshold.

        Args:
            threshold: Minimum similarity percentage (0.0 to 1.0). Converted
                      internally to maximum allowed Hamming distance between hashes.
            engine: Hash engine collaborator (default: shared ``HashEngine``).
            machine: Machine-state service used to size the decode pool.
            device: Torch device override (default: auto-picked per run).
            log_cb: Receives each human-readable progress line (default: the
                ``core.terminal`` notice channel). An embedder that wants the
                run quiet passes a no-op, one that wants it in its own log
                passes its own writer.
            progress_cb: Receives ``(done, total, phase)`` milestones --
                ``phase`` is ``"hash"`` for the decode/hash pass (images) and
                ``"compare"`` for the pairwise pass (comparisons) -- for callers
                that want numbers instead of prose.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        # Convert percentage threshold to max Hamming distance
        # pHash produces 64-bit hashes; lower distance = more similar
        # At 75% threshold, allow up to ~16 differing bits out of 64
        self.max_distance = int((1.0 - threshold) * 64)
        self.engine = engine or HashEngine(machine=machine)
        self._device = device
        self._log_cb = log_cb
        self._progress_cb = progress_cb
        self._log(f"[Hash] Using max Hamming distance: {self.max_distance} "
                  f"(equivalent to {threshold*100:.0f}% similarity)")

    def _log(self, text: str) -> None:
        """Route one output line to the caller's ``log_cb``, or to the terminal."""
        if self._log_cb is not None:
            self._log_cb(text)
        else:
            notice(text)

    def _milestone(self, done: int, total: int, phase: str) -> None:
        """Report a structured progress milestone, if the caller asked for one."""
        if self._progress_cb is not None:
            self._progress_cb(done, total, phase)

    def find_similar_groups(self, image_paths: List[Path]) -> List[List[str]]:
        """Find all groups of visually-similar images using perceptual hashing.

        Computes both pHash and dHash for each image, then groups images whose
        threshold-passing pairs seed candidate components, refined by
        complete-linkage clustering so every pair inside a final group stays
        within the configured Hamming distance threshold (no single-linkage
        chaining).

        Args:
            image_paths: List of paths to process.

        Returns:
            List of groups where each group contains paths of mutually-similar images.
        """
        total = len(image_paths)
        self._log(f"[Hash] Computing perceptual hashes for {total} images...")

        # CUDA preferred with a tiny VRAM gate (a hash batch needs <2 MB); the
        # engine falls back to CPU with a loud banner when the GPU is unusable.
        device = self._device or pick_device(min_free_mb=HASH_MIN_FREE_MB)
        reported = [0]

        def _on_hash_progress(done: int) -> None:
            if self._progress_cb is not None:
                self._milestone(done, total, "hash")
            # Same ~50-image cadence as the per-image loop this replaced (the
            # engine flushes in batches, so lines land on batch boundaries).
            if done >= reported[0] + 50 or done >= total:
                reported[0] = done
                self._log(f"[Hash] Hashed {min(done, total)}/{total} images")

        payloads = self.engine.compute_batch(image_paths, device,
                                             progress_cb=_on_hash_progress)
        if self._progress_cb is not None:
            # Unreadable images were still attempted; finish the scan bar and
            # retain their separate skip warnings.
            self._milestone(total, total, "hash")

        # Pairwise pass in row blocks - numpy on CPU, or the CUDA XOR+popcount
        # kernel that returns only the passing pairs, so this pass uses bounded
        # row blocks. Only pairs that pass the threshold touch the
        # Python-level union-find (either hash within max_distance triggers a
        # link; same OR semantics as before); the roots form CANDIDATE
        # components that are refined by complete-linkage clustering below,
        # which is what actually decides group membership.
        paths = [p for p in payloads
                 if "phash" in payloads[p] and "dhash" in payloads[p]]
        n = len(paths)
        # core.cluster.UnionFind is the project's single union-find (path
        # compression + union by rank); the local closure pair this replaces had
        # no rank, so its find() was O(n) worst case on chain-shaped corpora.
        uf = UnionFind(n)

        total_pairs = n * (n - 1) // 2
        self._log(f"[Hash] Comparing {n} images pairwise ({total_pairs:,} comparisons)...")
        # Progress milestones (~every 25%) so long runs don't look frozen.
        milestones = [(int(total_pairs * pct / 100), pct) for pct in (25, 50, 75)]
        milestone_idx = 0
        comparisons = 0

        if n > 1:
            packed = {name: _pack_uint64([payloads[p][name] for p in paths])
                      for name in ("phash", "dhash")}
            chunk = max(64, min(2048, n))
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                for table in packed.values():
                    hits = hamming_hits_uint64(table, start, end,
                                               self.max_distance, device=device)
                    for gi, j in hits.tolist():
                        uf.union(gi, j)

                comparisons += sum(n - 1 - i for i in range(start, end))
                while milestone_idx < len(milestones) and comparisons >= milestones[milestone_idx][0]:
                    _, pct = milestones[milestone_idx]
                    self._log(f"[Hash]   ...{pct}% compared ({comparisons:,}/{total_pairs:,})")
                    self._milestone(comparisons, total_pairs, "compare")
                    milestone_idx += 1

        # Small corpora finish inside a single row block: flush any un-emitted
        # milestones so the progress report still covers the full comparison.
        if total_pairs > 0:
            while milestone_idx < len(milestones):
                _, pct = milestones[milestone_idx]
                self._log(f"[Hash]   ...{pct}% compared ({comparisons:,}/{total_pairs:,})")
                self._milestone(comparisons, total_pairs, "compare")
                milestone_idx += 1

        self._log(f"[Hash] Completed {comparisons} pairwise hash comparisons")
        self._milestone(comparisons, total_pairs, "compare")

        # Candidate components (union-find roots) are chain-linked sets from
        # the threshold pass; refine each with complete-linkage clustering so
        # EVERY pair inside a final group stays within max_distance bits.
        # distance(i, j) = min(pHash, dHash) Hamming distance matches the OR
        # condition that seeded the components (min <= t <=> pHash <= t OR
        # dHash <= t), so linkage at the same cutoff only ever SPLITS chains --
        # it can never merge across components (every cross-component pair
        # exceeds the cutoff by construction). Large components are rejected
        # before dense refinement can exhaust RAM.
        comps = uf.groups()

        similar_groups = []
        for members in comps.values():
            if len(members) < 2:
                continue
            if len(members) == 2:
                # A two-member component exists only because the pair itself
                # passed the threshold: mutually similar by construction.
                similar_groups.append([paths[i] for i in members])
                continue
            check_linkage_budget(len(members))
            sub = np.asarray(members, dtype=np.int64)
            dist = np.minimum(hamming_matrix_uint64(packed["phash"][sub]),
                              hamming_matrix_uint64(packed["dhash"][sub]))
            condensed = squareform(dist, checks=False).astype(np.float64)
            labels = fcluster(linkage(condensed, method="complete"),
                              t=float(self.max_distance),
                              criterion="distance")
            split = {}
            for member, label in zip(members, labels, strict=True):
                split.setdefault(label, []).append(member)
            similar_groups.extend(
                [paths[i] for i in idxs]
                for idxs in split.values() if len(idxs) > 1)

        total_similar = sum(len(g) for g in similar_groups)
        self._log(f"[Hash] Found {len(similar_groups)} groups containing "
                  f"{total_similar} similar images")

        return similar_groups
