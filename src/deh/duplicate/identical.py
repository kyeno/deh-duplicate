# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Binary-Identical File Resolver Module

Detects byte-for-byte identical files within a scan and collapses each duplicate
set down to a single surviving copy *before* any similarity comparison happens.

Unlike perceptual / deep similarity (which groups visually similar images), this pass
only collapses files whose raw bytes are exactly the same, so exact duplicates never
consume hashing or feature-extraction budget nor show up as separate members of a
similarity group. It runs "in front" of every other phase.

Survivor selection: within an identical set the DEEPEST file in the directory tree
(most path components relative to the scan root) is kept; all shallower copies are
returned for quarantine/skipping. Depth ties fall back to the lexicographically
smallest absolute path so the choice is deterministic regardless of input order.
"""

import hashlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..core.terminal import notice


class IdenticalFileResolver:
    """Finds byte-identical files and partitions them into keepers vs redundant copies."""

    # Read size when hashing full contents (bounds memory on large files).
    _CHUNK = 1024 * 1024  # 1 MiB

    def __init__(self, root_dir, log_cb: Optional[Callable[[str], None]] = None):
        self.root_dir = Path(root_dir).resolve()
        self._log_cb = log_cb

    def _log(self, text: str) -> None:
        """Send one warning line to the caller's ``log_cb`` (default: terminal)."""
        if self._log_cb is not None:
            self._log_cb(text)
        else:
            notice(text)

    # -- detection -----------------------------------------------------------
    def group(self, image_paths: Iterable[Path]) -> List[List[Path]]:
        """Return groups (>1 member each) whose paths share identical bytes.

        Two-stage filter: first bucket by file size (cheap), then compute a SHA-256
        digest only for buckets that contain more than one candidate. Files with a
        unique size can never have an exact duplicate and are therefore never hashed.
        """
        candidates = [Path(p) for p in image_paths]

        # Stage 1: bucket by size; singletons cannot be duplicates.
        by_size: Dict[int, List[Path]] = {}
        for p in candidates:
            try:
                size = p.stat().st_size
            except OSError as e:
                self._log(f"[Identical] Warning: could not stat {p}: {e}")
                continue
            by_size.setdefault(size, []).append(p)

        multi_sizes = {size: ps for size, ps in by_size.items() if len(ps) > 1}

        # Stage 2: within same-size buckets, split by full-content digest.
        by_digest: Dict[Tuple[int, str], List[Path]] = {}
        for size, paths in multi_sizes.items():
            digests: Dict[str, List[Path]] = {}
            for p in paths:
                try:
                    digest = self._sha256(p)
                except OSError as e:
                    self._log(f"[Identical] Warning: could not hash {p}: {e}")
                    continue
                digests.setdefault(digest, []).append(p)
            for digest, group in digests.items():
                if len(group) > 1:
                    by_digest[(size, digest)] = group

        return list(by_digest.values())

    def _sha256(self, path: Path) -> str:
        """Full-file SHA-256 hex digest, read in bounded chunks."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(self._CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- survivor selection --------------------------------------------------
    def depth_of(self, path: Path) -> int:
        """Directory levels below the scan root (a file directly under root == 0)."""
        try:
            rel = Path(path).resolve().relative_to(self.root_dir)
        except ValueError:
            rel = Path(path)
        return max(len(rel.parts) - 1, 0)   # the filename itself counts as one part

    def keeper_for(self, group: List[Path]) -> Path:
        """Pick the surviving copy from an identical set.

        The deepest file wins; equal depths resolve to the lexicographically smallest
        absolute path so the result is deterministic and independent of input order.
        """
        ranked = sorted(group, key=lambda p: (-self.depth_of(p), str(Path(p))))
        return Path(ranked[0])

    def resolve(self, image_paths: Iterable[Path]) -> Tuple[List[Path], List[Path]]:
        """Partition a pool into ``(keepers, redundant)`` covering only duplicate files.

        Together the two lists contain exactly the members of every byte-identical
        group found in ``image_paths`` (non-duplicate files appear in neither list):
        ``keepers`` holds the single survivor per group and ``redundant`` holds the
        copies that should be quarantined (--move) or skipped from comparison (dry-run).
        """
        keepers: List[Path] = []
        redundant: List[Path] = []
        for grp in self.group(image_paths):
            keeper = self.keeper_for(grp)
            keepers.append(keeper)
            redundant.extend([p for p in grp if p != keeper])
        return keepers, redundant
