# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Quarantine Manager Module

Moves or deletes 'losing' duplicate/similar images. In move mode (default)
losers go into a dedicated _quarantine subdirectory within the scan root,
preserving original path structure; in delete mode they are unlinked after a
safety check that each resolved path stays under the scan root.
"""

import shutil
from pathlib import Path
from typing import Callable, List, Optional

from ..core.terminal import notice


class QuarantineManager:
    """Manages moving (or deleting) similar-image losers."""

    QUARANTINE_DIRNAME = "_quarantine"
    VALID_ACTIONS = ("move", "delete")

    def __init__(self, root_dir: str, log_cb: Optional[Callable[[str], None]] = None):
        """Initialize the manager for one scan root.

        Args:
            root_dir: Directory being scanned; nothing outside it is touched.
            log_cb: Receives each human-readable output line (the operation
                header, per-file results, errors). Default: the
                ``core.terminal`` notice channel -- the CLI wants them on
                stdout, an embedder routes them wherever its UI shows them
                (review 5.3).
        """
        self.root_dir = Path(root_dir).resolve()
        self.quarantine_root = self.root_dir / self.QUARANTINE_DIRNAME
        self._log_cb = log_cb

    def _log(self, text: str) -> None:
        """Route one output line to the caller's ``log_cb``, or to the terminal."""
        if self._log_cb is not None:
            self._log_cb(text)
        else:
            notice(text)

    @staticmethod
    def count_key(action: str) -> str:
        """Result-dict key holding the processed-file count for ``action``."""
        return "moved" if action == "move" else "deleted"

    def quarantine(self, loser_paths: List[str], action: str = "move",
                   dry_run: bool = False,
                   progress_cb: Optional[Callable[[int, int], None]] = None) -> dict:
        """Move (or delete) all specified files.

        Args:
            loser_paths: List of absolute file paths to process.
            action: ``"move"`` (default) moves losers into _quarantine while
                preserving directory structure; ``"delete"`` unlinks them after
                verifying each resolved path stays under the scan root.
            dry_run: If True, only report what would happen without touching anything.
            progress_cb: Receives ``(done, total)`` after each processed file,
                for callers driving a progress bar.

        Returns:
            Dictionary with operation results including counts and any errors.
        """
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Unknown action {action!r}; expected one of {self.VALID_ACTIONS}")
        count_key = self.count_key(action)
        if not loser_paths:
            return {count_key: 0, "errors": [], "dry_run": dry_run}

        header_verb = {"move": "Would move", "delete": "Would delete"}[action] \
            if dry_run else {"move": "Moving", "delete": "Deleting"}[action]
        target = f" to {self.quarantine_root}" if action == "move" else ""
        self._log(f"\n[Quarantine] {header_verb} {len(loser_paths)} image(s){target}")

        processed_count = 0
        total = len(loser_paths)
        errors = []

        def _advance() -> None:
            """Count one processed file and report it to a progress bar, if any."""
            nonlocal processed_count
            processed_count += 1
            if progress_cb is not None:
                progress_cb(processed_count, total)

        for src_path in loser_paths:
            try:
                src = Path(src_path).resolve()

                # Safety guard (both modes): only touch files under the scan root.
                rel_path = src.relative_to(self.root_dir)
                if rel_path.parts[0] == self.QUARANTINE_DIRNAME:
                    raise ValueError("source is already inside _quarantine")

                if action == "move":
                    # Destination preserves directory structure within quarantine
                    dest = self.quarantine_root / rel_path

                    if dry_run:
                        self._log(f"  [DRY-RUN] {src.name} -> {dest}")
                        _advance()
                        continue

                    # Reject a symlinked quarantine root or any symlinked parent
                    # that would redirect the move outside the scan root.
                    if not dest.parent.resolve().is_relative_to(self.quarantine_root):
                        raise ValueError("quarantine destination escapes _quarantine")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.parent.resolve().is_relative_to(self.quarantine_root):
                        raise ValueError("quarantine destination escapes _quarantine")

                    # Handle filename collisions (shouldn't happen but be safe)
                    final_dest = dest
                    counter = 1
                    while final_dest.exists() or final_dest.is_symlink():
                        stem = dest.stem
                        suffix = dest.suffix
                        final_dest = dest.with_name(f"{stem}_{counter}{suffix}")
                        counter += 1

                    shutil.move(str(src), str(final_dest))
                    self._log(f"  Moved: {rel_path}")
                else:   # delete
                    if dry_run:
                        self._log(f"  [DRY-RUN] would delete {src.name}")
                        _advance()
                        continue

                    src.unlink()
                    self._log(f"  Deleted: {rel_path}")
                _advance()

            except Exception as e:
                error_msg = f"Failed to {action} {src_path}: {e}"
                self._log(f"  ERROR: {error_msg}")
                errors.append(error_msg)

        result = {
            count_key: processed_count,
            "errors": errors,
            "dry_run": dry_run,
            "quarantine_dir": str(self.quarantine_root)
        }

        if not dry_run and processed_count > 0:
            done_verb = "moved" if action == "move" else "deleted"
            self._log(f"\n[Quarantine] Done. {processed_count} image(s) {done_verb}.")
            if action == "move":
                self._log(f"[Quarantine] Quarantine location: {self.quarantine_root}")

        return result
