# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Duplicate-approval GUI: review a group of similar images before quarantining.

The dialog shows every candidate side-by-side. The keeper box - the full winner
design, green border and green tint - opens on the quality-assessed winner and the
arrow keys move it; Enter confirms the image under the box as the keeper, closing
the dialog and quarantining the rest. Enter on an excluded image is a strict noop
(Space restores it first). The confirmed keeper carries only its star and its note
line, so the green look is free to move; on an excluded image the box burns
brighter red over the card's own red tint. Space excludes / restores the image
under the box: dropping the keeper takes it out of the group and hands the keeper
role to the runner-up, so the group is never keeper-less while a keeper-eligible
image remains. Esc noops the whole group. Mouse: left-click keeps, right-click
excludes.
"""


from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

from PIL import Image

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from .._meta import APP_NAME
from ..config import DEFAULT as DEFAULT_CONFIG
from .app import center_on_screen, ensure_app, enlarge_to_screen, pil_to_qt_image
from .dialog import CardGridDialog
from .imagecard import ImageCard


class Decision(Enum):
    APPROVED = "approved"
    NOOP = "noop"
    QUIT = "quit"


class GroupApprovalDialog(CardGridDialog):
    """Modal dialog showing a group of similar images for approval/noop decision."""

    def __init__(self, title: str, paths: List[str], winners: List[str],
                 losers: List[str], scores: dict, parent=None, scale: float = 1.0,
                 group_number: Optional[int] = None, total_groups: Optional[int] = None,
                 notes: Optional[dict] = None):
        super().__init__(title, scale=scale, parent=parent)
        # Keep the historical object name (some tooling / tests key off it).
        self.setObjectName("approvalDlg")
        # Canonicalize every input to a plain string so state keys, card lookups and the
        # returned partition always agree — callers may pass str OR pathlib.Path objects
        # (ImageScanner.scan returns resolved PosixPath instances).
        paths = [str(p) for p in paths]
        winners = [str(w) for w in winners]
        losers = [str(l) for l in losers]
        if scores is not None:
            scores = {str(k): v for k, v in scores.items()}
        # Optional per-path scorer-reason strings; rendered as an extra line on each card.
        self.notes = ({str(k): v for k, v in notes.items() if v}
                      if notes else {})
        # Optional "Group X of Y" progression for the top header (set by show_group_approval);
        # when absent the legacy count-based wording is used instead.
        self.group_number = group_number
        self.total_groups = total_groups

        self.paths = paths
        self.winners_set = set(winners)
        self.losers_set = set(losers)
        self.scores = scores
        self.decision = None

        # --- interactive selection state (mouse- or keyboard-driven) ---
        # Every path gets exactly one of 'winner' | 'loser'; 'excluded' is added by
        # right-clicks / Space presses. With no mouse/keyboard interaction the
        # partition below reproduces the original winners/losers lists verbatim
        # (same membership AND order), so approving an unmodified dialog quarantines
        # exactly what it did before this feature.
        self.state = {}
        for p in list(winners) + list(losers):
            if p not in self.state:
                self.state[p] = "winner" if p in self.winners_set else "loser"
        for p in paths:                       # defensive: anything unclassified -> loser
            self.state.setdefault(p, "loser")
        self._base_order = [p for p in list(winners) + list(losers) if p in self.state] \
                           + [p for p in paths if p not in self.state]

        # Keyboard cursor position (index into self.paths); starts on the initial
        # winner. Arrows move it; Space applies right-click semantics there; mouse
        # clicks park it on the clicked card.

        self.cursor = next((i for i, p in enumerate(self.paths)
                            if self.state.get(p) == "winner"), 0)

        self._build_ui()

    def _build_ui(self):
        if self.group_number is not None and self.total_groups is not None:
            header_text = f"Group {self.group_number} of {self.total_groups}"
        else:
            header_text = (f"Group with {len(self.paths)} image(s). "
                           f"{len(self.winners_set)} winner(s), {len(self.losers_set)} to quarantine.")
        self.header_label.setText(header_text)

        n_images = len(self.paths)
        if n_images <= 2:
            cols = n_images
        elif n_images <= 4:
            cols = 3
        else:
            cols = 4
        # Remember the column count so ↑/↓ can jump a full row when navigating by keyboard.
        self.set_grid_columns(cols)

        # Common ancestor directory -> show only outstanding subpath/name per card.
        try:
            base_dir = Path(os.path.commonpath([str(p) for p in self.paths]))
            if base_dir.is_file():
                base_dir = base_dir.parent
        except (ValueError, OSError):
            base_dir = None

        def _display_name(p):
            if base_dir is not None:
                rel = os.path.relpath(str(p), str(base_dir))
                if rel and rel != ".":
                    return rel
            return Path(p).name

        for i, path in enumerate(self.paths):
            is_winner = path in self.winners_set
            score = self.scores.get(path)
            card = ImageCard(path, is_winner, score, scale=self.scale,
                             display_name=_display_name(path),
                             on_select=lambda p: self._on_card_selected(p),
                             on_exclude=lambda p: self._on_card_excluded(p),
                             note=self.notes.get(path))
            row = i // cols
            col = i % cols
            self.grid.addWidget(card, row, col)
            setattr(self, f"_card_{i}", card)

        self.instr_label.setText(
            "[Enter] Confirm the keeper \u2014 the image in the green box; quarantines the rest    "
            "[Esc/Backspace] Noop this group (keep all)\n"
            "[Space] Exclude / restore the image in the box \u2014 dropping the keeper hands it to the runner-up    "
            "[\u2190 \u2192 \u2191 \u2193] Move the keeper box\n"
            "Enter on a red (excluded) image does nothing \u2014 Space restores it    "
            "[Left-click] keep \u00b7 [Right-click] exclude")

        # Populate each card with its image (must happen after all cards exist).
        self._load_all_images()

        # Ensure the dialog holds keyboard focus from the start (arrow-key navigation).
        self.setFocus()
        self._refresh_cursor()

    def _load_all_images(self):
        """Load all images in the group into their respective cards.

        Prefers direct QPixmap(path) loading via Qt's native image plugins
        (no manual pixel-buffer juggling). Falls back to PIL conversion for
        formats Qt cannot decode natively.
        """
        for i, path in enumerate(self.paths):
            card = getattr(self, f"_card_{i}")
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                # Fallback: load via PIL then convert through an in-memory buffer
                try:
                    pil_img = Image.open(path).convert('RGB')
                    pixmap, _size = pil_to_qt_image(pil_img)
                except Exception as e:
                    print(f"[GUI] Warning: Could not load image {path}: {e}")
                    card.image_label.setText("(failed to load)")
                    continue
            if pixmap.isNull():
                print(f"[GUI] Warning: Empty/invalid image at {path}")
                card.image_label.setText("(failed to load)")
                continue
            card.set_image_pixmap(pixmap)

    # ------------------------------------------------------------------ selection
    def _on_card_selected(self, path: str):
        """Left-click handler: make `path` the sole keeper of this group."""
        msg = self.select_winner(path)
        if msg is not None:
            self.status_label.setText(msg)
            self.status_label.setStyleSheet("color: #35e07d; font-weight: bold;")
        self._park_cursor_on(path)

    def _on_card_excluded(self, path: str):
        """Right-click / Space handler: exclude `path` from the group (or restore it)."""
        msg = self.exclude_image(path)
        if msg is not None:
            color = "#ff9f43" if "Restored" in msg else "#e0454a"
            self.status_label.setText(msg)
            self.status_label.setStyleSheet(f"color: {color};")
        self._park_cursor_on(path)

    def _move_cursor(self, delta: int):
        """Arrow-key navigation: move the keyboard cursor ``delta`` slots in grid order.

        ``±1`` steps sideways between neighbours; ``±_grid_cols`` jumps a full row
        up/down. Wraps around the whole grid and deliberately does NOT skip excluded
        cards — landing on one never restores it (only Space/right-click do), and being
        able to target an excluded card is what makes keyboard-only restoration possible.
        Moving the cursor changes nothing about keeper/exclusion state.
        """
        n = len(self.paths)
        if n == 0:
            return
        step = abs(delta) or 1
        sign = 1 if delta > 0 else -1
        self.cursor = (self.cursor + sign * step) % n
        self._refresh_cursor()

    def _park_cursor_on(self, path: str):
        """Park the keyboard cursor on `path`'s card (mouse clicks).

        The keeper box rides on the cursor, so a left-click moves the whole winner
        design onto the clicked card and a right-click parks it on the red card.
        """
        try:
            self.cursor = self.paths.index(path)
        except ValueError:
            return
        self._refresh_cursor()

    def _refresh_cursor(self):
        """Put the keeper box on the cursor card and keep it inside the viewport.

        The box is the full winner design (green border + green tint), brighter red
        when it lands on an excluded card; the cursor position drives both the
        keeper box and the shared cursor outline.
        """
        for i in range(len(self.paths)):
            card = getattr(self, f"_card_{i}")
            card.set_highlighted(i == self.cursor)
            card.set_keeper(i == self.cursor)
        if 0 <= self.cursor < len(self.paths):
            self.scroll.ensureWidgetVisible(getattr(self, f"_card_{self.cursor}"))

    def select_winner(self, path: str):
        """Make `path` THE winner: demote any current winner(s) to loser and clear
        its exclusion. Selecting an already-selected winner is a strict no-op.

        Returns a status message for the dialog's status line, or None when nothing
        changed (already selected).
        """
        if self.state.get(path) == "winner":
            return None                      # already selected -> no-op
        affected = [p for p, st in list(self.state.items())
                    if st == "winner" and p != path]
        for p in affected:
            self.state[p] = "loser"      # deselect original winner(s)
        self.state[path] = "winner"      # also un-excludes it if it had been excluded
        for p in affected + [path]:
            self._refresh_card(p)
        name = Path(path).name
        return f"\u2605 Selected {name} as keeper \u2014 previous selection replaced."

    def _keeper_successor(self, exclude: str) -> Optional[str]:
        """The best keeper-eligible card left once `exclude` leaves the group.

        Keeper-eligible = not excluded. Ranked by the scorer's display score
        (highest first) with ties broken by the original winners-then-losers
        order, so dropping the scored keeper hands the role to the runner-up the
        QualityAssessor already ranked. ``None`` when the group has nothing left.
        """
        candidates = [p for p in self._base_order
                      if p != exclude and self.state.get(p) != "excluded"]
        if not candidates:
            return None
        return max(candidates,
                   key=lambda p: (self.scores.get(p) is not None,
                                  self.scores.get(p) or 0.0))

    def exclude_image(self, path: str):
        """Space / right-click semantics: remove `path` from the group entirely
        (neither kept nor quarantined), shown with a red border; repeat to restore it.

        The keeper IS excludable — the box is the keeper, so dropping the card
        under it takes that image out of the group and hands the keeper role to
        the best remaining candidate (:meth:`_keeper_successor`); the group is
        never left keeper-less while a keeper-eligible image remains. The box
        stays on the card the user acted on — it follows the user, not the role —
        so Space never approves anything: Enter on the red card is still a noop.
        Excluding the last keeper-eligible image leaves the group with no keeper
        at all: Enter does nothing and Esc keeps every image in place.

        Returns a status message.
        """
        cur = self.state.get(path)
        name = Path(path).name
        if cur == "excluded":
            self.state[path] = "loser"
            self._refresh_card(path)
            return f"\u21ba Restored {name} to the group (will be quarantined unless selected)."
        self.state[path] = "excluded"
        self._refresh_card(path)
        if cur != "winner":
            return f"\u2715 Excluded {name} from this group \u2014 will neither be kept nor quarantined."
        successor = self._keeper_successor(exclude=path)
        if successor is None:
            return (f"\u2715 Excluded {name} \u2014 no keeper left in this group. "
                    f"Enter does nothing; Esc keeps all {len(self.paths)} image(s).")
        self.state[successor] = "winner"
        self._refresh_card(successor)
        return (f"\u2715 Excluded {name} \u2014 keeper moved to {Path(successor).name}: "
                f"move the box there and press Enter to keep it.")

    def _refresh_card(self, path: str):
        """Re-apply visual state for one card (the 'Group X of Y' header stays put)."""
        for i in range(len(self.paths)):
            if self.paths[i] == path:
                getattr(self, f"_card_{i}").set_state(self.state[path])
                break

    def final_partition(self):
        """Return (winners, losers, excluded) reflecting all clicks so far.

        Order follows the original winners-then-losers input order, so an unmodified
        dialog yields exactly the lists QualityAssessor produced. The three lists are
        always a disjoint partition of every path in the group.
        """
        winners = [p for p in self._base_order if self.state.get(p) == "winner"]
        losers = [p for p in self._base_order if self.state.get(p) == "loser"]
        excluded = [p for p in self._base_order if self.state.get(p) == "excluded"]
        return winners, losers, excluded

    def keyPressEvent(self, event):
        """Enter keeps the image under the cursor and approves; Esc/Backspace noop;
        arrows move the keyboard cursor; Space applies right-click semantics at the
        cursor (exclude/restore).

        Enter on an excluded card is a strict noop: the keeper box stays exactly
        where it is (burning brighter red on that card) and nothing about the
        partition changes. Silently un-excluding a card the user deliberately
        dropped would make the keeper key destructive, so Space has to restore it
        first. (A left-click on an excluded card still promotes it: pointing at it
        with the mouse is an explicit request.)
        """
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if not self.paths:
                return
            path = self.paths[self.cursor]
            if self.state.get(path) == "excluded":
                name = Path(path).name
                self.status_label.setText(
                    f"\u2715 {name} is excluded \u2014 press Space to restore it, "
                    f"then Enter to keep it.")
                self.status_label.setStyleSheet("color: #e0454a; font-weight: bold;")
                return
            self.select_winner(path)
            self.decision = Decision.APPROVED
            self.status_label.setText("\u2713 Approved \u2014 losers will be quarantined")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.accept()
        elif key == Qt.Key.Key_Space and self.paths:
            # Keyboard twin of a right-click on the card under the cursor.
            self._on_card_excluded(self.paths[self.cursor])
        elif key in (Qt.Key.Key_Escape, Qt.Key.Key_Backspace):
            self.decision = Decision.NOOP
            self.status_label.setText("✗ Noop'd — all images kept in place")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.reject()
        elif key == Qt.Key.Key_Left:
            self._move_cursor(-1)
        elif key == Qt.Key.Key_Right:
            self._move_cursor(1)
        elif key == Qt.Key.Key_Up:
            self._move_cursor(-self._grid_cols)
        elif key == Qt.Key.Key_Down:
            self._move_cursor(self._grid_cols)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window-manager close (X button click).

        Distinguished from keyboard Escape by routing through this handler
        instead of keyPressEvent. If the user has not already decided
        (Enter/Esc), treat closing the window as quitting the whole app.
        """
        if self.decision is None:
            self.decision = Decision.QUIT
        super().closeEvent(event)


def show_group_approval(paths: List[str], winners: List[str],
                        losers: List[str], scores: dict, group_index: int,
                        scale: Optional[float] = None,
                        total_groups: Optional[int] = None, notes: Optional[dict] = None):
    """Display the group approval dialog and return the user's outcome.

    Args:
        paths: All image paths in this similar group.
        winners: Paths determined as highest quality (to keep).
        losers: Paths to quarantine.
        scores: Dict mapping path → quality score.
        group_index: Index of this group for display purposes.
        total_groups: Optional total number of similar groups found; when given the
            dialog's top header reads exactly "Group X of Y".
        notes: Optional dict mapping path → short reason string explaining that
            member's selection (e.g. why a lower-scoring image was kept over a
            higher-scoring one); rendered on each card when present.

    Returns:
        Tuple ``(decision, final_winners, final_losers, final_excluded)`` where
        ``decision`` is the Decision enum (APPROVED or NOOP; QUIT exits the app).
        The three lists reflect any mouse- or keyboard-driven changes made in the
        dialog — with no interaction they are exactly the original ``winners``/
        ``losers`` and an empty exclusion list. Excluded images are neither kept
        nor quarantined.
    """
    # ensure_app() pins the QApplication process-wide (gui/app.py): sip
    # destroys the app with the last Python reference, and the dialog's first
    # QWidget would then abort the process.
    #
    # A failed review dialog must not silently authorize filesystem changes.
    try:
        ensure_app()
    except Exception as e:
        raise RuntimeError(
            f"Could not start the review dialog ({e}). "
            "Use --no-gui to explicitly approve groups automatically."
        ) from e

    title = f"{APP_NAME} — Group {group_index + 1} Approval"
    # Resolve the GUI scale (env-driven; the explicit arg overrides for tests/flexibility).
    try:
        sc = float(scale if scale is not None
                   else DEFAULT_CONFIG.env_float("GUI_SCALE", 1.0))
    except (TypeError, ValueError):
        sc = 1.0
    sc = max(0.5, min(sc, 4.0))   # sanity clamp to avoid absurd sizes

    dlg = GroupApprovalDialog(title, paths, winners, losers, scores, scale=sc,
                              group_number=group_index + 1, total_groups=total_groups,
                              notes=notes)
    enlarge_to_screen(dlg)      # always open large on the primary monitor...
    center_on_screen(dlg)       # ...and centered on it (unconditional placement)

    try:
        dlg.exec()
    except KeyboardInterrupt:
        print("\n[GUI] User interrupted — exiting.")
        sys.exit(0)

    # Read the terminal decision tracked on the dialog itself (robust to how
    # the window was actually dismissed), plus any mouse-driven re-selection.
    if dlg.decision == Decision.QUIT:
        print("\n[GUI] Window closed — exiting application.")
        sys.exit(0)
    final_winners, final_losers, final_excluded = dlg.final_partition()
    if dlg.decision == Decision.APPROVED:
        return (Decision.APPROVED, final_winners, final_losers, final_excluded)
    else:
        return (Decision.NOOP, final_winners, final_losers, final_excluded)
