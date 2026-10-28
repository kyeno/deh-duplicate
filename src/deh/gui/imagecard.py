# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""The shared image-card widget used by the review dialogs.

A card shows a thumbnail plus filename / score / resolution / filesize and
supports live state styling (winner / loser / excluded) and left/right-click
callbacks routed to the owning dialog.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

from ..core.imageio import image_metadata
from .theme import DARK_THEME_CSS


class ImageCard(QFrame):
    """Single image card: thumbnail + filename/subpath/score/resolution/filesize."""

    def __init__(self, path: str, is_winner: bool, score=None, parent=None,
                 scale: float = 1.0, display_name: str | None = None,
                 on_select=None, on_exclude=None, note: str | None = None):
        super().__init__(parent)
        self._path_key = path   # EXACT value received; echoed back verbatim by click callbacks
        self.path = Path(path)
        self.is_winner = is_winner
        self.score = score
        self.note_text = (note or "").strip() or None   # optional scorer-reason line
        s = max(0.5, float(scale))
        self.scale = s
        # Exactly one visual treatment for winners; every other element stays borderless.
        self.setObjectName("winnerCard" if is_winner else "loserCard")
        self._state = "winner" if is_winner else "loser"
        self._on_select_cb = on_select      # dialog callback for left-clicks
        self._on_exclude_cb = on_exclude    # ...and right-clicks
        self._full_pixmap = None            # full-resolution source, fitted on resize
        self.score_label = None             # exists only when a score was given
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet(DARK_THEME_CSS)   # forced dark mode + single winner outline

        layout = QVBoxLayout(self)
        layout.setContentsMargins(int(8 * s), int(8 * s), int(8 * s), int(8 * s))
        layout.setSpacing(int(6 * s))

        # --- image area (responsive: fills available space, re-fits on resize) ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(int(320 * s), int(240 * s))
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.image_label)

        # --- filename / relative-subpath label ---
        show_name = display_name or self.path.name
        if len(show_name) > 50:
            show_name = "..." + show_name[-47:]
        self.display_text = show_name
        fname_label = QLabel(show_name)
        fname_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fname_label.setFont(QFont("monospace", max(10, round(11 * s))))
        fname_label.setToolTip(str(self.path))
        layout.addWidget(fname_label)

        # --- score label (prominent when provided; color via dark-theme CSS) ---
        if score is not None:
            score_text = f"score: {score:.3f}"
            if is_winner:
                score_text += " \u2605"
            slabel = QLabel(score_text)
            slabel.setObjectName("scoreWinner" if is_winner else "scoreLoser")
            slabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            slabel.setFont(QFont("sans-serif", max(11, round(12 * s))))
            layout.addWidget(slabel)
            self.score_label = slabel       # kept so set_state() can move the star live

        # --- resolution + file size (always shown to aid comparison) ---
        res_text, size_text = image_metadata(self.path)
        self.resolution_text = res_text
        self.filesize_text = size_text
        meta_label = QLabel(f"{res_text}  \u00b7  {size_text}")
        meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta_label.setFont(QFont("monospace", max(9, round(10 * s))))
        layout.addWidget(meta_label)

        # --- optional scorer-reason line (e.g. "cropped variant of X -- full-frame
        # image preferred") so the dialog can explain why this card is framed even
        # when its score is lower than another member's; omitted entirely otherwise.
        if self.note_text:
            note_label = QLabel(self.note_text)
            note_label.setObjectName("cardNote")
            note_label.setWordWrap(True)
            note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            note_label.setToolTip(self.note_text)
            layout.addWidget(note_label)
            self.note_label = note_label

    def set_image_pixmap(self, pixmap: QPixmap):
        """Store full-resolution source and fit it into the current image area."""
        self._full_pixmap = pixmap
        self._fit_image()

    def _fit_image(self):
        pm = self._full_pixmap
        if pm is None or pm.isNull():
            return
        w = max(self.image_label.size().width(), self.image_label.minimumWidth())
        h = max(self.image_label.size().height(), self.image_label.minimumHeight())
        scaled = pm.scaled(w, h,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    def mousePressEvent(self, event):
        """Route clicks on the card (or its child labels) to dialog callbacks.

        Left button  -> select this image as the group keeper.
        Right button -> exclude / restore it from the group.
        Child QLabels do not consume button events, so they bubble up here and
        every part of the card stays clickable even when loading failed.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            cb = self._on_select_cb
            if cb is not None:
                cb(self._path_key)
        elif event.button() == Qt.MouseButton.RightButton:
            cb = self._on_exclude_cb
            if cb is not None:
                cb(self._path_key)
        else:
            super().mousePressEvent(event)
            return
        event.accept()

    def set_state(self, state: str):
        """Apply a visual state ('winner' | 'loser' | 'excluded').

        Swaps the objectName used by the dark-theme CSS: red for 'excluded',
        neutral for 'loser'. 'winner' marks the keeper with the star on its score label
        plus its note line, and carries no colour of its own - the green keeper
        design belongs to the keeper box (set_keeper), which the arrows move.
        Re-polishes so the new rule applies immediately.
        """
        self._state = state
        object_names = {"winner": "winnerCard",
                        "excluded": "excludedCard"}
        self.setObjectName(object_names.get(state, "loserCard"))

        slabel = self.score_label
        if slabel is not None and self.score is not None:
            text = f"score: {self.score:.3f}" + (" \u2605" if state == "winner" else "")
            slabel.setText(text)
            slabel.setObjectName("scoreWinner" if state == "winner" else "scoreLoser")
        self._repolish()

    def set_highlighted(self, on: bool) -> None:
        """Mark this card as the keyboard cursor (border-only outline via CSS).

        The card grid moves a cursor with the arrow keys; the include/exclude
        tint underneath stays readable.
        """

        self.setProperty("highlighted", bool(on))
        self._repolish()

    def set_keeper(self, on: bool) -> None:
        """Move the keeper box onto this card (duplicate review).

        The box IS the winner design - green border and green tint - so the arrow
        keys move the "this is the keeper" look from card to card, and Enter
        confirms whatever the box is sitting on. The confirmed keeper itself keeps
        only its star and its note line; when the box lands on an excluded card
        its border burns brighter red while the card's own red tint stays put.
        """
        self.setProperty("keeper", bool(on))
        self._repolish()

    def _repolish(self) -> None:
        """Re-apply the stylesheet after an objectName / property change."""
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()
