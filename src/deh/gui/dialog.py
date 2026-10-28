# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Base class for modal dialogs that present a grid of image cards.

Encapsulates the scaffolding the review dialogs share: a dark-themed
top-level window (with min/max/close buttons), a header label, a scrollable
card grid, an instructions line and a status line. Subclasses populate the
grid and implement their own interaction (selection, buttons, keyboard).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QDialog, QGridLayout, QLabel, QScrollArea,
                             QVBoxLayout, QWidget)

from .app import window_control_flags
from .theme import DARK_THEME_CSS


class CardGridDialog(QDialog):
    """Dark-themed dialog with a header, a scrollable card grid and a status line."""

    def __init__(self, title: str, scale: float = 1.0, parent=None):
        super().__init__(parent)
        self.scale = max(0.5, float(scale))
        self.setObjectName("cardGridDlg")
        self.setStyleSheet(DARK_THEME_CSS)   # forced dark mode across the whole dialog
        # Make it a normal top-level window (not a Dialog) so window managers such as
        # GNOME/Adwaita render the standard minimize / maximize / close buttons together.
        wf = window_control_flags()
        if wf:
            self.setWindowFlags(wf)
        self.setWindowTitle(title)
        # Small safe floor so the window always fits any display; bigness comes from the
        # initial resize below plus enlarge_to_screen clamping to the monitor area.
        self.setMinimumSize(int(min(640 * self.scale, 720)), int(min(520 * self.scale, 640)))
        open_w = int(max(900, 1100 * self.scale))
        open_h = int(max(640, 820 * self.scale))
        self.resize(open_w, open_h)

        self._main = QVBoxLayout(self)
        self._main.setContentsMargins(10, 10, 10, 10)
        self._main.setSpacing(10)

        self.header_label = QLabel("")
        self.header_label.setFont(QFont("sans-serif", 11))
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main.addWidget(self.header_label)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(8)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_host)
        # Keep keyboard focus on the dialog itself so arrow keys reach keyPressEvent
        # even though the viewport is technically focusable on some platforms/themes.
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._main.addWidget(self.scroll)

        self.instr_label = QLabel("")
        self.instr_label.setFont(QFont("sans-serif", 9))
        self.instr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instr_label.setStyleSheet("color: #c9c9c9;")
        self._main.addWidget(self.instr_label)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("sans-serif", 10, QFont.Weight.Bold))
        self._main.addWidget(self.status_label)

    def set_grid_columns(self, cols: int) -> None:
        """Remember the grid column count (used for row-jumping arrow navigation)."""
        self._grid_cols = max(1, cols)
