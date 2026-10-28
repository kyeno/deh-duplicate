# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""PyQt6 GUI package: shared infrastructure + the duplicate feature dialog.

- :mod:`deh.gui.app`        — QApplication bootstrap, window geometry, image conversion
- :mod:`deh.gui.theme`      — dark-theme stylesheet + palette constants
- :mod:`deh.gui.imagecard`  — the shared image-card widget
- :mod:`deh.gui.dialog`     — CardGridDialog base (header + scroll grid + status)
- :mod:`deh.gui.duplicate`  — GroupApprovalDialog + show_group_approval (duplicate feature)

Importing this package requires PyQt6. The duplicate pipeline imports it for
group approval; the core modules do not depend on Qt.
"""

from .app import (center_on_screen, compute_center_rect, ensure_app,
                  enlarge_to_screen, pil_to_qt_image, select_target_screen,
                  target_screen_geometry, window_control_flags, _resolve_flag)
from .theme import DARK_THEME_CSS
from .imagecard import ImageCard
from .dialog import CardGridDialog
from .duplicate import Decision, GroupApprovalDialog, show_group_approval
from ..core.imageio import human_bytes, image_metadata

__all__ = [
    # app
    "ensure_app", "window_control_flags", "target_screen_geometry",
    "select_target_screen", "enlarge_to_screen", "center_on_screen",
    "compute_center_rect", "pil_to_qt_image", "_resolve_flag",
    # theme
    "DARK_THEME_CSS",
    # widgets
    "ImageCard", "CardGridDialog",
    # duplicate feature
    "Decision", "GroupApprovalDialog", "show_group_approval",
    # pure helpers (re-exported for convenience)
    "human_bytes", "image_metadata",
]
