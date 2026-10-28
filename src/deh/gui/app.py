# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Shared QApplication bootstrap + window geometry + image conversion.

Everything here is used by BOTH feature dialogs (and any future GUI), so it
lives in one place instead of being duplicated across modules. Importing this
module requires PyQt6; it is only imported by GUI code (never by the core
library or the non-GUI CLI paths), so ``import deh`` stays light.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication


#: Process-wide pin for the QApplication. PyQt6/sip destroys the C++ application
#: the moment its last Python reference dies, and the next QWidget construction
#: then aborts the process ("QWidget: Must construct a QApplication before a
#: QWidget" -> SIGABRT, core dump). Call sites may discard ensure_app()'s
#: result precisely because THIS module holds the reference for the process.
#: A lint sweep that deletes the reference as "unused" breaks every GUI entry
#: point in exactly that way (commit 6484702 did); tests/gui/test_app.py guards
#: the invariant in a fresh interpreter.
_APP_KEEPALIVE: QApplication | None = None


def ensure_app() -> QApplication:
    """Return the running QApplication, creating one (Fusion style) if needed.

    The Fusion style is pre-set via argv to avoid the adwaita platform-theme
    warning during construction; the original argv is restored afterwards.
    Offscreen-safe: works with ``QT_QPA_PLATFORM=offscreen`` (tests/CI).

    The app is pinned in :data:`_APP_KEEPALIVE` before being returned, so
    callers may drop the reference safely. An app that already exists (the
    pytest ``qapp`` fixture, an embedding application) is pinned too: Qt
    allows exactly one QApplication per process, so that is the same app,
    not a second one.
    """
    global _APP_KEEPALIVE
    app = QApplication.instance()
    if app is not None:
        _APP_KEEPALIVE = app
        return app
    original_args = list(sys.argv)
    try:
        sys.argv = [sys.argv[0], "-style", "Fusion"] + original_args[1:]
        app = QApplication(sys.argv)
    finally:
        sys.argv = original_args
    _APP_KEEPALIVE = app
    return app


def _resolve_flag(name):
    """Resolve a single Qt window-flag/hint by name across possible enum namespaces.

    Tolerant of PyQt6 binding differences where these live under ``Qt.WindowFlag`` in
    some builds but ``Qt.WindowType`` (or unscoped) in others. Returns 0 if not found
    anywhere.
    """
    for ns_name in ("WindowFlag", "WindowType"):
        ns = getattr(Qt, ns_name, None)
        if ns is not None and hasattr(ns, name):
            try:
                return int(getattr(ns, name))
            except (TypeError, ValueError):   # enum member that is not int-able
                pass
    if hasattr(Qt, name):
        try:
            return int(getattr(Qt, name))
        except (TypeError, ValueError):
            pass
    return 0


def window_control_flags():
    """Build a portable 'normal resizable top-level window' flag value with minimize,
    maximize and close buttons enabled.

    Returns the combined value AS A NATIVE Qt ENUM OBJECT (not a bare int): some PyQt6
    builds require ``setWindowFlags`` to receive a ``Qt.WindowType`` instance rather than
    an integer. Using the base ``Window`` type (rather than ``Dialog``) is also what makes
    window managers such as GNOME/Adwaita render the min/max/close title-bar buttons;
    dialogs are policy-restricted to just the close button even when hints are requested.
    """
    wanted = ("Window", "WindowTitleHint", "WindowSystemMenuHint",
              "WindowCloseButtonHint", "WindowMinimizeButtonHint",
              "WindowMaximizeButtonHint")

    def member(name):
        # Plain attribute reads: the hasattr() guards above already exclude the
        # only failure mode a lookup on a Qt enum namespace has, so there is
        # nothing to catch here (a broad 'except Exception: pass' used to turn a
        # binding change into a silently missing window flag).
        for ns_name in ("WindowFlag", "WindowType"):
            ns = getattr(Qt, ns_name, None)
            if ns is not None and hasattr(ns, name):
                return getattr(ns, name)
        if hasattr(Qt, name):
            return getattr(Qt, name)
        return None

    parts = [m for m in (member(n) for n in wanted) if m is not None]
    if not parts:
        return 0
    val = parts[0]
    for m in parts[1:]:
        val = val | m
    return val

def select_target_screen(screens, primary=None):
    """Pick the screen the dialogs open on: the primary monitor, else the first one.

    Pure preference helper (no Qt calls) so the order is unit-testable without a
    display server: *primary* wins when available; headless/offscreen platforms
    that report no primary fall back to the first screen. Returns None when
    *screens* is empty.
    """
    if primary is not None:
        return primary
    return screens[0] if screens else None


def target_screen_geometry():
    """Return the available geometry of the primary monitor, or None.

    The primary screen is what Qt / ``xrandr --primary`` designates as the main
    display — on a multi-monitor desk that is the monitor dialogs should open on
    (the middle one on this workstation's three-screen layout). Every dialog
    helper below shares it, so sizing and centering always agree.

    Note: explicit placement is an X11-only request — Wayland compositors
    ignore it and position toplevels themselves (KDE's default policy looks
    centred anyway). See CONTRIBUTING.md for the GUI conventions.
    """
    try:
        app = QApplication.instance()
        screens = app.screens() if app else []
        target = select_target_screen(screens, app.primaryScreen() if app else None)
        return target.availableGeometry() if target is not None else None
    except Exception:
        return None


def compute_center_rect(sx, sy, sw, sh, dw, dh):
    """Return (x, y, w, h) centering a dw×dh dialog within screen rect (sx,sy,sw,sh).

    Clamps to the available area when the dialog is larger than the screen so it
    always stays fully visible; internal scrolling handles any overflow.
    """
    w = min(int(dw), int(sw))
    h = min(int(dh), int(sh))
    x = int(sx) + max(0, (int(sw) - w) // 2)
    y = int(sy) + max(0, (int(sh) - h) // 2)
    return x, y, w, h


def enlarge_to_screen(dialog) -> bool:
    """Open `dialog` large on the primary monitor, clamped so it never overflows it."""
    try:
        geo = target_screen_geometry()
        if geo is None:
            return False
        aw, ah = int(geo.width()), int(geo.height())
        min_w, min_h = dialog.minimumWidth(), dialog.minimumHeight()
        w = min(max(min_w, int(aw * 0.90)), aw)   # big, but never wider than the screen
        h = min(max(min_h, int(ah * 0.92)), ah)   # ...and never taller than the screen
        dialog.resize(w, h)
        return True
    except Exception as e:
        print(f"[GUI] Warning: could not enlarge dialog ({e}); keeping default size.")
        return False


def center_on_screen(dialog) -> bool:
    """Center `dialog` on the primary monitor without shrinking it below its chosen size."""
    try:
        geo = target_screen_geometry()
        if geo is None:
            return False
        dw = max(dialog.width(), dialog.sizeHint().width()) or 1
        dh = max(dialog.height(), dialog.sizeHint().height()) or 1
        x, y, w, h = compute_center_rect(int(geo.x()), int(geo.y()),
                                         int(geo.width()), int(geo.height()), dw, dh)
        dialog.setGeometry(x, y, w, h)
        return True
    except Exception as e:
        print(f"[GUI] Warning: could not center dialog ({e}); using default placement.")
        return False


def pil_to_qt_image(pil_img):
    """Convert a PIL Image to QPixmap via QImage with explicit format specification.

    Returns ``(QPixmap, (width, height))``.
    """
    img = pil_img.convert('RGBA')
    data = img.tobytes()
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg), img.size
