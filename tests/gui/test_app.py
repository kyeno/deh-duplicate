"""Tests for deh.gui.app: window geometry and the QApplication lifetime.

The lifetime test runs in a *fresh interpreter*. What it guards is a C-level
abort (SIGABRT, core dump), not a Python exception, so it cannot be asserted
in-process: the session ``qapp`` fixture holds a live QApplication, which
keeps the app alive and hides the bug — that is exactly why the CLI crashed
while the suite stayed green.
"""
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

from deh.gui.app import (compute_center_rect, select_target_screen,
                         target_screen_geometry)


def _make_img(path, color=(120, 40, 200), size=(240, 180)):
    """Small valid JPEG with detail, so the approval dialog has real images."""
    img = Image.new("RGB", size, color)
    px = img.load()
    for y in range(0, size[1], 8):
        for x in range(size[0]):
            px[x, y] = (255, 255, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)


class TestLayoutHelpers:
    """Pure helpers for multi-monitor centering."""

    def test_compute_center_rect_centers_dialog(self):
        x, y, w, h = compute_center_rect(0, 0, 1920, 1080, 800, 600)
        assert (w, h) == (800, 600)
        assert x == (1920 - 800) // 2   # 560
        assert y == (1080 - 600) // 2   # 240

    def test_compute_center_rect_clamps_when_larger_than_screen(self):
        x, y, w, h = compute_center_rect(0, 0, 1000, 700, 1400, 900)
        assert (w, h) == (1000, 700)   # clamped to the available area
        assert (x, y) == (0, 0)


class TestAppLifetime:
    """ensure_app() must keep the app alive when the caller drops the reference.

    PyQt6/sip destroys the C++ QApplication with its last Python reference; the
    next QWidget construction then aborts the process. Commit 6484702 deleted
    the two reference-keeping locals as "unused" (ruff F841) and broke every
    GUI entry point with them — the CLI core-dumped, the suite stayed green.
    """

    _CHILD = '''
"""Bootstrap invariant: dropping ensure_app()'s result must not kill the app."""
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QWidget

from deh.gui.app import ensure_app

ensure_app()                    # result deliberately discarded by the caller
assert QApplication.instance() is not None, "app was garbage-collected"
QWidget()                       # aborts here if the app died with its reference
print("widget OK")

import deh.gui as g


def _watchdog():
    app = QApplication.instance()
    if app is not None:
        app.exit(3)


_real_exec = g.GroupApprovalDialog.exec


def _auto_exec(self):
    """The real dialog, auto-accepted so exec() returns without a user."""
    self.decision = g.Decision.APPROVED
    QTimer.singleShot(0, self.accept)
    QTimer.singleShot(20000, _watchdog)
    return _real_exec(self)


g.GroupApprovalDialog.exec = _auto_exec
paths = ["a.jpg", "b.jpg"]
out = g.show_group_approval(paths=paths, winners=["a.jpg"], losers=["b.jpg"],
                            scores={"a.jpg": 0.9, "b.jpg": 0.4}, group_index=0)
assert out[0] == g.Decision.APPROVED, out
assert out[1] == ["a.jpg"] and out[2] == ["b.jpg"], out
print("show_group_approval OK")
'''

    def test_public_entry_points_survive_dropped_app_reference(self, tmp_path):
        """The CLI crash, headless: no app reference held, then a dialog opens."""
        _make_img(tmp_path / "a.jpg")
        _make_img(tmp_path / "b.jpg", (200, 60, 60))
        src = str(Path(__file__).resolve().parents[2] / "src")
        env = dict(os.environ, PYTHONPATH=src, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run([sys.executable, "-c", self._CHILD],
                              cwd=str(tmp_path), env=env,
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, (proc.stdout + proc.stderr)[-400:]
        assert "widget OK" in proc.stdout
        assert "show_group_approval OK" in proc.stdout


class TestTargetScreen:
    """Dialogs open on the primary monitor (the middle one on this desk)."""

    def test_primary_screen_wins(self):
        primary = QRect(1080, 320, 2560, 1600)
        left, right = QRect(0, 0, 1080, 1920), QRect(3640, 0, 1080, 1920)
        assert select_target_screen([left, right], primary) is primary

    def test_falls_back_to_first_screen_without_primary(self):
        first = QRect(0, 0, 1080, 1920)
        assert select_target_screen([first, QRect(1080, 0, 640, 480)], None) is first

    def test_no_screens_returns_none(self):
        assert select_target_screen([], None) is None

    def test_target_screen_geometry_matches_primary(self, qapp):
        primary = QApplication.instance().primaryScreen()
        assert target_screen_geometry() == primary.availableGeometry()
