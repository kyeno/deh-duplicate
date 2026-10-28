"""Shared pytest configuration & fixtures for deh-duplicate.

Placing this file at the project root anchors pytest's ``rootdir`` here and lets
us put the src-layout package on ``sys.path`` so tests can do ``import deh``,
``from deh.gui import ...`` or ``from deh.cli.duplicate import parse_args``
regardless of how pytest is invoked (no editable install required).

Qt is forced to the offscreen platform *before* any widget/QApplication is
created so GUI widgets can be constructed without a display server (CI-safe).
This only affects the pytest process — it does not change how the CLI entrypoints
run interactively.
"""
import os
import sys
from pathlib import Path

# Make the src-layout package importable in tests without an editable install:
_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """A single session-wide QApplication instance required by Qt widgets."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
