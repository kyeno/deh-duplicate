# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Shared, feature-agnostic primitives for deh-duplicate.

Everything in this package is reusable building-block code with no
feature-specific logic: image scanning, clustering mechanics, device
resolution, perceptual hashing and image I/O. The ``deh.duplicate`` feature
imports from here instead of duplicating these primitives.


Import submodules directly (e.g. ``from deh.core.scanner import
ImageScanner``). This package intentionally does **not** re-export them at
the top level so that ``import deh`` stays light (no torch import).
"""
