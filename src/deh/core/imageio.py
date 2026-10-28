# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Pure image/byte helpers (no Qt, no torch).

Small formatting/metadata utilities shared by the GUI cards and reports.
Kept in ``core`` because they depend only on the standard library and PIL,
so they can be used (and tested) without a display or a Qt runtime.

``sample_image_pixels`` is the corpus half of the resource planner
(``core.machine``): physical size is what drives every per-image buffer
estimate, so stages sample the corpus instead of trusting a constant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image


def human_bytes(num) -> str:
    """Format a byte count as a compact human-readable string ('512 B', '6.2 MB')."""
    try:
        num = float(num)
    except (TypeError, ValueError):
        return "\u2014"   # em dash placeholder on failure
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while num >= 1024 and i < len(units) - 1:
        num /= 1024.0
        i += 1
    if i == 0:
        return f"{int(round(num))} B"
    return f"{num:.1f} {units[i]}"


def image_metadata(path) -> tuple[str, str]:
    """Return ``(resolution_str, filesize_str)`` for *path*.

    Each part falls back to an em-dash placeholder if it cannot be determined
    (unreadable file, missing size, ...).
    """
    res = "\u2014"
    try:
        with Image.open(str(path)) as im:
            w, h = im.size
        res = f"{w} \u00d7 {h} px"
    except (OSError, ValueError):      # unreadable, truncated or unidentifiable image
        pass
    size = "\u2014"
    try:
        size = human_bytes(os.path.getsize(str(path)))
    except OSError:                    # vanished or unreadable file
        pass
    return res, size

@dataclass(frozen=True)
class ImageSample:
    """Dimension sample of a corpus: median/p90 megapixels + failure count."""

    count: int
    median_px: float                 # megapixels (median of the sample)
    p90_px: float
    failed: int = 0
    sampled: int = 0


def _nearest_rank(sorted_values, q: float) -> float:
    """Nearest-rank quantile of an already-sorted sequence."""
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = min(len(sorted_values) - 1, max(0, idx))
    return float(sorted_values[idx])


def sample_image_pixels(paths, max_sample: int = 64) -> ImageSample:
    """Median/p90 megapixels over a *strided* sample, reading headers only.

    The stride (not ``paths[:N]``) matters: corpora are usually sorted by
    shoot session, so the first N files can be all portraits while the corpus

    also holds 6000x4000 originals. ``Image.open(...).size`` does not decode
    pixels (~0.08 ms/path on a warm page cache), so sampling 64 files costs
    single-digit milliseconds. Unreadable files are counted in ``failed`` and
    excluded from the statistics; an empty/failed sample returns zeros so the
    caller can fall back to a conservative constant.
    """
    items = list(paths)
    total = len(items)
    if total == 0:
        return ImageSample(0, 0.0, 0.0)
    if total <= max_sample:
        picked = items
    else:
        stride = total / float(max_sample)
        picked = [items[min(total - 1, int(i * stride))]
                  for i in range(max_sample)]
    px, failed = [], 0
    for path in picked:
        try:
            with Image.open(str(path)) as im:
                w, h = im.size
            px.append((w * h) / 1e6)
        except (OSError, ValueError):      # unreadable image: counted as a failure
            failed += 1
    if not px:
        return ImageSample(total, 0.0, 0.0, failed, len(picked))
    px.sort()
    return ImageSample(count=total, median_px=_nearest_rank(px, 0.5),
                       p90_px=_nearest_rank(px, 0.9), failed=failed,
                       sampled=len(picked))
