# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Image Scanner Module

Recursively scans a directory for image files, filtering by extension
and skipping hidden/non-image files.
"""

import os
from pathlib import Path
from typing import List


class ImageScanner:
    """Scans directories recursively for image files."""

    # Supported image extensions (lowercase)
    IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'
    }

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.is_dir():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")

    def scan(self) -> List[Path]:
        """Scan the root directory recursively for image files.

        Returns:
            List of absolute paths to discovered image files.
        """
        images = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip hidden directories and quarantine folder during initial scan
            dirnames[:] = [d for d in dirnames
                          if not d.startswith('.') and d != '_quarantine']

            for filename in filenames:
                filepath = Path(dirpath) / filename
                if self._is_image(filepath):
                    images.append(filepath.resolve())

        return sorted(images)

    def _is_image(self, path: Path) -> bool:
        """Check if a file is an image based on its extension."""
        return path.suffix.lower() in self.IMAGE_EXTENSIONS
