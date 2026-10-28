# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""deh-duplicate — image duplicate-detection toolkit.

Bundles a reusable image-comparison *library* (the core modules below) plus a
thin command-line driver under :mod:`deh.cli`. Other projects can simply::

    from deh.core.scanner import ImageScanner
    from deh.duplicate.quality import QualityAssessor

Bootstrap (model cache dir + optional ``.env``) is configured here so it runs
before any torch import regardless of how the package or its entrypoints are launched.
"""

import os

from .config import ProjectConfig

# Resolve every project path from a single, CWD-independent source of truth
# (see :mod:`deh.config`). This runs before any torch import so the model
# cache location is fixed no matter how the package or its entrypoints launch.
_CFG = ProjectConfig.load()

# Configure local model cache directory BEFORE any torch/torchvision imports.
os.environ.setdefault("TORCH_HOME", str(_CFG.models_dir))

# Load project-level .env if present. ``override=True`` makes the file the
# source of truth for both the console entry point and the shell launcher.

try:
    from dotenv import load_dotenv as _load_dotenv

    if _CFG.env_file.exists():
        _load_dotenv(_CFG.env_file, override=True)
except ImportError:  # pragma: no cover - python-dotenv optional at runtime
    pass

# --- Public identity -------------------------------------------------------
from ._meta import APP_NAME, __version__

# --- Reusable library API --------------------------------------------------
from .core.scanner import ImageScanner
from .duplicate.features import FeatureExtractor
from .duplicate.hashing import HashComparator
from .duplicate.comparator import DeepSimilarityComparator
from .duplicate.quality import QualityAssessor
from .duplicate.quarantine import QuarantineManager
from .duplicate.identical import IdenticalFileResolver

#: The library API re-exported here. Listed so the re-exports are explicit
#: (and so a linter can tell a facade from an unused import).
__all__ = [
    "APP_NAME", "__version__",
    "ImageScanner",
    "FeatureExtractor",
    "HashComparator",
    "DeepSimilarityComparator",
    "QualityAssessor",
    "QuarantineManager",
    "IdenticalFileResolver",
]
