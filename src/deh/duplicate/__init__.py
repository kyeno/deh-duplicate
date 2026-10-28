# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""The "find & quarantine similar images" feature.

Public surface
--------------
* :mod:`deh.duplicate.pipeline` — ``DuplicateConfig`` + ``DuplicatePipeline``
  (the composition root).
* :mod:`deh.duplicate.features` — ``FeatureExtractor`` (deep features).
* :mod:`deh.duplicate.hashing` — ``HashComparator`` (perceptual hashes).
* :mod:`deh.duplicate.comparator` — ``DeepSimilarityComparator``.
* :mod:`deh.duplicate.quality` — ``QualityAssessor``.
* :mod:`deh.duplicate.quarantine` — ``QuarantineManager``.
* :mod:`deh.duplicate.identical` — ``IdenticalFileResolver``.

Intentionally **no re-exports here**: importing this package does not pull
torch or Qt transitively. Import what you need from the concrete module, or
use the aliases in :mod:`deh`.
"""
