# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Deep Learning Similarity Comparator Module

Compares image feature vectors using cosine similarity to identify groups
of visually similar images above a configurable threshold. Uses embeddings
extracted by pre-trained convolutional neural networks.

Supports both CPU (numpy) and GPU (torch) paths for zero-copy pipeline:
- When features are torch.Tensor on CUDA, uses torch.matmul on GPU (N1)
- When features are numpy arrays, uses numpy matmul on CPU (backward compatible)
"""

from typing import Callable, Dict, List, Optional, Union
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..core.cluster import UnionFind, check_linkage_budget
from ..core.terminal import notice

try:
    import torch
except ImportError:  # torch is a hard dep of the project; guard anyway
    torch = None


class DeepSimilarityComparator:
    """Groups similar images based on deep feature vector cosine similarity."""

    def __init__(self, threshold: float = 0.82,
                 device: Optional[Union[str, "torch.device"]] = None,
                 log_cb: Optional[Callable[[str], None]] = None,
                 progress_cb: Optional[Callable[[int, int], None]] = None):
        """Initialize comparator with similarity threshold.

        Args:
            threshold: Minimum cosine similarity (0.0 to 1.0) for images
                       to be considered similar. Default 0.82 (82%).
            device: Torch device for GPU computation. If None, auto-detects
                    from input features (uses CUDA if any feature is on CUDA).
            log_cb: Receives each human-readable progress line. Defaults to the
                ``core.terminal`` notice channel, which ``DEH_QUIET`` silences.
            progress_cb: Receives ``(completed_pairs, total_pairs)`` while
                comparing feature vectors.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        self._device = device
        self._log_cb = log_cb
        self._progress_cb = progress_cb

    def _log(self, text: str):
        """Send one progress line to the caller's ``log_cb`` (default: terminal)."""
        if self._log_cb is not None:
            self._log_cb(text)
        else:
            notice(text)

    def find_similar_groups(self, features: Dict[str, Union[np.ndarray, "torch.Tensor"]]) -> List[List[str]]:
        """Find all groups of mutually-similar images.

        Pairwise comparisons at the threshold seed candidate components
        (union-find); each component is then refined by complete-linkage
        clustering on cosine distance (1 - cosine similarity) with cutoff
        ``1 - threshold``, so every pair inside a final group is mutually
        similar -- single-linkage chaining (A~B, B~C while A!~C) cannot glue
        unrelated images into one oversized group.

        Args:
            features: Dictionary mapping path strings to normalized feature vectors.
                     Values can be numpy arrays (CPU) or torch.Tensor (GPU).

        Returns:
            List of groups, where each group is a list of image paths that are
            similar to at least one other member in the group. Groups containing
            only one image (no similarities found) are excluded.
        """
        paths = list(features.keys())
        n = len(paths)

        self._log(f"[Comparator] Comparing {n} images at threshold {self.threshold:.2f}")

        # Build adjacency structure using Union-Find for efficient grouping:
        # core.cluster.UnionFind is the project's single implementation (path
        # compression + union by rank), shared with duplicate.hashing.
        uf = UnionFind(n)

        # Determine device from features if not explicitly set
        device = self._device
        if device is None and n > 0:
            first_feat = next(iter(features.values()))
            if torch is not None and isinstance(first_feat, torch.Tensor):
                device = first_feat.device

        comparisons = 0
        total_pairs = n * (n - 1) // 2
        if self._progress_cb is not None:
            self._progress_cb(0, total_pairs)
        if n > 1:
            # Shape contract: one pooled 1-D vector per image, so the stacked
            # matrix is [n, d].
            #
            # This is asserted rather than auto-flattened on purpose. A 3-D
            # (C, H, W) feature map means the upstream extractor stopped
            # pooling; raveling it into a 62720-dim vector would silently
            # compare embeddings in a different space than the 1280-d ones
            # (and, on the GPU path, made F[start:end] @ F.mT broadcast into a
            # wrong-shaped result that returned ZERO groups). Raising turns a
            # silent quality regression into an immediate error.
            if torch is not None and device is not None and device.type == "cuda":
                # GPU path (N1): features are already on GPU as torch.Tensor
                F_list = []
                for p in paths:
                    feat = features[p]
                    if not isinstance(feat, torch.Tensor):
                        feat = torch.from_numpy(
                            np.ascontiguousarray(self._as_array(feat)))
                    F_list.append(feat.to(device))
                F = torch.stack(F_list)  # [n, d] on GPU
            else:
                # CPU path (numpy) - backward compatible. NOTE: no ravel here;
                # a non-1-D entry must reach _assert_2d and raise. Mixed input
                # (numpy rows + torch tensors, CPU or CUDA) is normalised by
                # _as_array, the module's guarded tensor normaliser.
                F = np.stack([self._as_array(features[p]) for p in paths])
            self._assert_2d(F, n)

            # Chunk size to bound peak memory: ~32 MB of float32 per row block.
            # Each row block computes a [chunk, n] similarity matrix.
            chunk = max(1, min(n, 8_000_000 // max(n, 1)))
            is_torch = torch is not None and isinstance(F, torch.Tensor)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                # F[start:end] @ F.mT -> [c, n] (GPU) / @ F.T -> [c, n] (numpy)
                S = F[start:end] @ F.mT if is_torch else F[start:end] @ F.T
                # Only the upper triangle (j > i) is inspected, so each pair is
                # counted once and unioned once.
                for local_i in range(end - start):
                    gi = start + local_i
                    row = S[local_i, gi + 1:]
                    if is_torch:
                        js = torch.nonzero(row >= self.threshold, as_tuple=True)[0]
                        js = js.tolist()
                    else:
                        js = np.flatnonzero(row >= self.threshold)
                    for j in js:
                        uf.union(gi, gi + 1 + int(j))
                    comparisons += n - 1 - gi
                    if (self._progress_cb is not None
                            and (gi % max(1, n // 100) == 0 or gi == n - 1)):
                        self._progress_cb(comparisons, total_pairs)

        self._log(f"[Comparator] Completed {comparisons} pairwise comparisons")

        # Candidate components (union-find roots) come from the threshold pass;
        # refine each with complete-linkage clustering on cosine distance so
        # every pair inside a final group is mutually similar. Cross-component
        # pairs are all below threshold by construction, so the cutoff
        # (1 - threshold) can only split chains, never merge across components.
        comps = uf.groups()

        similar_groups = []
        for members in comps.values():
            if len(members) < 2:
                continue
            if len(members) == 2:
                # A two-member component exists only because the pair itself
                # passed the threshold: mutually similar by construction.
                similar_groups.append([paths[i] for i in members])
                continue
            check_linkage_budget(len(members))
            # Plain int-list indexing works for both numpy and torch (CPU/CUDA).
            if torch is not None and isinstance(F, torch.Tensor):
                Fs = F[members]                     # [k, d] on original device
                sim = (Fs @ Fs.mT).detach().cpu().numpy()
            else:
                Fs = F[members]
                sim = Fs @ Fs.T
            dist = np.clip(1.0 - sim, 0.0, None).astype(np.float64)
            np.fill_diagonal(dist, 0.0)
            condensed = squareform(dist, checks=False)
            labels = fcluster(linkage(condensed, method="complete"),
                              t=1.0 - self.threshold,
                              criterion="distance")
            split = {}
            for member, label in zip(members, labels, strict=True):
                split.setdefault(label, []).append(member)
            similar_groups.extend(
                [paths[i] for i in idxs]
                for idxs in split.values() if len(idxs) > 1)

        total_similar = sum(len(g) for g in similar_groups)
        self._log(f"[Comparator] Found {len(similar_groups)} groups containing "
                  f"{total_similar} similar images")

        return similar_groups

    @staticmethod
    def _as_array(v) -> np.ndarray:
        """Normalise one feature entry to a numpy array (shape preserved).

        This is the module's **only** tensor normaliser, and the guarded one:
        it uses the module-level ``torch`` and tests ``torch is not None``
        first, so a host without torch takes the numpy path instead of
        raising ``AttributeError`` on ``None.Tensor``. A torch tensor is
        detached and moved to CPU first, so a CUDA feature on the numpy path
        is converted rather than failing inside ``np.asarray``.

        Shape is deliberately preserved: a non-2-D entry must reach
        :meth:`_assert_2d` and raise, not be silently ravelled into a
        wide vector in a different embedding space.
        """
        if torch is not None and isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    @staticmethod
    def _assert_2d(matrix, n: int) -> None:
        """Fail loudly if the stacked feature matrix is not [n, d].

        The whole pipeline assumes one L2-normalized vector per image. A 3-D
        (n, C, H, W) matrix means the upstream extractor stopped pooling, which
        would otherwise make the row-block matmul broadcast into a wrong-shaped
        result and quietly yield zero groups. Raising here turns a silent
        quality regression into an immediate, debuggable error.
        """
        ndim = matrix.ndim if hasattr(matrix, "ndim") else np.ndim(matrix)
        if ndim != 2:
            raise ValueError(
                "Feature matrix must be 2-D [n, d] (one pooled vector per "
                f"image), got {ndim}-D with shape {tuple(matrix.shape)}. "
                "This usually means the feature extractor is emitting an "
                "unpooled feature map; check that the model trunk ends with "
                "AdaptiveAvgPool2d(1) + Flatten(1)."
            )
        rows = matrix.shape[0]
        if rows != n:
            raise ValueError(
                f"Feature matrix has {rows} rows but {n} paths were given."
            )
