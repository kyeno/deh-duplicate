# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Clustering utilities for the duplicate pipeline.

Union-find (single-linkage) over similarity edges. The duplicate comparators
apply their similarity thresholds before joining image pairs here.
"""

from .machine import machine


def check_linkage_budget(size: int) -> None:
    """Reject a component too large for complete-linkage's dense work arrays.

    The comparators build distance arrays and scipy copies a condensed array.
    Budget 48 bytes per pair, then limit that estimate to one quarter of live
    available RAM (at most 2 GiB) so one component cannot exhaust the host.
    """
    if size < 1000:
        return
    estimated_bytes = size * (size - 1) // 2 * 48
    available_mb = machine().live().ram.available_mb
    budget_bytes = min(2 * 1024**3, max(1, available_mb // 4) * 1024**2)
    if estimated_bytes > budget_bytes:
        raise RuntimeError(
            f"Similarity component of {size:,} images needs roughly "
            f"{estimated_bytes / 1024**3:.1f} GiB for complete linkage; "
            "use a stricter threshold or split the scan into smaller directories."
        )


class UnionFind:
    """Standard union-find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict:
        """Map root index -> sorted list of member indices."""
        out = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out
