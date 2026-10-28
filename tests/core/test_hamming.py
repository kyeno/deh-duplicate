"""Tests for core.hamming: full matrix and row-block hits (CPU vs CUDA)."""
import numpy as np
import pytest

from deh.core.hamming import hamming_hits_uint64, hamming_matrix_uint64


def _brute(hashes, start, end, max_distance):
    """Reference passing pairs: rows [start:end) vs later columns only."""
    return [(i, j) for i in range(start, end)
            for j in range(i + 1, len(hashes))
            if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= max_distance]


def _sorted(pairs):
    return sorted(tuple(p) for p in pairs)


class TestHits:
    def test_matches_brute_force(self):
        rng = np.random.default_rng(3)
        hashes = rng.integers(0, 2 ** 63, size=40).astype(np.uint64)
        hits = hamming_hits_uint64(hashes, 0, 40, 12, device="cpu")
        assert _sorted(hits.tolist()) == _brute(hashes, 0, 40, 12)

    def test_row_block_bounds_and_upper_triangle(self):
        hashes = np.zeros(6, dtype=np.uint64)   # identical -> every pair passes
        hits = hamming_hits_uint64(hashes, 2, 4, 0, device="cpu")
        assert _sorted(hits.tolist()) == _brute(hashes, 2, 4, 0)

    def test_hits_agree_with_matrix_threshold(self):
        rng = np.random.default_rng(7)
        hashes = rng.integers(0, 2 ** 63, size=32).astype(np.uint64)
        matrix = hamming_matrix_uint64(hashes)
        rows, cols = np.triu_indices(len(hashes), k=1)
        mask = matrix[rows, cols] <= 10
        reference = _sorted(zip(rows[mask].tolist(), cols[mask].tolist()))
        hits = hamming_hits_uint64(hashes, 0, len(hashes), 10)
        assert _sorted(hits.tolist()) == reference

    def test_cpu_gpu_agree(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        rng = np.random.default_rng(5)
        hashes = rng.integers(0, 2 ** 63, size=64).astype(np.uint64)
        cpu = hamming_hits_uint64(hashes, 0, 64, 20, device="cpu")
        gpu = hamming_hits_uint64(hashes, 0, 64, 20, device="cuda")
        assert _sorted(cpu.tolist()) == _sorted(gpu.tolist())

    def test_degenerate_inputs(self):
        assert hamming_hits_uint64(np.array([], dtype=np.uint64), 0, 0, 5,
                                   device="cpu").shape == (0, 2)
        one = np.array([0], dtype=np.uint64)
        assert hamming_hits_uint64(one, 0, 1, 5, device="cpu").shape == (0, 2)
