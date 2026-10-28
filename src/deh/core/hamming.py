# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Hamming-distance math for packed uint64 perceptual hashes.

Used by ``duplicate.hashing`` to compare 64-bit image hashes. Keeping the
popcount logic here lets callers use either a full matrix or passing pairs.

- ``hamming_matrix_uint64`` - full (n, n) uint8 distance matrix (GPU or CPU).
- ``hamming_hits_uint64``   - passing (i, j) pairs for a row block (GPU or CPU).
- ``_pack_uint64``          - pack hex strings / ``imagehash`` values to uint64.
- ``_split_16bit``          - split uint64 words into four 16-bit lanes.
- ``_hamming_rows``         - Hamming distances for a row block vs all columns.

Vectorized, chunked computation keeps even 50k-image corpora in the
seconds range (50k x 50k uint8 = 2.5 GB unchunked). The hits API
never materialises the full matrix: on CUDA only the passing pairs leave the
device, so ``duplicate.hashing`` stays flat at any corpus size.
"""

import numpy as np

try:
    import torch
except ImportError:  # torch is a hard dep of the project; guard anyway
    torch = None

#: Transient budget for the CUDA index tensor [sub, n, 8] int32 built per row
#: sub-block - bounds memory so a huge corpus never materialises an (n, n).
_GPU_INDEX_BUDGET_BYTES = 256 * 1024 * 1024


def _pack_uint64(hashes) -> np.ndarray:
    """Pack hash values (hex strings or ``imagehash.ImageHash``) into uint64."""
    return np.array([np.uint64(int(str(h), 16)) for h in hashes], dtype=np.uint64)


def _split_16bit(arr: np.ndarray):
    """Split packed uint64 words into four 16-bit lanes for the popcount LUT."""
    mask = np.uint64(0xFFFF)
    return ((arr & mask).astype(np.uint64),
            (arr >> np.uint64(16)) & mask,
            (arr >> np.uint64(32)) & mask,
            (arr >> np.uint64(48)) & mask)


def _hamming_rows(lanes, start: int, end: int, lut: np.ndarray) -> np.ndarray:
    """Hamming distances from row block [start:end) against ALL columns.

    Returns an (end-start, n) uint8 matrix; chunked so memory stays flat.
    """
    lo, hi, mid, top = lanes
    return (lut[lo[start:end, None] ^ lo[None, :]]
            + lut[hi[start:end, None] ^ hi[None, :]]
            + lut[mid[start:end, None] ^ mid[None, :]]
            + lut[top[start:end, None] ^ top[None, :]])


def _popcount_lut_16() -> np.ndarray:
    """16-bit popcount lookup table (the numpy kernel's LUT)."""
    return np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def _row_hits(dist: np.ndarray, start: int, max_distance: int) -> np.ndarray:
    """Global (row, col) pairs from a distance block, keeping col > row only."""
    rows, cols = np.nonzero(dist <= max_distance)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    rows = rows + start
    keep = cols > rows
    return np.stack([rows[keep], cols[keep]], axis=1).astype(np.int64, copy=False)


def _cuda_hashes(hashes: np.ndarray):
    """Byte-view uint64 hashes + 256-entry popcount LUT, both on CUDA."""
    n = len(hashes)
    byte_view = hashes.view(np.uint8).reshape(n, 8)
    t = torch.from_numpy(byte_view.copy()).to("cuda")
    lut = torch.tensor([bin(i).count("1") for i in range(256)],
                       dtype=torch.uint8, device="cuda")
    return t, lut


def _cuda_block_dist(t, lut, start: int, end: int):
    """[end-start, n] uint8 Hamming distances for a row block, on GPU."""
    xored = t[start:end, None, :] ^ t[None, :, :]      # [c, n, 8] uint8
    # Advanced indexing needs an integer-dtype index tensor; cast from uint8.
    return lut[xored.to(torch.int32)].sum(dim=2, dtype=torch.uint8)


def _hamming_matrix_gpu(hashes: np.ndarray, chunk: int) -> np.ndarray:
    """CUDA path: byte-view uint64 hashes, XOR + popcount LUT on GPU."""
    n = len(hashes)
    t, lut = _cuda_hashes(hashes)
    dist = torch.zeros((n, n), dtype=torch.uint8, device="cuda")
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist[start:end] = _cuda_block_dist(t, lut, start, end)
    return dist.cpu().numpy()


def _hamming_row_hits_gpu(hashes: np.ndarray, start: int, end: int,
                          max_distance: int) -> np.ndarray:
    """CUDA row-block hits: only the passing pairs leave the device."""
    n = len(hashes)
    t, lut = _cuda_hashes(hashes)
    cols_idx = torch.arange(n, device="cuda")
    # Row sub-blocks keep the transient [sub, n, 8] int32 index tensor bounded.
    sub = max(1, min(end - start,
                     int(_GPU_INDEX_BUDGET_BYTES // max(1, n * 8 * 4))))
    rows_out, cols_out = [], []
    for s in range(start, end, sub):
        e = min(s + sub, end)
        dist = _cuda_block_dist(t, lut, s, e)
        mask = (dist <= max_distance) & (
            cols_idx[None, :] > torch.arange(s, e, device="cuda")[:, None])
        rows, cols = torch.nonzero(mask, as_tuple=True)
        if rows.numel():
            rows_out.append((rows + s).cpu().numpy())
            cols_out.append(cols.cpu().numpy())
    if not rows_out:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([np.concatenate(rows_out), np.concatenate(cols_out)],
                    axis=1).astype(np.int64, copy=False)


def hamming_matrix_uint64(hashes: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Pairwise Hamming distance matrix for packed uint64 hashes.

    Returns an (n, n) uint8 matrix. Uses CUDA (XOR + popcount lookup) when
    torch+CUDA are available, chunked so memory stays flat; numpy fallback
    otherwise (16-bit popcount LUT applied to both halves of each 64-bit word).
    """
    n = len(hashes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if n == 1:
        return np.zeros((1, 1), dtype=np.uint8)

    from .device import pick_device
    if pick_device().type == "cuda":
        return _hamming_matrix_gpu(np.ascontiguousarray(hashes), chunk)

    pop_lut = _popcount_lut_16()
    dist = np.zeros((n, n), dtype=np.uint8)
    lanes = _split_16bit(hashes)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist[start:end] = _hamming_rows(lanes, start, end, pop_lut)
    return dist


def hamming_hits_uint64(hashes: np.ndarray, start: int, end: int,
                        max_distance: int, device=None) -> np.ndarray:
    """(row, col) pairs within Hamming distance inside row block [start:end).

    Rows [start:end) are compared against all columns, and only pairs with
    col > row come back, so callers can union the hits of several hash tables
    (pHash OR dHash) without double counting. The CUDA path computes the block
    with the byte-XOR popcount LUT and transfers just the passing pairs back;
    the numpy fallback uses the 16-bit lane LUT. Returns an (k, 2) int64 array
    of global indices.
    """
    n = len(hashes)
    if n < 2 or start >= end or end <= 0:
        return np.empty((0, 2), dtype=np.int64)
    start, end = max(0, start), min(n, end)
    hashes = np.ascontiguousarray(hashes)

    if device is None:
        from .device import pick_device
        device = pick_device()
    if torch is not None and torch.device(device).type == "cuda":
        return _hamming_row_hits_gpu(hashes, start, end, max_distance)

    lanes = _split_16bit(hashes)
    dist = _hamming_rows(lanes, start, end, _popcount_lut_16())
    return _row_hits(dist, start, max_distance)
