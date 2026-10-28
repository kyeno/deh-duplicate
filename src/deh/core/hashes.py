# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Perceptual-hash engine (pHash/dHash).

GPU-accelerated batched implementation: PIL decode + LANCZOS resize on CPU
(decode is inherently CPU-bound) spread across a machine-sized thread pool with
a bounded prefetch window, then DCT-II in **float64 on the picked torch device**
via a precomputed DCT matrix. The math exactly replicates ``imagehash.phash`` /
``imagehash.dhash`` (same resize, same unnormalized scipy DCT-II, same median
rule via quantile), so hashes are bit-compatible with the reference
implementation — verified by tests.

Consumer: ``duplicate.hashing.HashComparator`` (packs the hex hashes to uint64
and clusters by Hamming proximity). No module-level mutable state: DCT matrices
are instance state.
"""


from collections import deque
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from .machine import machine as default_machine
from .terminal import notice

#: Hash size producing 64-bit hashes (imagehash default).
HASH_SIZE = 8
#: The hash stage itself needs <2 MB of VRAM, but the CUDA context needs a few
#: hundred MB - 1 GB is enough while still refusing a truly starved GPU (the
#: project-wide 2 GB gate exists for the model-loading stages).
HASH_MIN_FREE_MB = 1024


class HashEngine:
    """Batched pHash/dHash engine, bit-compatible with imagehash."""

    def __init__(self, machine=None, log_cb: Optional[Callable[[str], None]] = None):
        """Args:
            log_cb: Receives each human-readable output line. Default: the
                ``core.terminal`` notice channel, which ``DEH_QUIET`` silences.
        """
        self._log_cb = log_cb
        self.machine = machine or default_machine()
        # (device, n) -> float64 DCT-II matrix (unnormalized, as scipy).
        self._dct_mat = {}

    def _log(self, text: str) -> None:
        """Route one output line to the caller's ``log_cb``, or to the terminal."""
        if self._log_cb is not None:
            self._log_cb(text)
        else:
            notice(text)

    def _dct_matrix(self, device: torch.device, n: int) -> torch.Tensor:
        """Unnormalized DCT-II matrix M with (M @ X) == scipy.fftpack.dct(X, axis=0)."""
        key = (str(device), n)
        if key not in self._dct_mat:
            k = torch.arange(n, dtype=torch.float64, device=device)
            c = torch.arange(n, dtype=torch.float64, device=device)
            m = 2.0 * torch.cos(torch.pi * (k[:, None] * (2 * c[None, :] + 1)) / (2 * n))
            self._dct_mat[key] = m
        return self._dct_mat[key]

    @staticmethod
    def _pack_bits(bits_cpu) -> str:
        """Pack a flat bool array (row-major) into a hex string (MSB first)."""
        return np.packbits(np.asarray(bits_cpu, dtype=bool)).tobytes().hex()

    def compute_batch(self, paths, device: torch.device, hash_size: int = HASH_SIZE,
                      progress_cb=None, decode_workers: int | None = None,
                      batch_size: int = 64, prefetch_batches: int = 3) -> dict:
        """Compute pHash/dHash for many images: parallel decode, batched DCT.

        Decode+resize runs on a machine-sized thread pool with a bounded
        prefetch window (PIL releases the GIL in its C decode/resize kernels);
        each batch's DCT runs as one matrix product on *device*.

        Returns:
            Dict path-str -> {"phash", "dhash"}; unreadable images are skipped
            with a warning. ``progress_cb(done)`` fires per flushed batch.
        """
        paths = list(paths)
        total = len(paths)
        out = {}
        if not total:
            if progress_cb:
                progress_cb(0)
            return out

        img_size = hash_size * 4
        # Decode pool sized from the live machine state (same pattern as
        # FeatureExtractor.extract_batch): the planner charges the
        # in-flight decoded crops against the RAM budget, and the pool decodes
        # one image per task so every worker stays busy (PIL releases the GIL
        # in its C decode/resize kernels). The window is `workers` batches'
        # worth of images to keep the device flushes fed.
        plan = self.machine.plan_images(
            "phash-decode", paths,
            parent_fixed_bytes=float(img_size * img_size * 8
                                     + (hash_size + 1) * hash_size),
            in_flight_per_worker=max(1, prefetch_batches), cap_workers=16,
            env_prefix="PHASH_DECODE")

        workers = max(1, int(decode_workers or plan.jobs))
        window = max(1, workers * max(1, prefetch_batches))

        def decode_one(p):
            """Decode + grayscale-resize one image (thread-safe; PIL only)."""
            try:
                img = Image.open(p).convert("L")
                small = img.resize((img_size, img_size), Image.LANCZOS)
                dh_small = img.resize((hash_size + 1, hash_size), Image.LANCZOS)
                dh_px = np.asarray(dh_small)
                return (str(p), np.asarray(small, dtype=np.float64),
                        dh_px[:, 1:] > dh_px[:, :-1])
            except Exception as e:
                self._log(f"[Hash] Skipping {p}: {e}")
                return None

        m = self._dct_matrix(device, img_size)

        def flush(batch):
            """DCT one prepared batch on the device and emit its payloads."""
            if not batch:
                return
            arr = np.stack([item[1] for item in batch])       # [N, s, s]
            t = torch.from_numpy(arr).to(device)
            dct = m @ t @ m.T                                 # [N, s, s]
            low = dct[:, :hash_size, :hash_size]              # [N, 8, 8]
            med = torch.quantile(low.flatten(1), 0.5, dim=1)  # [N]
            bits = (low > med[:, None, None]).cpu().numpy()
            for i, (p, _, dh_bits) in enumerate(batch):
                out[p] = {"phash": self._pack_bits(bits[i].flatten()),
                          "dhash": self._pack_bits(dh_bits.flatten())}
            if progress_cb:
                progress_cb(len(out))

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            queue = iter(paths)
            pending = deque()
            for _ in range(min(window, total)):           # prime the window
                p = next(queue, None)
                if p is None:
                    break
                pending.append(pool.submit(decode_one, p))
            batch = []
            for _ in range(total):
                future = pending.popleft()
                nxt = next(queue, None)
                if nxt is not None:
                    pending.append(pool.submit(decode_one, nxt))
                item = future.result()
                if item is not None:
                    batch.append(item)
                if len(batch) >= batch_size:
                    flush(batch)
                    batch = []
            flush(batch)
        finally:
            pool.shutdown(wait=True)
        if progress_cb:
            progress_cb(len(out))
        return out

    @staticmethod
    def pick_device(min_free_mb: float = HASH_MIN_FREE_MB) -> torch.device:
        """Device for one hash batch (CUDA preferred, tiny VRAM gate)."""
        from .device import pick_device
        return pick_device(min_free_mb=min_free_mb)
