# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Feature Extraction Module

Uses a pre-trained MobileNetV2 CNN (via torchvision) to extract high-dimensional
feature vectors from images. These embeddings capture semantic visual content
and enable accurate similarity comparisons via cosine distance.

MobileNetV2 was chosen over ResNet for faster inference speed while maintaining
good accuracy for image similarity detection tasks.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from concurrent.futures import ThreadPoolExecutor

# Optional: torchvision.io for nvJPEG decode (CUDA only)
try:
    from torchvision.io import decode_jpeg, ImageReadMode
    _HAS_TORCHVISION_IO = True
except Exception:
    _HAS_TORCHVISION_IO = False

from ..core.device import pick_device
from ..core.machine import machine as default_machine
from ..core.terminal import notice


class FeatureExtractor:
    """Extracts deep learning feature vectors from images using MobileNetV2."""

    def __init__(self, machine=None, log_cb: Optional[Callable[[str], None]] = None):
        """Initialize the feature extractor.

        The device is chosen automatically via ``pick_device()``: CUDA if
        available and it has enough free VRAM, else CPU (with a big red
        warning). The model load is the big allocation, so the decision is
        made here, at load time.

        Args:
            machine: Machine-state service used to size the decode pool.
            log_cb: Receives each human-readable output line (device selection,
                decode-path notices, per-image warnings). Default: the
                ``core.terminal`` notice channel; an embedder passes its own
                writer, or a no-op for a silent run (review 5.3).
        """
        self.device = pick_device()
        # Machine service: sizes the decode pool from the live RAM/CPU state.
        self.machine = machine or default_machine()
        self._log_cb = log_cb

        self._log(f"[Features] Using device: {self.device}")
        if str(self.device) == "cuda" or str(self.device).startswith("cuda"):
            try:
                self._log(f"[Features]   torch={torch.__version__} "
                          f"cuda_available={torch.cuda.is_available()} "
                          f"gpu='{torch.cuda.get_device_name(0)}'")
            except Exception as e:
                self._log(f"[Features]   CUDA query failed: {e}")

        # Load pretrained MobileNetV2 and strip the classification head to get
        # pooled 1280-dim features.
        #
        # NOTE: do NOT build this as ``list(model.children())[:-1]``. torchvision
        # folded MobileNetV2's avgpool/dropout/fc into a single ``classifier``
        # Sequential, so ``children()`` is now just ['features', 'classifier']
        # and the old slicing kept ONLY 'features' -- emitting an unpooled
        # (B, 1280, 7, 7) map. That broke the comparator's row-block matmul
        # (``F[start:end] @ F.mT`` no longer yields [chunk, n]) and made the
        # GPU path silently return zero groups. Compose the trunk explicitly so
        # the 1280-dim contract in ``feature_dim`` actually holds.
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        self.model = torch.nn.Sequential(
            model.features,          # -> (B, 1280, 7, 7)
            nn.AdaptiveAvgPool2d(1),  # -> (B, 1280, 1, 1)
            nn.Flatten(1),           # -> (B, 1280)
        ).to(self.device)
        self.model.eval()

        # Verify parameters actually landed on the intended device
        param_dev = next(self.model.parameters()).device
        self._log(f"[Features] Model parameters resident on: {param_dev}")

        # Standard ImageNet normalization transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def _log(self, text: str) -> None:
        """Route one output line to the caller's ``log_cb``, or to the terminal."""
        if self._log_cb is not None:
            self._log_cb(text)
        else:
            notice(text)

    def extract(self, image_path: Path) -> np.ndarray:
        """Extract feature vector from a single image.

        Args:
            image_path: Path to the input image file.

        Returns:
            Normalized 1280-dimensional float32 numpy array.
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image {image_path}: {e}") from e

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        # Flatten and L2-normalize for cosine similarity computation
        features = features.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features

    def extract_batch(self, image_paths: List[Path], batch_size: int = 32,
                      prefetch_batches: int = 3, decode_workers: Optional[int] = None,
                      progress_cb: Optional[Callable[[int, int], None]] = None) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Extract features for multiple images in batches.

        Decoding/resizing is CPU-heavy and used to run serially right ahead of
        GPU inference, so the two phases stalled each other (GPU idle while one
        core decoded). This version keeps a small pool of decoder threads
        preparing upcoming batches while the current batch runs on the device,
        overlapping decode with inference.

        On CUDA with JPEG images, uses nvJPEG (torchvision.io.decode_jpeg) for
        GPU-accelerated decode + resize, avoiding CPU->GPU transfer of pixels.

        Args:
            image_paths: List of paths to process.
            batch_size: Number of images per GPU/CPU batch.
            prefetch_batches: How many prepared batches may wait in flight.
            decode_workers: Decoder thread count (default: min(cpu_count, 16)).
            progress_cb: Receives ``(done, total)`` after each reported batch
                (~every 5%), for callers that want a progress bar instead of a
                printed line. The printed line still goes to ``log_cb``.

        Returns:
            Dictionary mapping path strings to normalized feature vectors.
            On CUDA: returns torch.Tensor on device (zero-copy for comparator).
            On CPU: returns numpy arrays (backward compatible).
        """
        results = {}
        total = len(image_paths)
        if total == 0:
            return results

        # Decode pool sized from the live machine state (was a fixed
        # min(cpu_count, 16)): one in-flight item here is a whole *batch*, so
        # the plan charges the batch's bytes against the RAM budget. The
        # prefetch window below already bounds how many batches may exist.
        plan = self.machine.plan_images(
            "mobilenet-decode", image_paths,
            parent_bytes_per_px=3.0 * batch_size,
            parent_fixed_bytes=float(batch_size * 3 * 224 * 224 * 4),
            in_flight_per_worker=max(1, prefetch_batches), cap_workers=8,
            env_prefix="FEATURES_DECODE")

        decode_workers = max(1, int(decode_workers or plan.jobs))

        use_gpu_decode = (self.device.type == "cuda" and _HAS_TORCHVISION_IO)
        if use_gpu_decode:
            self._log("[Features] Using nvJPEG GPU decode for JPEG images")

        skipped: List[str] = []

        def prepare(start):
            tensors, valid_indices = [], []
            for idx, p in enumerate(image_paths[start:start + batch_size]):
                # Try GPU decode first for JPEG on CUDA
                # p might be str or Path
                p_path = Path(p) if isinstance(p, str) else p
                tensor = None
                if use_gpu_decode and p_path.suffix.lower() in (".jpg", ".jpeg"):
                    tensor = self._decode_and_transform_gpu(p_path)
                # Fallback to CPU PIL path
                if tensor is None:
                    tensor = self._transform_path(p_path)
                if tensor is not None:
                    tensors.append(tensor)
                    valid_indices.append(idx)
                else:
                    # Undecodable on BOTH paths (a truly broken file). Record it:
                    # it silently drops out of the comparison pool below, and a
                    # dropped file can never be reported as a duplicate.
                    skipped.append(str(p))
            return tensors, valid_indices

        starts = list(range(0, total, batch_size))
        last_reported = 0
        report_step = max(batch_size, total // 20)   # ~5% progress milestones

        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            pending = {s: pool.submit(prepare, s) for s in starts[:max(1, prefetch_batches)]}

            for si, start in enumerate(starts):
                fut = pending.pop(start)
                nxt = si + max(1, prefetch_batches)
                if nxt < len(starts):
                    pending[starts[nxt]] = pool.submit(prepare, starts[nxt])

                tensors, valid_indices = fut.result()

                if not tensors:
                    continue

                # Stack into batch and run inference (device phase).
                # A batch can hold tensors from BOTH devices: nvJPEG yields
                # GPU tensors, the PIL fallback yields CPU ones, and one
                # corrupt JPEG is enough to make the batch mixed. torch.stack
                # requires a single device, so normalise BEFORE stacking --
                # a trailing .to() on the stacked result is too late and
                # raises "Expected all tensors to be on the same device".
                batch_tensor = self._stack_on_device(tensors)
                with torch.no_grad():
                    batch_features = self.model(batch_tensor)

                # Process each result in the batch
                for j, idx in enumerate(valid_indices):
                    feat = batch_features[j].squeeze()
                    # Return on device for zero-copy (N2), or CPU for backward compat
                    if self.device.type == "cuda":
                        # Keep on GPU for N1/N2 pipeline
                        norm = torch.linalg.norm(feat)
                        if norm > 0:
                            feat = feat / norm
                        results[str(image_paths[start + idx])] = feat
                    else:
                        feat = feat.cpu().numpy().astype(np.float32)
                        norm = np.linalg.norm(feat)
                        if norm > 0:
                            feat /= norm
                        results[str(image_paths[start + idx])] = feat

                # Progress indicator (~every 5%, always at completion)
                processed = min(start + batch_size, total)
                if processed >= last_reported + report_step or processed == total:
                    self._log(f"[Features] Extracted features for {processed}/{total} images")
                    if progress_cb is not None:
                        progress_cb(processed, total)
                    last_reported = processed

        if skipped:
            self._log(f"[Features] Skipped {len(skipped)} image(s) undecodable on every "
                      f"path; they are excluded from comparison and can never be "
                      f"reported as duplicates:")
            for p in skipped[:10]:
                self._log(f"[Features]   skipped: {p}")
            if len(skipped) > 10:
                self._log(f"[Features]   ... and {len(skipped) - 10} more")

        return results

    def _stack_on_device(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Stack a possibly mixed-device batch onto ``self.device``.

        ``torch.stack`` requires every input to already live on one device, so
        each tensor is moved first. Mirrors the ordering the comparator already
        uses in ``comparator.py`` (``.to(device)`` then ``torch.stack``).
        """
        return torch.stack([t.to(self.device) for t in tensors])

    def _transform_path(self, image_path):
        """Decode + transform one image (thread-safe; runs on worker threads).

        Returns the transformed tensor, or None after logging a warning when
        the file cannot be read/converted.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            self._log(f"[Features] Warning: Could not process {image_path}: {e}")
            return None

    def _decode_and_transform_gpu(self, image_path: Path) -> Optional[torch.Tensor]:
        """Decode JPEG via nvJPEG on GPU, resize with F.interpolate, normalize.

        Only works for JPEG files on CUDA with torchvision.io available.
        Returns None if not applicable (non-JPEG, no CUDA, no torchvision.io).
        """
        if not _HAS_TORCHVISION_IO:
            return None
        if self.device.type != "cuda":
            return None
        if image_path.suffix.lower() not in (".jpg", ".jpeg"):
            return None

        try:
            # Read file bytes
            with open(image_path, "rb") as f:
                data = f.read()
            # nvJPEG decode directly to GPU tensor (uint8, CHW)
            # Use copy() to avoid non-writable tensor warning
            img_tensor = decode_jpeg(torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy()),
                                     device=str(self.device), mode=ImageReadMode.RGB)
            # img_tensor is [C, H, W] uint8 on GPU
            # Resize to 256 on shorter side, then center crop 224
            # Use bilinear interpolation (matches PIL LANCZOS reasonably for this use case).
            # antialias=True is REQUIRED: PIL's resize is antialiased, and without
            # it the two decode paths diverge ~20x in mean abs pixel error (measured
            # 0.0197 vs 0.0010 on a sample JPEG), so GPU-decoded JPEGs and
            # PIL-decoded non-JPEGs in the same corpus would get systematically
            # different embeddings.
            c, h, w = img_tensor.shape
            if h < w:
                new_h, new_w = 256, int(256 * w / h)
            else:
                new_h, new_w = int(256 * h / w), 256
            img_tensor = F.interpolate(img_tensor.unsqueeze(0).float() / 255.0,
                                       size=(new_h, new_w), mode="bilinear",
                                       align_corners=False, antialias=True).squeeze(0)
            # Center crop 224x224
            start_h = (new_h - 224) // 2
            start_w = (new_w - 224) // 2
            img_tensor = img_tensor[:, start_h:start_h+224, start_w:start_w+224]
            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            return img_tensor
        except Exception as e:
            # Only the first line: the raw nvJPEG error spans several lines of
            # C++ stack trace, which buried the actual message on corpus scans.
            reason = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            self._log(f"[Features] GPU decode failed for {image_path}, "
                      f"falling back to CPU: {reason}")
            return None

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        return 1280
