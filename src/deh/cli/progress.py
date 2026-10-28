"""Terminal progress bars for interactive duplicate scans."""

from __future__ import annotations

import sys

from tqdm import tqdm

from ..core.terminal import notice


_STAGES = {
    "features": ("Extracting features", "image", "cyan"),
    "deep_pairs": ("Comparing features", "pair", "blue"),
    "hash_images": ("Hashing images", "image", "green"),
    "hash_pairs": ("Comparing hashes", "pair", "magenta"),
}


class ProgressBars:
    """Render counted pipeline stages while preserving ordinary notices."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self._bars = {}
        self._completed = set()

    def update(self, stage: str, done: int, total: int) -> None:
        if total < 1 or stage in self._completed:
            return
        bar = self._bars.get(stage)
        if bar is None:
            description, unit, colour = _STAGES[stage]
            bar = tqdm(total=total, desc=description, unit=unit,
                       dynamic_ncols=True, colour=colour, file=self.stream)
            self._bars[stage] = bar
        target = min(total, max(0, done))
        if target > bar.n:
            bar.update(target - bar.n)
        if target >= total:
            bar.close()
            del self._bars[stage]
            self._completed.add(stage)

    def log(self, message: str) -> None:
        # Counted lines and per-group hash verification details are represented
        # by the bars and the later review. Keep warnings and phase headings.
        if message.startswith((
            "[Features] Extracted features for ",
            "[Hash] Computing perceptual hashes for ",
            "[Hash] Hashed ",
            "[Hash] Comparing ",
            "[Hash]   ...",
            "[Hash] Completed ",
            "[Hash] Using max Hamming distance:",
            "[Hash] Found ",
            "[Comparator] Comparing ",
            "[Comparator] Completed ",
        )) or message.strip().startswith("[timing] Perceptual hash verification:"):
            return
        with tqdm.external_write_mode(file=self.stream):
            notice(message, stream=self.stream)

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()
