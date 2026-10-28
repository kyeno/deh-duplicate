# `deh-duplicate`

Image duplicate detection: find duplicate and near-duplicate images in a directory tree.
By default, byte-identical extras are quarantined automatically. Similar image groups are reviewed and approved in a dialog before their losers are moved; use `--no-gui` for automatic approval.
Quarantined files go under one `_quarantine` directory at the scan root, with their relative paths preserved.

This is part of the "dehordeur" project, which I'm mostly vibe-coding after hours.
It came to life because I couldn't find any *reasonable* image deduplication tool.
`findimagedupes` is inconvenient, `jdupes` mostly works only on binary-identical files, and `geeqie`'s deduplication function is painfully slow and cumbersome to review.

`deh-duplicate` is fast to process (CUDA, multi-threading) and quick to review (Qt UI with keyboard-first navigation).

## Install

Requires Python 3.10+ on Linux.
Clone this repository, then run:

```bash
cd deh-duplicate
./setup.sh
```

`setup.sh` is idempotent and does four things:

1. creates `.venv/` — the interpreter `bin/deh-duplicate.sh` runs
2. `pip install -e .` — torch from PyPI is GPU-ready and large; `--cpu` takes the
   small CPU-only wheels, `--dev` adds pytest + ruff
3. copies the annotated `.env.example` to the local `.env`
4. downloads the MobileNetV2 weights into `var/models/` — required by
   `--mode deep` / `--mode hybrid`, and cached offline from then on

On a headless box, pass `--no-gui` to skip the approval dialog. The package
still installs PyQt6 as a dependency.

## deh-duplicate

```bash
./bin/deh-duplicate.sh ./photos --mode hybrid
```

Options:

```
directory                    Root directory to scan recursively for images
-t, --threshold FLOAT        Hash threshold in hash/hybrid (default: 75), deep in deep (default: 82)
--mode {hash,deep,hybrid}    Comparison method (default: hybrid)
--deep-threshold FLOAT       Deep threshold used by deep/hybrid modes (default: 82)
--delete                     Delete losers instead of moving them to _quarantine
--dry-run                    Preview only; change nothing on disk
--no-gui                     Skip the per-group approval dialog (fully automatic)
--no-identical-dedupe        Skip the pre-pass that collapses byte-for-byte identical files
-v, --verbose                Detailed output, including per-image scores
--machine                    Dump the machine state (cores/RAM/GPU/tool versions) as JSON
```

`hybrid` compares all image pairs with deep features at `--deep-threshold`
(82% by default), then requires a perceptual hash match at `--threshold`
(75% by default) within each deep group. This filters out many images with
similar subjects but different visual composition.
`-t` also sets the deep threshold in hybrid mode unless `--deep-threshold` is
given. `deep` compares all image pairs with an 82% default deep threshold;
`hash` uses a 75% default perceptual hash threshold. The deep comparison costs roughly
quadratic time as the image count grows. Very large connected groups stop
with a clear memory budget error before dense clustering.

If the review dialog cannot start, the run stops with an error. Pass `--no-gui`
to explicitly allow automatic approval. File operation failures also produce a
nonzero exit status; any moves completed before the failure remain in quarantine.
Interactive terminal runs show progress bars for feature extraction, hashing,
and pairwise comparisons. `--no-gui` and redirected output retain plain progress
lines.

Other options also have environment variables (all namespaced with `DEH_`, read through
`deh.config` and documented in [`.env.example`](.env.example)):
`DEH_DUPLICATE_THRESHOLD`, `DEH_DUPLICATE_MODE`, `DEH_DUPLICATE_DEEP_THRESHOLD`,
`DEH_DUPLICATE_VERBOSE`, `DEH_DUPLICATE_IDENTICAL_DEDUPE`,
`DEH_GUI_SCALE`, `DEH_QUIET`.

## Using it as a library

The duplicate feature is a plain library; the CLI is a thin adapter over it:

```python
from deh.core.scanner import ImageScanner
from deh.duplicate.pipeline import DuplicateConfig, DuplicatePipeline
from deh.duplicate.quality import QualityAssessor
```

## Tests

```bash
./setup.sh --dev                        # adds pytest + ruff to .venv
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

The CUDA parity tests (`tests/core/test_hamming.py`, `tests/duplicate/test_hashing.py`,
`tests/duplicate/test_comparator.py`) need free VRAM: on a busy GPU they fail with
`CUDA error: out of memory`, which is the host, not the code.

## License

MIT
