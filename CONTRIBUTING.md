# Contributing to deh-duplicate

Follow these conventions for every change; when in doubt, copy the shape of the
module next door.

## Setup

```bash
./setup.sh --dev          # .venv + deps + pytest/ruff + the MobileNetV2 weights
```

`setup.sh` is the supported way to build a working tree, and it is idempotent —
re-running repairs what is missing and touches nothing else. It creates `.venv`
(the interpreter `bin/*.sh` runs: that path is hard-coded there, so the venv is
named `.venv` by contract), installs the project editable, copies `.env.example`
→ `.env` when no `.env` exists, and downloads the MobileNetV2 checkpoint into
`var/models/` — the `TORCH_HOME` that `deh/__init__.py` pins, so deep/hybrid mode
needs no network access afterwards.

- Requires **Python 3.10+** on Linux (`requires-python` in `pyproject.toml`);
  `PYTHON=/path/to/python ./setup.sh` picks a specific interpreter.
- `--cpu` takes the CPU-only torch wheels from `download.pytorch.org`; the default
  PyPI wheel bundles the CUDA runtime.
- `--dev` installs the `dev` extra (`pytest`, `ruff`) — the gates in §Tests.
- Nothing under `var/` is committed: a fresh clone has no `var/`, and the weights
  are fetched by torchvision (sha256-verified), never vendored.

## Layout

- **src-layout:** code lives in `src/deh/`; the import name is `deh`, the
  distribution is `deh-duplicate`. Never add packages outside `src/`.
- **Feature-first packaging:** one subpackage per feature (`deh/duplicate/`).
  Anything shared and feature-agnostic goes to `deh/core/` (scanner, clustering,
  device, image I/O).
- **No flat modules at package root:** only `deh/config.py`, `deh/_meta.py`,
  `deh/__init__.py` and the subpackages.
- **Path bootstrap (`deh/__init__.py`)** stays first: it fixes `TORCH_HOME`
  from `ProjectConfig` *before* anything can import torch.

```
src/deh/
├── config.py     ProjectConfig — single source of truth for every path
├── _meta.py      APP_NAME / __version__
├── core/         feature-agnostic primitives (no GUI/CLI imports)
│   ├── banner.py     shared startup banner
│   ├── cluster.py    Union-Find single-linkage clustering
│   ├── device.py     GPU/CPU device resolution + VRAM probing (GpuProbe)
│   ├── hamming.py    Hamming-distance math for packed uint64 hashes
│   ├── hashes.py     batched pHash/dHash engine (parallel decode + GPU DCT)
│   ├── imageio.py    image loading + human-readable byte formatting
│   ├── machine.py    machine-state service + per-stage resource budgets
│   ├── scanner.py    ImageScanner (recursive image discovery)
│   └── terminal.py   shared terminal-output helpers (banners, TTY detection)
├── cli/          thin drivers (parse → config → run)
├── duplicate/    duplicate/near-duplicate feature
└── gui/          shared widgets + the duplicate approval dialog
```

## One class per module

- A module exposes **one primary class** (`QualityAssessor`, `QuarantineManager`,
  …). The pipeline that owns a feature is the composition root
  (`DuplicatePipeline`).
- Free functions are allowed only for trivial pure helpers (e.g. `human_bytes`)
  or process-wide output helpers (the terminal banners in `core/terminal.py`).
  If it holds state or needs collaborators, make it a class.
- **No module-level mutable state.** Caches (`_model_cache`, DCT matrices) are
  instance attributes created in `__init__`. Module-level constants are fine —
  but only *immutable data*.
- **Sanctioned process-wide services (the only exceptions):**
  `deh.config.DEFAULT` (immutable path config) and
  `deh.core.machine.machine()` — the machine-state service (cores/RAM/GPU/tool
  versions captured eagerly at startup, live readings TTL-cached behind a
  lock, resettable via `set_machine`/`reset_machine`). Everything else stays
  injected by the composition root.

## Dependency injection

- Classes take collaborators as **constructor arguments with sensible
  defaults**:

  ```python
  class HashComparator:
      def __init__(self, threshold=0.75, engine=None, machine=None):
          self.engine = engine or HashEngine()
          self.machine = machine or machine()
  ```

- The pipeline wires the graph; callers (and tests) may override any node.
  The CLI only parses arguments, builds the config, and calls `run()`.
- **Output is injected the same way.** A module that reports progress takes a
  `log_cb` (one line of human-readable prose) and, for its loops, a
  `progress_cb(done, total, ...)` milestone callback; the pipeline hands its own
  output sink down to every collaborator, so one callback captures a whole run.
  The fallback when a caller passes none is `core.terminal.notice()` — the
  single flush point — which `DEH_QUIET` silences. **No `print()` in library
  modules**: `cli/` and `core/terminal.py` own the terminal.
- No singletons, no globals beyond the **two sanctioned process-wide
  services** (above), no reaching into another module for its state.
  Load `deh.config` values once and pass them down.

## Paths

- **Project runtime paths come from `deh.config.ProjectConfig`** (`root`,
  `var_dir`, `models_dir`, `cache_dir`, `env_file`, …). The image scan root comes
  from the CLI directory argument. Never write CWD-relative `"var/..."`
  literals — the packaged CLI is invoked from anywhere.
- **`var/` is runtime data, not source.** `var/models` (the `TORCH_HOME` torch
  writes to) and `var/cache` are created at setup/run time and are git-ignored;
  the MobileNetV2 checkpoint lives in `var/models/hub/checkpoints/` and is
  downloaded by `./setup.sh`. Never commit weights, caches, or anything under `var/`.

## Configuration

- Every environment variable the project reads is namespaced with `DEH_`
  (`DEH_DUPLICATE_THRESHOLD`, `DEH_DUPLICATE_MODE`, `DEH_GUI_SCALE`, ...).
  Bare names (`MODE`, `VERBOSE`, `THRESHOLD`, `DRY_RUN`, `IDENTICAL_DEDUPE`...)
  collide with shells, CI runners and neighbouring projects, so they are not
  read at all.
- Read variables through `ProjectConfig` — `env()`, `env_float()`,
  `env_int()`, `env_bool()` — never `os.environ` directly. The name list then
  lives in one file, `.env` and the CLI resolve through the same code, and
  boolean settings accept `1/true/yes/on` (with anything unparseable falling
  back to the documented default). `--delete` and `--dry-run` are CLI-only.
- Third-party variables keep their established names (`TORCH_HOME`, `NO_COLOR`);
  `deh/__init__.py` seeds the first from `ProjectConfig` before any torch import.

## CLI

- One module per command under `deh/cli/` with
  `parse_args()` → `resolve_config()` (returns a dataclass, not a dict) →
  `main()`. `main()` does nothing but wire those three together and exit.
- Declare the entry point in `[project.scripts]`; keep `bin/*.sh` launchers in
  sync (they check that `.venv/bin/python` exists — pointing at `./setup.sh` when
  it does not — set `PYTHONPATH`, load `.env`, then call
  `python -m deh.cli.<command>`).

## GUI (`deh/gui/`)

- Reusable pieces live in `app.py` (QApplication bootstrap: `ensure_app()`),
  `theme.py` (CSS/palette), `imagecard.py` (image card widget) and `dialog.py`
  (`CardGridDialog` base: keyboard navigation, status lines).
- The feature dialog subclasses `CardGridDialog` and renders its items with
  `ImageCard`, so grid sizing, dark-theme colours and metadata lines stay
  identical across dialogs; only the interaction differs. Feature-specific
  controls are added by the subclass.
- Keep Qt imports out of non-GUI modules; `core/` must stay importable without
  a display server.
- Feature dialogs are placed with the shared `app.py` helpers —
  `enlarge_to_screen()` then `center_on_screen()` before `exec()` — so every
  window opens large and centered on the primary monitor. Explicit placement is
  honoured on X11; Wayland compositors position toplevels themselves.
- Explicit placement is an X11-only request: Wayland compositors ignore it and
  position toplevels themselves (KDE's default policy looks centred anyway),
  and tiling WMs tile the dialogs unless you add a float rule. Force X11 with
  `QT_QPA_PLATFORM=xcb` if needed.

### Keyboard bindings

In the duplicate approval the box the arrows move **is** the winner design — green
border and green tint — so the dialog opens with the keeper already in place on the
scorer's winner and the arrows move that look from card to card; the confirmed keeper
keeps only its star and its note line. Landing on an excluded card burns the box's
border to a brighter red (`KEEPER_HOT_BORDER` in `theme.py`) while the card's own red
tint stays put. There is no third cursor colour.

| Key | Duplicate approval |
| --- | --- |
| `← → ↑ ↓` | Move the keeper box (wraps; never skips excluded) |
| `Enter` | Confirm the image in the box as keeper and approve the group (quarantines the rest); **noop on an excluded image** |
| `Space` | Exclude / restore the image in the box; **dropping the keeper hands the role to the runner-up** |
| `Esc` | Noop this group (keep all) |
| `Backspace` | Noop this group (keep all) |
| Left / right click | Keep / exclude (a click parks the box on that card) |

`Enter` never un-excludes: an excluded card must be restored with `Space` first,
so the keeper key cannot silently undo a deliberate exclusion. A left-click on an
excluded card *does* promote it — pointing at it with the mouse is explicit.

The keeper is excludable: `Space` on the image in the box takes it out of the
group and hands the keeper role to the best remaining candidate (highest score,
then scorer order), so the group is never keeper-less while a keeper-eligible
image remains. The box stays on the card that was used — it follows the user, not
the role — so `Space` never approves anything. Excluding the last keeper-eligible
image leaves the group with no keeper: `Enter` does nothing and `Esc` keeps all.

## Tests

- Mirror the package layout: `tests/core/`, `tests/duplicate/`, `tests/gui/`
  (+ root-level `tests/test_scanner.py`).
- `conftest.py` at the repo root puts `src/` on `sys.path` and forces
  `QT_QPA_PLATFORM=offscreen` — no editable install required to run the suite.
- **Keep the suite green at every commit.**

```bash
./setup.sh --dev                       # .venv + pytest + ruff (the `dev` extra)
.venv/bin/python -m pytest -q          # full suite
.venv/bin/python -m pytest -q tests/duplicate/test_pipeline.py
.venv/bin/ruff check src tests         # static gate (config in pyproject.toml)
```

## Style

- Don't add new dependencies to `pyproject.toml` for something already
  available (torch/numpy/PIL/PyQt6). Prefer the stdlib.
- Docstrings: the module docstring explains *why* the module exists and its
  payload/milestone status; public methods get one-liners. Keep comments
  focused on non-obvious invariants (schemas, thresholds, engines).
- Type hints on public signatures; `from __future__ import annotations` where
  already present. No formatter is enforced — match surrounding style.
- **`ruff check src tests` must stay clean.** The rule set lives in
  `pyproject.toml` and is chosen for what it *finds* (unused/undefined names,
  redefinitions, bugbear footguns), not for how much it reformats. Line length
  is not enforced. A new suppression needs a reason in the config, next to the
  rule it disables.
- Keep the CLI thin: if `main()` grows logic, it belongs in the feature class.

## Review checklist

The two things every review checks: **does the suite pass**, and **is any code
path still CWD-dependent?** Everything else follows from those.

The suite has a documented environment dependency: the CUDA parity tests
(`tests/core/test_hamming.py`, `tests/duplicate/test_hashing.py`,
`tests/duplicate/test_comparator.py`) need free VRAM. On a machine where the
GPU is busy they fail with `CUDA error: out of memory` — that is the host, not the
change. Check the same failure list on the base commit before blaming your diff.


## No Manual Preconfiguration

**Any by-hand preconfiguration (like hand-labelling ground-truth data, manual
seed files, or human-curated baselines) is strictly forbidden.** The software
must run as automated as possible. All thresholds, baselines, and quality
decisions must be derived from automated corpus replay and measurable metrics,
never from human-labelled data.

## Compute & Parallelism

- **Always prefer CUDA/GPU over CPU** for computation whenever possible. The
  machine service (`deh.core.machine.GpuProbe`) captures VRAM and device
  capabilities at startup — use it to size batches and worker pools.
- **Utilise CPU parallelism** for non-GPU stages (decoding, hashing, I/O).
  `machine().cpu_budget` gives the recommended worker count; never hard-code
  pool sizes.
- Device resolution is centralised in `deh.core.device.resolve_device()` —
  always go through it rather than importing `torch.cuda` directly.
- The *live* device for an allocation step comes from `deh.core.device.pick_device()`,
  which falls back to CPU when the shared GPU has too little free VRAM (another
  process holding it is the normal case). Its red alarm is announced **once per
  pipeline phase**: `pick_device` runs per hash batch / per distance matrix, so
  call `deh.core.device.begin_phase("<phase>")` at each phase boundary of a
  pipeline (see `duplicate/pipeline.py`) instead of printing your own warning per
  call. The device decision itself is never cached — only the alarm is.

## Singleton Pattern

The `machine()` service in `deh.core.machine` is a sanctioned process-wide
singleton (see §Dependency injection above). This pattern is
**not an anti-pattern here** — it captures expensive-to-probe hardware state
once at startup and serves read-only (TTL-cached) queries thereafter. Do not
refactor it to dependency injection without a concrete reason; the explicit
`set_machine`/`reset_machine` test hooks are the intended override mechanism.

## Adding Dependencies

New dependencies in `pyproject.toml` are acceptable when they provide
**measurably better or faster functionality**. Examples:
- `opencv-python` for the quality metrics' fast paths (median blur, NCC), with a
  SciPy fallback when it is missing

Prefer the stdlib for trivial helpers, but don't avoid a dependency that
eliminates a bottleneck or unlocks a capability the stdlib cannot provide.
