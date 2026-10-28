# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
#!/usr/bin/env python3
"""Image Similarity Comparison & Duplicate Management Tool (thin CLI adapter).

Finds visually similar images using perceptual hashing, deep learning feature
extraction, or both. Within each group of similar images, ranks them by quality
metrics (resolution, sharpness, file size). Losers are moved to a _quarantine
subdirectory by default; use --delete to remove them permanently, or --dry-run
to preview without changing anything on disk.

The implementation lives in :mod:`deh.duplicate.pipeline` (``DuplicateConfig``
+ ``DuplicatePipeline``); this module is only the argument-parsing adapter:
``parse_args()`` → ``resolve_config()`` → ``DuplicatePipeline(config).run()``.

Usage:
    deh-duplicate <directory> [options]

Other options can also be set via environment variables or a .env file (every
variable is namespaced with ``DEH_``; see :mod:`deh.config` and
``.env.example``):
    DEH_DUPLICATE_THRESHOLD=75          # Hash threshold in hash/hybrid (default 75%)
    DEH_DUPLICATE_MODE=hybrid           # hash | deep | hybrid (recommended default)
    DEH_DUPLICATE_DEEP_THRESHOLD=82     # Deep threshold in deep/hybrid (default 82%)
    DEH_DUPLICATE_VERBOSE=true          # Enable detailed output
    DEH_GUI_SCALE=1.4                   # Enlarge dialog/cards by this factor (>= 1.0; try 1.2-1.6)
                                        # (dialogs always open centered on the primary monitor;
                                        #  placement conventions live in CONTRIBUTING.md)
    DEH_DUPLICATE_IDENTICAL_DEDUPE=true # Collapse byte-for-byte identical files first (default true)
    DEH_QUIET=true                      # Silence the library's progress lines (banner + alarms stay)
"""

import argparse
import sys

from .._meta import APP_NAME, __version__
from ..config import DEFAULT as DEFAULT_CONFIG
from ..core.banner import startup_banner
from ..core.terminal import notice
from ..duplicate.pipeline import DuplicateConfig, DuplicatePipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and quarantine visually similar/duplicate images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s ./photos                    # Move losers to _quarantine (default)
    %(prog)s ./photos --dry-run          # Preview only; change nothing on disk
    %(prog)s ./photos --delete           # Delete losers instead of moving them
    %(prog)s ./photos -t 80              # Use an 80%% comparison threshold

Environment variables (or .env file, all namespaced): DEH_DUPLICATE_THRESHOLD,
DEH_DUPLICATE_MODE, DEH_DUPLICATE_DEEP_THRESHOLD,
DEH_DUPLICATE_VERBOSE, DEH_GUI_SCALE, DEH_DUPLICATE_IDENTICAL_DEDUPE
""")

    parser.add_argument("directory",
                       help="Root directory to scan recursively for images")

    parser.add_argument("-t", "--threshold", type=float, default=None,
                       help="Similarity threshold percentage (0-100): hash in "
                       "hash/hybrid modes, deep in deep mode. In deep/hybrid "
                       "modes, --deep-threshold takes precedence. "
                       "Defaults: 75%% hash, 82%% deep.")

    parser.add_argument("--delete", action="store_true", default=False,
                       help="Delete losing images instead of moving them to _quarantine.")

    parser.add_argument("--dry-run", action="store_true", default=False,
                       help="Preview only: show what would happen without changing any files.")

    parser.add_argument("-v", "--verbose", action="store_true", default=None,
                       help="Enable detailed output including per-image scores.")

    parser.add_argument("--mode", choices=["hash", "deep", "hybrid"], default=None,
                       help="Comparison method: hash (fast), deep (full comparison), "
                       "hybrid (deep matches verified by perceptual hashes). "
                       "Default: hybrid.")

    parser.add_argument("--deep-threshold", type=float, default=None,
                       help="Deep similarity threshold percentage (0-100) for deep/hybrid modes. "
                       "Default: 82%% or $DEH_DUPLICATE_DEEP_THRESHOLD.")

    parser.add_argument("--no-gui", action="store_true", default=False,
                       help="Disable interactive GUI approval dialogs; run fully automatic.")

    parser.add_argument("--no-identical-dedupe", action="store_true", default=False,
                        help="Skip the pre-pass that collapses byte-for-byte identical files "
                             "(enabled by default). See DEH_DUPLICATE_IDENTICAL_DEDUPE.")

    parser.add_argument("--machine", action="store_true", default=False,
                        help="Print the machine state (cores/RAM/GPU/tool versions) "
                             "as JSON, then exit.")

    parser.add_argument("--version", action="version",
                        version=f"{APP_NAME} {__version__}",
                        help="Show the tool name and version, then exit.")

    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> DuplicateConfig:
    """Resolve final configuration from CLI args, environment, and defaults.

    Precedence is CLI -> environment -> built-in default throughout.  The
    environment is read through :data:`DEFAULT_CONFIG` so the variable names
    live in one place (:mod:`deh.config`, prefix ``DEH_``) and this adapter
    never touches ``os.environ`` itself.
    """
    env = DEFAULT_CONFIG

    # Mode: CLI > ENV > default (hybrid)
    mode = args.mode or env.env("DUPLICATE_MODE") or "hybrid"

    # Hash threshold: CLI > ENV > default. The template .env defines 75%,
    # so deep mode must use its separate deep threshold even when that file
    # has not been customized.
    if args.threshold is not None:
        hash_threshold = args.threshold / 100.0
    else:
        env_threshold = env.env_float("DUPLICATE_THRESHOLD")
        hash_threshold = (env_threshold / 100.0) if env_threshold is not None else 0.75

    # Deep threshold for deep/hybrid: explicit CLI > -t CLI > ENV > default.
    if args.deep_threshold is not None:
        deep_threshold = args.deep_threshold / 100.0
    elif mode in ("deep", "hybrid") and args.threshold is not None:
        deep_threshold = args.threshold / 100.0
    else:
        env_deep = env.env_float("DUPLICATE_DEEP_THRESHOLD")
        deep_threshold = (env_deep / 100.0) if env_deep is not None else 0.82

    threshold = deep_threshold if mode == "deep" else hash_threshold

    # Only the explicit action flags change the default filesystem behavior.
    action = "delete" if args.delete else "move"
    dry_run = args.dry_run

    verbose = args.verbose or env.env_bool("DUPLICATE_VERBOSE")

    use_gui = not args.no_gui

    # Identical-file pre-pass: ON by default; disable via --no-identical-dedupe
    # or DEH_DUPLICATE_IDENTICAL_DEDUPE=false.
    identical_dedupe = (not args.no_identical_dedupe
                        and env.env_bool("DUPLICATE_IDENTICAL_DEDUPE", True))

    return DuplicateConfig(
        directory=args.directory,
        threshold=threshold,
        mode=mode,
        action=action,
        dry_run=dry_run,
        verbose=verbose,
        use_gui=use_gui,
        identical_dedupe=identical_dedupe,
        deep_threshold=deep_threshold,
    )


def _print_machine():
    """--machine: dump the machine-state snapshot (cores/RAM/GPU/tools) as JSON."""
    import json

    from ..core.machine import machine

    print(json.dumps(machine().dump(), indent=2, sort_keys=True))


def main() -> None:
    """Thin adapter: parse args → build DuplicateConfig → run the pipeline."""
    try:
        args = parse_args()
    except SystemExit as e:
        sys.exit(e.code)

    if args.machine:
        _print_machine()
        return

    config = resolve_config(args)

    # Show startup banner with configuration
    if config.dry_run:
        output_desc = f"DRY-RUN (would {config.action} losers)"
    elif config.action == "delete":
        output_desc = "DELETE losers permanently"
    else:
        output_desc = "MOVE losers to _quarantine"

    threshold_lines = (
        [f"Deep threshold: {config.deep_threshold:.2f} ({config.deep_threshold * 100:.0f}%)",
         f"Hash threshold: {config.threshold:.2f} ({config.threshold * 100:.0f}%)"]
        if config.mode == "hybrid"
        else [f"Threshold:   {config.threshold:.2f} ({config.threshold * 100:.0f}%)"]
    )
    banner_lines = [
        f"Directory:   {config.directory}",
        *threshold_lines,
        f"Mode:        {config.mode.upper()}",
        f"Output:      {output_desc}",
        f"Verbose:     {config.verbose}",
    ]
    startup_banner(
        "deh-duplicate",
        __version__,
        banner_lines,
    )

    # The pipeline and its library collaborators write through this sink, so
    # the run's progress goes through the project's output channel: DEH_QUIET
    # silences it, and the CLI's own banner above stays visible (it is the
    # CLI's output, not the library's).
    progress = None
    if config.use_gui and sys.stdout.isatty() and not DEFAULT_CONFIG.env_bool("QUIET"):
        from .progress import ProgressBars

        progress = ProgressBars()

    try:
        pipeline_kwargs = (dict(out=progress.log, progress_cb=progress.update)
                           if progress is not None else dict(out=notice))
        DuplicatePipeline(config, **pipeline_kwargs).run()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if progress is not None:
            progress.close()


if __name__ == "__main__":
    main()
