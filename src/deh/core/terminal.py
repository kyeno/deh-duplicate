# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Shared terminal-output helpers (alarms, notices, TTY detection).

Process-wide output primitives shared by the device policy (``core.device``),
the pipeline stages (``duplicate.pipeline``), and the quality/hash helpers.
Kept in ``core`` — the bottom of the dependency graph — so every layer can use
them without importing upward into ``cli``/``gui``. These are the

"process-wide output helpers" that CONTRIBUTING.md allows as free functions:
they hold no state and take no collaborators.

- ``red_banner``     — an alarm block (white on red on a TTY, ``!!`` otherwise).
- ``notice``         — one library output line: the fallback channel when a
                       caller passed no ``log_cb`` (silence with ``DEH_QUIET``).
- ``_color_enabled`` — whether ANSI colors should be emitted.
"""

import os
import sys
from typing import Optional, TextIO

from ..config import DEFAULT as _CONFIG

#: Environment switch that silences every library ``notice`` (review 5.3).
#: Declared here for documentation; it is read through ``ProjectConfig``, which
#: applies the ``DEH_`` prefix (CONTRIBUTING.md: no bare ``os.environ`` names).
QUIET_ENV = "DEH_QUIET"


def _quiet() -> bool:
    """True when ``DEH_QUIET`` asks for silent library output.

    Resolved by :meth:`ProjectConfig.env_bool`, so ``1``/``true``/``yes``/``on``
    mean quiet, ``0``/``false``/``no``/``off``/garbage mean loud, and the value
    is read at call time -- a process that clears the switch after import gets
    its output back.
    """
    return _CONFIG.env_bool("QUIET")


def _color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def red_banner(lines) -> None:
    """Print an alarm block (white on red background when on a TTY)."""
    if isinstance(lines, str):
        lines = [lines]
    if _color_enabled():
        for line in lines:
            print(f"\033[41;97;1m{line:<78}\033[0m", flush=True)
    else:
        for line in lines:
            print(f"!! {line}", flush=True)


def notice(text: str, *, stream: Optional[TextIO] = None) -> None:
    """Emit one library output line through the project's single flush point.

    Library modules used to call ``print()`` directly for their progress and
    notices (review 5.3): with no logging configured, an embedder importing
    ``deh`` as a library had no way to redirect, capture, or silence that
    output short of monkeypatching ``builtins.print``. This is the fallback
    channel for them.

    The primary channel is a callback: extractors, comparators and the
    quarantine manager take ``log_cb`` / ``progress_cb`` and route their output
    there. ``notice`` is what runs when the caller passed none -- the CLI,
    which *wants* the output.

    Args:
        text: The line to emit, verbatim (a leading ``\\n`` is kept, so a
            caller that wants a blank line before it gets one).
        stream: Where to write it (default ``sys.stdout``). Passing a stream
            captures output without touching global state.

    Silencing: ``DEH_QUIET`` in the environment suppresses every notice, the
    same way ``NO_COLOR`` suppresses ANSI here. It uses the project's boolean
    spellings (``1``/``true``/``yes``/``on``, case-insensitive), so
    ``DEH_QUIET=0`` means *not* quiet. Read at call time, so the switch is
    honoured by a process that sets it after import.
    """
    if _quiet():
        return
    out = stream if stream is not None else sys.stdout
    out.write(f"{text}\n")
    out.flush()
