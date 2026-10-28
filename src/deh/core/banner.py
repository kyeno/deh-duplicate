# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Unified startup banner for deh-duplicate.

Provides a consistent, branded startup banner that displays the tool name,
version, and arbitrary configuration lines. Uses the same TTY/color detection
as :mod:`deh.core.terminal` for consistent output behavior.
"""

from __future__ import annotations

from .terminal import _color_enabled


def startup_banner(
    tool_name: str,
    version: str,
    config_lines: list[str] | None = None,
    *,
    out: callable = print,
) -> None:
    """Print a branded startup banner for a deh-duplicate run.

    Args:
        tool_name: Short name of the tool (e.g., "deh-duplicate").
        version: Version string from ``deh._meta.__version__``.
        config_lines: Optional list of ``"Key: Value"`` lines to display
            under a "Configuration:" header.
        out: Output callable (defaults to ``print``). Useful for testing or
            redirecting output.
    """
    # Build the banner content
    title = f" {tool_name} v{version} "
    border = "=" * max(60, len(title) + 4)

    if _color_enabled():
        # Cyan bold for the title, white on cyan for the border
        out(f"\033[36;1m{border}\033[0m")
        out(f"\033[36;1m{title:^{len(border)}}\033[0m")
        out(f"\033[36;1m{border}\033[0m")
    else:
        out(border)
        out(title.center(len(border)))
        out(border)

    if config_lines:
        out("")
        out("Configuration:")
        for line in config_lines:
            out(f"  {line}")
        out("")


def _demo() -> None:
    """Quick manual demo when run as ``python -m deh.core.banner``."""
    startup_banner(
        "deh-duplicate",
        "0.1.0",
        [
            "Directory:   /photos",
            "Threshold:   0.75 (75%)",
            "Mode:        HYBRID",
            "Output:      MOVE losers to _quarantine",
            "Verbose:     False",
        ],
    )


if __name__ == "__main__":
    _demo()
