# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Project-wide path configuration.

This module defines project runtime paths such as the model cache and .env file.
They are derived from the *project root* — the directory containing ``src/``
and ``var/`` — resolved from this file's location. The directory supplied for
an image scan is separate and resolves from the caller's working directory.

Import this from anywhere that needs a project path::

    from deh.config import ProjectConfig
    cfg = ProjectConfig.load()
    cfg.cache_dir      # <root>/var/cache
"""


from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (the parent of ``src/``).

    This file lives at ``<root>/src/deh/config.py``, so the root is two levels
    up from the package directory.
    """
    return Path(__file__).resolve().parents[2]


#: Prefix of every environment variable this project owns.
#:
#: Bare names (``THRESHOLD``, ``MODE``, ``DRY_RUN``, ``VERBOSE``,
#: ``IDENTICAL_DEDUPE``...) collide with variables that shells, CI runners and
#: neighbouring projects already define, so every variable deh-duplicate reads is
#: namespaced: ``DEH_DUPLICATE_THRESHOLD``, ``DEH_GUI_SCALE``, ...  Read them
#: through :meth:`ProjectConfig.env` and its typed helpers rather than
#: ``os.environ`` directly.  ``NO_COLOR`` and ``TORCH_HOME`` are third-party
#: conventions that must keep their exact names.
ENV_PREFIX = "DEH_"


#: Spellings accepted by :meth:`ProjectConfig.env_bool` (case-insensitive).
_TRUTHY = frozenset(("1", "true", "yes", "on"))
_FALSY = frozenset(("0", "false", "no", "off"))


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable, CWD-independent map of the paths the project uses.

    Attributes
    ----------
    root:
        Project root (contains ``src/`` and ``var/``).
    var_dir:
        Runtime-data root: ``<root>/var``.
    models_dir:
        Model weights: ``<root>/var/models`` (also used as ``TORCH_HOME``).
    cache_dir:
        Derived caches: ``<root>/var/cache``.
    env_file:
        Project-level ``.env``: ``<root>/.env``.


    The same object is the entry point for the project's environment variables:
    :meth:`env` (and :meth:`env_float`, :meth:`env_int`, :meth:`env_bool`)
    read them under the :data:`ENV_PREFIX` namespace, so no module has to
    spell out a bare ``os.environ`` name of its own.
    """

    root: Path
    var_dir: Path
    models_dir: Path
    cache_dir: Path
    env_file: Path


    @classmethod
    def load(cls, root: Path | str | None = None) -> "ProjectConfig":
        """Build a config rooted at *root*.

        When *root* is ``None`` the real project root is used (resolved from
        this file's location). Pass an explicit *root* (e.g. a temp dir) in
        tests.
        """
        root = Path(root).resolve() if root is not None else _project_root()
        var = root / "var"
        models = var / "models"
        return cls(
            root=root,
            var_dir=var,
            models_dir=models,
            cache_dir=var / "cache",
            env_file=root / ".env",
        )


    # ------------------------------------------------------------- env vars
    def env(self, name: str, default: str | None = None) -> str | None:
        """Read the namespaced variable ``DEH_<NAME>``.

        *name* is the project-owned suffix (``"DUPLICATE_MODE"`` reads
        ``DEH_DUPLICATE_MODE``); it is upper-cased so call sites can write it
        in the conventional env spelling.
        """
        return os.environ.get(f"{ENV_PREFIX}{name.upper()}", default)

    def env_float(self, name: str, default: float | None = None) -> float | None:
        """:meth:`env` as a float; *default* when unset or not a number."""
        raw = self.env(name)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def env_int(self, name: str, default: int | None = None) -> int | None:
        """:meth:`env` as an int; *default* when unset or not a number."""
        raw = self.env(name)
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def env_bool(self, name: str, default: bool = False) -> bool:
        """:meth:`env` as a flag: ``1/true/yes/on`` and ``0/false/no/off``.

        Anything else (including an unset variable) resolves to *default*, so
        a typo in ``.env`` can never flip a flag on its own.
        """
        raw = self.env(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in _TRUTHY:
            return True
        if value in _FALSY:
            return False
        return default


#: Process-wide default config, resolved once at import time.
DEFAULT = ProjectConfig.load()
