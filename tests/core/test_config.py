"""Tests for :mod:`deh.config` (ProjectConfig)."""

import dataclasses

import pytest

from deh.config import ProjectConfig


def test_root_is_cwd_independent(tmp_path, monkeypatch):
    """The resolved root must not depend on the current working directory."""
    before = ProjectConfig.load()
    monkeypatch.chdir(tmp_path)  # change CWD; the root must stay the same
    after = ProjectConfig.load()
    assert before.root == after.root
    assert before.root.is_absolute()


def test_subpaths_are_derived_from_root():
    cfg = ProjectConfig.load()
    assert cfg.var_dir == cfg.root / "var"
    assert cfg.models_dir == cfg.var_dir / "models"
    assert cfg.cache_dir == cfg.var_dir / "cache"
    assert cfg.env_file == cfg.root / ".env"


def test_explicit_root(tmp_path):
    cfg = ProjectConfig.load(root=tmp_path)
    assert cfg.root == tmp_path
    assert cfg.var_dir == tmp_path / "var"
    assert cfg.models_dir == tmp_path / "var" / "models"
    assert cfg.env_file == tmp_path / ".env"


def test_config_is_frozen():
    cfg = ProjectConfig.load()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.root = cfg.root / "nope"


class TestEnvAccess:
    """Environment variables are read through the DEH_ namespace."""

    def test_env_is_namespaced(self, monkeypatch):
        cfg = ProjectConfig.load()
        monkeypatch.setenv("DEH_DUPLICATE_MODE", "deep")
        monkeypatch.setenv("MODE", "hash")          # a foreign bare name
        assert cfg.env("DUPLICATE_MODE") == "deep"
        assert cfg.env("duplicate_mode") == "deep"  # suffix spelling is normalized
        assert cfg.env("MODE") is None              # the bare name is never read
        assert cfg.env("NO_SUCH_VAR") is None
        assert cfg.env("NO_SUCH_VAR", "fallback") == "fallback"

    def test_env_numbers(self, monkeypatch):
        cfg = ProjectConfig.load()
        monkeypatch.setenv("DEH_DUPLICATE_THRESHOLD", "62.5")
        monkeypatch.setenv("OTHER_DUPLICATE_THRESHOLD", "16384")  # not ours to read
        assert cfg.env_float("DUPLICATE_THRESHOLD") == pytest.approx(62.5)
        assert cfg.env_int("DUPLICATE_THRESHOLD") is None      # not an int
        assert cfg.env_int("DUPLICATE_THRESHOLD", 75) == 75    # ...so the default
        assert cfg.env_float("OTHER_DUPLICATE_THRESHOLD") is None   # needs the DEH_ prefix
        assert cfg.env_float("MISSING", 1.0) == 1.0


    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("NO", False), ("off", False),
    ])
    def test_env_bool_spellings(self, monkeypatch, raw, expected):
        cfg = ProjectConfig.load()
        monkeypatch.setenv("DEH_DUPLICATE_VERBOSE", raw)
        assert cfg.env_bool("DUPLICATE_VERBOSE") is expected

    def test_env_bool_garbage_keeps_the_default(self, monkeypatch):
        """A typo in .env must never flip a flag on its own."""
        cfg = ProjectConfig.load()
        monkeypatch.setenv("DEH_DUPLICATE_VERBOSE", "maybe")
        assert cfg.env_bool("DUPLICATE_VERBOSE") is False
        assert cfg.env_bool("DUPLICATE_VERBOSE", True) is True
        monkeypatch.delenv("DEH_DUPLICATE_VERBOSE")
        assert cfg.env_bool("DUPLICATE_VERBOSE", True) is True
