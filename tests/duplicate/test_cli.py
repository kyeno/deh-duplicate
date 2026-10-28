"""Tests for CLI argument parsing & config resolution (incl. --no-gui default)."""
import argparse
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from deh.cli import duplicate as main


def _ns(**kw):
    """A parsed-args Namespace with *kw* applied on top of the real defaults.

    Built from ``parse_args()`` itself, not a hand-written attribute list: the
    attribute set is whatever the parser actually declares, so renaming or
    dropping a flag breaks these tests loudly instead of leaving
    ``resolve_config`` reading an attribute no real invocation ever produces.
    """
    base = vars(main.parse_args(["./x"]))
    base.update(kw)
    return argparse.Namespace(**base)


ENV_VARS = ("DEH_DUPLICATE_THRESHOLD", "DEH_DUPLICATE_MODE", "DEH_DUPLICATE_DRY_RUN",
            "DEH_DUPLICATE_VERBOSE", "DEH_DUPLICATE_IDENTICAL_DEDUPE")


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "./samples"])
        ns = main.parse_args()
        assert ns.directory == "./samples"
        assert ns.no_gui is False
        assert ns.threshold is None and ns.mode is None

    def test_flags_parsed(self, monkeypatch):
        argv = ["prog", "./s", "--no-gui", "-t", "90",
                "--mode", "deep", "--delete"]
        monkeypatch.setattr(sys, "argv", argv)
        ns = main.parse_args()
        assert ns.no_gui is True
        assert ns.threshold == 90.0
        assert ns.mode == "deep"
        assert ns.delete is True
        assert ns.dry_run is False

    def test_dry_run_flag_parsed(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "./s", "--dry-run"])
        ns = main.parse_args()
        assert ns.dry_run is True and ns.delete is False


def test_cli_exits_nonzero_when_pipeline_reports_failure(monkeypatch, capsys):
    class FailingPipeline:
        def __init__(self, config, out):
            pass

        def run(self):
            raise RuntimeError("move failed")

    monkeypatch.setattr(main, "DuplicatePipeline", FailingPipeline)
    monkeypatch.setattr(main, "startup_banner", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["deh-duplicate", "./photos"])

    with pytest.raises(SystemExit) as exited:
        main.main()

    assert exited.value.code == 1
    assert "move failed" in capsys.readouterr().err


def test_terminal_bars_only_for_gui_mode(monkeypatch):
    from types import SimpleNamespace

    from deh.cli import progress as progress_module
    from deh.core.terminal import notice

    calls = []

    class FakeProgress:
        def __init__(self):
            self.log = lambda message: None
            self.update = lambda stage, done, total: None
            calls.append(("open", self))

        def close(self):
            calls.append(("close", self))

    class FakePipeline:
        def __init__(self, config, **kwargs):
            calls.append(("pipeline", config, kwargs))

        def run(self):
            pass

    monkeypatch.setattr(progress_module, "ProgressBars", FakeProgress)
    monkeypatch.setattr(main, "DuplicatePipeline", FakePipeline)
    monkeypatch.setattr(main, "startup_banner", lambda *args: None)
    monkeypatch.setenv("DEH_QUIET", "false")

    for flags, uses_bars in (([], True), (["--no-gui"], False)):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["deh-duplicate", "./photos", *flags])
        monkeypatch.setattr(main, "sys", SimpleNamespace(
            stdout=SimpleNamespace(isatty=lambda: True),
            stderr=io.StringIO(),
            exit=sys.exit,
        ))
        main.main()
        pipeline_call = next(call for call in calls if call[0] == "pipeline")
        kwargs = pipeline_call[2]
        if uses_bars:
            assert kwargs["progress_cb"] is not None
            assert [call[0] for call in calls if call[0] != "pipeline"] == ["open", "close"]
        else:
            assert kwargs == {"out": notice}
            assert all(call[0] == "pipeline" for call in calls)


def test_shell_launcher_keeps_callers_working_directory(tmp_path):
    project = tmp_path / "project"
    launcher = project / "bin" / "deh-duplicate.sh"
    launcher.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parents[2] / "bin" / "deh-duplicate.sh", launcher)
    interpreter = project / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nprintf 'cwd=%s\\nscan=%s\\n' \"$PWD\" \"$3\"\n")
    interpreter.chmod(0o755)
    caller = tmp_path / "caller"
    caller.mkdir()

    result = subprocess.run([str(launcher), "photos"], cwd=caller,
                            capture_output=True, text=True, check=True)

    assert f"cwd={caller}" in result.stdout
    assert "scan=photos" in result.stdout


class TestResolveConfig:
    def test_no_gui_flag_disables_gui(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert main.resolve_config(_ns(no_gui=True)).use_gui is False
        # GUI enabled by default (regression guard for the --no-gui fallback).
        assert main.resolve_config(_ns(no_gui=False)).use_gui is True

    def test_threshold_cli_overrides_env_and_default(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert main.resolve_config(_ns(threshold=98)).threshold == pytest.approx(0.98)

        monkeypatch.setenv("DEH_DUPLICATE_THRESHOLD", "60")   # env fallback
        assert main.resolve_config(_ns()).threshold == pytest.approx(0.60)

        monkeypatch.delenv("DEH_DUPLICATE_THRESHOLD", raising=False)   # built-in default
        assert main.resolve_config(_ns()).threshold == 0.75

    def test_mode_defaults_to_hybrid(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert main.resolve_config(_ns()).mode == "hybrid"
        assert main.resolve_config(_ns(mode="deep")).mode == "deep"

    def test_deep_mode_uses_82_by_default_despite_template_hash_setting(self, monkeypatch):
        monkeypatch.setenv("DEH_DUPLICATE_THRESHOLD", "75")
        monkeypatch.setenv("DEH_DUPLICATE_DEEP_THRESHOLD", "82")
        assert main.resolve_config(_ns(mode="deep")).threshold == pytest.approx(0.82)
        monkeypatch.delenv("DEH_DUPLICATE_DEEP_THRESHOLD")
        assert main.resolve_config(_ns(mode="deep")).threshold == pytest.approx(0.82)
        monkeypatch.setenv("DEH_DUPLICATE_DEEP_THRESHOLD", "82")
        assert main.resolve_config(_ns(mode="hash")).threshold == pytest.approx(0.75)
        hybrid = main.resolve_config(_ns(mode="hybrid"))
        assert hybrid.threshold == pytest.approx(0.75)
        assert hybrid.deep_threshold == pytest.approx(0.82)

    def test_deep_mode_explicit_thresholds_override_default(self, monkeypatch):
        monkeypatch.setenv("DEH_DUPLICATE_DEEP_THRESHOLD", "88")
        assert main.resolve_config(_ns(mode="deep")).threshold == pytest.approx(0.88)
        assert main.resolve_config(_ns(mode="deep", threshold=90)).threshold == pytest.approx(0.90)
        assert main.resolve_config(_ns(mode="deep", deep_threshold=95)).threshold == pytest.approx(0.95)
        assert main.resolve_config(_ns(mode="deep", threshold=90,
                                       deep_threshold=95)).threshold == pytest.approx(0.95)

    def test_cli_threshold_affects_default_hybrid_mode(self, monkeypatch):
        monkeypatch.setenv("DEH_DUPLICATE_DEEP_THRESHOLD", "82")
        cfg = main.resolve_config(_ns(threshold=90))
        assert cfg.mode == "hybrid"
        assert cfg.deep_threshold == pytest.approx(0.90)

        cfg = main.resolve_config(_ns(threshold=90, deep_threshold=95))
        assert cfg.deep_threshold == pytest.approx(0.95)

    def test_action_polarity_defaults_to_moving(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        cfg = main.resolve_config(_ns())
        assert cfg.action == "move"          # moving losers is the default now
        assert cfg.dry_run is False

    def test_delete_flag_selects_deletion(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        cfg = main.resolve_config(_ns(delete=True))
        assert cfg.action == "delete" and cfg.dry_run is False

    def test_dry_run_overrides_everything_including_delete(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        cfg = main.resolve_config(_ns(dry_run=True, delete=True))
        assert cfg.action == "delete" and cfg.dry_run is True

    def test_stale_dry_run_env_does_not_change_default_action(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEH_DUPLICATE_DRY_RUN", "true")
        cfg = main.resolve_config(_ns())
        assert cfg.dry_run is False and cfg.action == "move"

    def test_bare_legacy_env_names_are_ignored(self, monkeypatch):
        """The pre-namespace spellings must not leak into the config."""
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("THRESHOLD", "10")
        monkeypatch.setenv("MODE", "hash")
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("VERBOSE", "true")
        monkeypatch.setenv("IDENTICAL_DEDUPE", "false")
        cfg = main.resolve_config(_ns())
        assert cfg.threshold == 0.75
        assert cfg.mode == "hybrid"
        assert cfg.dry_run is False
        assert cfg.verbose is False
        assert cfg.identical_dedupe is True

    def test_undeclared_flag_is_a_loud_error_not_a_default(self, monkeypatch):
        """``resolve_config`` has no getattr fallbacks (review 5.2).

        The fixture above is built from ``parse_args()``, so every attribute
        ``resolve_config`` reads exists in a real invocation. Deleting one here
        must raise: a renamed or dropped flag has to break loudly, never be
        papered over by a default the user never asked for.
        """
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        ns = _ns()
        del ns.deep_threshold
        with pytest.raises(AttributeError):
            main.resolve_config(ns)


class TestIdenticalDedupe:
    def test_enabled_by_default(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert main.resolve_config(_ns()).identical_dedupe is True

    def test_flag_disables_it(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert main.resolve_config(_ns(no_identical_dedupe=True)).identical_dedupe is False

    def test_env_false_disables_it(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEH_DUPLICATE_IDENTICAL_DEDUPE", "false")
        assert main.resolve_config(_ns()).identical_dedupe is False

    def test_cli_flag_beats_env_true(self, monkeypatch):
        for var in ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEH_DUPLICATE_IDENTICAL_DEDUPE", "true")
        assert main.resolve_config(_ns(no_identical_dedupe=True)).identical_dedupe is False
