"""Tests for the deh.core.terminal notice channel (review 5.3).

Library modules used to call ``print()`` directly, so an embedder importing
``deh`` as a library had no way to capture or silence that output. The
contract being pinned here: ``notice`` is the single fallback flush point, it
is redirectable through an argument, and ``DEH_QUIET`` silences it using the
project's boolean spellings.
"""
import io

import pytest

from deh.core.terminal import QUIET_ENV, _quiet, notice


class TestNoticeChannel:
    def test_writes_the_line_to_stdout(self, monkeypatch, capsys):
        monkeypatch.delenv(QUIET_ENV, raising=False)
        notice("[Hash] Hashed 50/200 images")
        assert capsys.readouterr().out == "[Hash] Hashed 50/200 images\n"

    def test_stream_argument_captures_output_without_touching_globals(self):
        buf = io.StringIO()
        notice("captured", stream=buf)
        assert buf.getvalue() == "captured\n"

    def test_leading_newline_is_preserved(self, monkeypatch, capsys):
        """Callers keep their blank-line-before-header formatting."""
        monkeypatch.delenv(QUIET_ENV, raising=False)
        notice("\n[Quarantine] Moving 2 image(s)")
        assert capsys.readouterr().out == "\n[Quarantine] Moving 2 image(s)\n"

    def test_quiet_switch_silences(self, monkeypatch, capsys):
        monkeypatch.setenv(QUIET_ENV, "1")
        notice("must not appear")
        assert capsys.readouterr().out == ""

    def test_switch_is_read_at_call_time(self, monkeypatch, capsys):
        """A process that clears the switch after import gets its output back."""
        monkeypatch.setenv(QUIET_ENV, "true")
        notice("quiet")
        assert capsys.readouterr().out == ""
        monkeypatch.delenv(QUIET_ENV)
        notice("loud")
        assert capsys.readouterr().out == "loud\n"


class TestQuietSwitch:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
    def test_truthy_values_silence(self, value, monkeypatch):
        monkeypatch.setenv(QUIET_ENV, value)
        assert _quiet()

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "banana"])
    def test_everything_else_stays_loud(self, value, monkeypatch):
        """``DEH_QUIET=0`` must mean *not* quiet, not 'quiet, spelled 0'."""
        monkeypatch.setenv(QUIET_ENV, value)
        assert not _quiet()

    def test_unset_is_loud(self, monkeypatch):
        monkeypatch.delenv(QUIET_ENV, raising=False)
        assert not _quiet()
