"""Tests for deh.core.imageio pure formatting/metadata helpers."""

from deh.core.imageio import human_bytes


class TestHumanBytes:

    def test_human_bytes_units(self):
        assert human_bytes(512) == "512 B"
        assert human_bytes(1024 * 1024) == "1.0 MB"
        assert human_bytes(int(2.5 * 1024 * 1024)) == "2.5 MB"

    def test_human_bytes_invalid_returns_placeholder(self):
        assert human_bytes(None) != ""
