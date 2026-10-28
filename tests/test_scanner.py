"""Tests for deh.core.scanner recursive image discovery & filtering."""
import pytest

from deh.core.scanner import ImageScanner


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 4)   # content irrelevant; ext matters


class TestImageScanner:
    def test_finds_images_recursively_and_filters(self, tmp_path):
        _touch(tmp_path / "a.jpg")
        _touch(tmp_path / "sub" / "b.png")
        (tmp_path / "notes.txt").write_text("x")                 # not an image
        q = tmp_path / "_quarantine"; q.mkdir()                  # skipped dir
        (q / "d.jpg").write_bytes(b"x")
        h = tmp_path / ".hidden"; h.mkdir()                       # hidden -> skipped
        (h / "e.jpg").write_bytes(b"x")

        found = {p.name for p in ImageScanner(str(tmp_path)).scan()}
        assert found == {"a.jpg", "b.png"}

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ImageScanner(str(tmp_path / "does_not_exist"))

    def test_returns_absolute_sorted_paths(self, tmp_path):
        _touch(tmp_path / "z.jpg")
        _touch(tmp_path / "a.jpg")
        paths = ImageScanner(str(tmp_path)).scan()
        assert all(p.is_absolute() for p in paths)
        assert paths == sorted(paths)
