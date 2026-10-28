"""Tests for deh.duplicate.quarantine move / dry-run behavior."""
from pathlib import Path

import pytest

from deh.duplicate.quarantine import QuarantineManager


def _mk(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 8)
    return str(p.resolve())


class TestDryRun:
    def test_reports_but_moves_nothing(self, tmp_path):
        src = _mk(tmp_path, "photos/a.jpg")
        qm = QuarantineManager(str(tmp_path))
        res = qm.quarantine([src], dry_run=True)

        assert res["moved"] == 1 and res["dry_run"] is True
        assert Path(src).exists()                          # original untouched
        dest = qm.quarantine_root / "photos" / "a.jpg"
        assert not dest.exists()                           # nothing created


class TestOutputRouting:
    """Output goes to the caller's ``log_cb`` / ``progress_cb`` (review 5.3)."""

    def test_log_cb_captures_the_lines_and_stdout_stays_clean(self, tmp_path, capsys):
        src = _mk(tmp_path, "photos/a.jpg")
        lines = []
        qm = QuarantineManager(str(tmp_path), log_cb=lines.append)
        qm.quarantine([src])

        assert capsys.readouterr().out == ""
        assert any("Moving 1 image(s)" in ln for ln in lines)
        assert any("Moved: photos/a.jpg" in ln for ln in lines)
        assert any("Done. 1 image(s) moved" in ln for ln in lines)

    def test_progress_cb_counts_each_processed_file(self, tmp_path):
        files = [_mk(tmp_path, f"p{i}.jpg") for i in range(3)]
        seen = []
        QuarantineManager(str(tmp_path), log_cb=lambda _text: None).quarantine(
            files, progress_cb=lambda done, total: seen.append((done, total)))
        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_progress_cb_advances_on_dry_run(self, tmp_path):
        files = [_mk(tmp_path, f"q{i}.jpg") for i in range(2)]
        seen = []
        QuarantineManager(str(tmp_path), log_cb=lambda _text: None).quarantine(
            files, dry_run=True, progress_cb=lambda done, total: seen.append((done, total)))
        assert seen == [(1, 2), (2, 2)]

    def test_error_lines_reach_the_log_cb_too(self, tmp_path, capsys):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("hi")
        lines = []
        res = QuarantineManager(str(root), log_cb=lines.append).quarantine([str(outside)])
        assert res["errors"] and any("ERROR" in ln for ln in lines)
        assert capsys.readouterr().out == ""


class TestMove:
    def test_moves_file_preserving_structure(self, tmp_path):
        src = _mk(tmp_path, "albums/2024/b.png")
        qm = QuarantineManager(str(tmp_path))
        res = qm.quarantine([src], dry_run=False)

        assert res["errors"] == [] and res["moved"] == 1
        assert not Path(src).exists()                      # original gone
        dest = qm.quarantine_root / "albums" / "2024" / "b.png"
        assert dest.exists()                               # moved in place

    def test_collision_gets_numeric_suffix(self, tmp_path):
        src = _mk(tmp_path, "c.jpg")
        qm = QuarantineManager(str(tmp_path))
        occupied = qm.quarantine_root / "c.jpg"            # pre-occupy the name
        occupied.parent.mkdir(parents=True, exist_ok=True)
        occupied.write_bytes(b"existing")

        qm.quarantine([src], dry_run=False)
        assert (qm.quarantine_root / "c_1.jpg").exists()

    def test_empty_list_short_circuits(self, tmp_path):
        qm = QuarantineManager(str(tmp_path))
        res = qm.quarantine([], dry_run=False)
        assert res["moved"] == 0 and res["dry_run"] is False


class TestErrors:
    def test_file_outside_root_is_an_error(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("hi")

        qm = QuarantineManager(str(root))
        res = qm.quarantine([str(outside)], dry_run=False)
        assert res["moved"] == 0
        assert len(res["errors"]) == 1


class TestDelete:
    def test_deletes_file(self, tmp_path):
        src = _mk(tmp_path, "albums/2024/b.png")
        qm = QuarantineManager(str(tmp_path))
        res = qm.quarantine([src], action="delete")

        assert res["errors"] == [] and res["deleted"] == 1
        assert not Path(src).exists()                      # original gone
        assert not qm.quarantine_root.exists()             # nothing moved anywhere

    def test_dry_run_with_delete_touches_nothing(self, tmp_path):
        src = _mk(tmp_path, "photos/a.jpg")
        qm = QuarantineManager(str(tmp_path))
        res = qm.quarantine([src], action="delete", dry_run=True)

        assert res["deleted"] == 1 and res["dry_run"] is True
        assert Path(src).exists()                          # untouched
        assert not qm.quarantine_root.exists()

    def test_outside_root_is_not_deleted(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("hi")

        qm = QuarantineManager(str(root))
        res = qm.quarantine([str(outside)], action="delete")
        assert res["deleted"] == 0
        assert len(res["errors"]) == 1
        assert outside.exists()                            # safety guard held


class TestActions:
    def test_invalid_action_raises(self, tmp_path):
        qm = QuarantineManager(str(tmp_path))
        with pytest.raises(ValueError):
            qm.quarantine([], action="copy")


class TestDestinationContainment:
    @pytest.mark.parametrize("nested", [False, True])
    def test_symlinked_destination_does_not_redirect_move(self, tmp_path, nested):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        src = Path(_mk(root, "album/image.jpg" if nested else "image.jpg"))
        quarantine = root / "_quarantine"
        if nested:
            quarantine.mkdir()
            (quarantine / "album").symlink_to(outside, target_is_directory=True)
        else:
            quarantine.symlink_to(outside, target_is_directory=True)

        result = QuarantineManager(str(root), log_cb=lambda _line: None).quarantine(
            [str(src)]
        )

        assert result["moved"] == 0
        assert result["errors"]
        assert src.exists()
        assert not (outside / "image.jpg").exists()
