"""End-to-end DuplicatePipeline mode acceptance tests (D2).

Runs the full ``run()`` flow with every collaborator stubbed except the real
QuarantineManager, so the filesystem effects of each mode are verified:
the default moves losers to _quarantine; --dry-run touches nothing even when
combined with --delete; and --delete removes losers permanently.
"""
from pathlib import Path

import pytest

from deh.duplicate.pipeline import DuplicateConfig, DuplicatePipeline
from deh.duplicate.quarantine import QuarantineManager
from deh.gui import Decision


def _make_images(root):
    a = root / "a.png"; b = root / "b.png"
    a.write_bytes(b"a" * 64)   # content irrelevant: assessor & comparator are stubs
    b.write_bytes(b"b" * 64)
    return [str(a), str(b)]


class _FakeAssessor:
    """Stub assessor: keeps the first path, fills the display score map."""

    def __init__(self, log_cb=None):
        pass

    def evaluate_with_notes(self, group, scores_out=None):
        if scores_out is not None:
            scores_out.update({p: 0.5 for p in group})
        return ([group[0]], list(group[1:]), {p: "" for p in group})

    def score_image(self, path):
        return 0.5


def _run_pipeline(root, paths, *, action="move", dry_run=False,
                  assessor_cls=None, approval_dialog=None, use_gui=False):
    lines = []

    class FakeScanner:
        def __init__(self, directory): pass
        def scan(self): return list(paths)

    class FakeHashComparator:
        def __init__(self, threshold=None, log_cb=None): pass
        def find_similar_groups(self, pool): return [list(pool[:2])]

    class FakeResolver:
        def __init__(self, directory, log_cb=None): pass
        def resolve(self, image_paths): return (list(image_paths), [])

    def _sink(*args):   # pipeline occasionally calls out() with no text
        lines.extend(str(a) for a in args)

    config = DuplicateConfig(
        directory=str(root), mode="hash", use_gui=use_gui, identical_dedupe=True,
        action=action, dry_run=dry_run,
    )
    kwargs = dict(scanner_cls=FakeScanner, hash_cls=FakeHashComparator,
                  resolver_cls=FakeResolver,
                  assessor_cls=assessor_cls or _FakeAssessor,
                  manager_cls=QuarantineManager, out=_sink)
    if approval_dialog is not None:
        kwargs["approval_dialog"] = approval_dialog
    pipeline = DuplicatePipeline(config, **kwargs)
    pipeline.run()
    return lines


class TestModeAcceptance:
    def test_default_moves_losers_to_quarantine(self, tmp_path):
        a, b = _make_images(tmp_path)
        lines = _run_pipeline(tmp_path, [a, b])

        assert Path(a).exists()                            # winner kept in place
        dest = tmp_path / "_quarantine" / "b.png"
        assert not Path(b).exists() and dest.exists()      # loser moved
        banner = "\n".join(lines)
        assert "Moved" in banner and "1" in banner  # completion summary

    def test_nested_similarity_loser_uses_scan_root_quarantine(self, tmp_path):
        root = tmp_path / "photos"
        root.mkdir()
        winner = root / "album" / "keeper.png"
        loser = root / "album" / "trip" / "loser.png"
        winner.parent.mkdir(parents=True)
        loser.parent.mkdir(parents=True)
        winner.write_bytes(b"winner")
        loser.write_bytes(b"loser")

        _run_pipeline(root, [str(winner), str(loser)])

        assert winner.exists()
        assert not loser.exists()
        assert (root / "_quarantine" / "album" / "trip" / "loser.png").exists()
        assert not (root / "album" / "_quarantine").exists()
        assert not (root / "album" / "trip" / "_quarantine").exists()

    def test_dry_run_touches_nothing_even_with_delete(self, tmp_path):
        a, b = _make_images(tmp_path)
        lines = _run_pipeline(tmp_path, [a, b], action="delete", dry_run=True)

        assert Path(a).exists() and Path(b).exists()       # nothing touched
        assert not (tmp_path / "_quarantine").exists()     # no side effects at all
        banner = "\n".join(lines)
        assert "would be" in banner and "1" in banner  # completion summary (dry-run)

    def test_delete_removes_losers_permanently(self, tmp_path):
        a, b = _make_images(tmp_path)
        lines = _run_pipeline(tmp_path, [a, b], action="delete")

        assert Path(a).exists()                            # winner kept in place
        assert not Path(b).exists()                        # loser gone for good
        assert not (tmp_path / "_quarantine").exists()
        banner = "\n".join(lines)
        assert "Deleted" in banner and "1" in banner  # completion summary

    @pytest.mark.parametrize(
        ("action", "dry_run"),
        [("move", False), ("move", True), ("delete", False), ("delete", True)],
    )
    def test_identical_prepass_respects_action_and_root(self, tmp_path, action, dry_run):
        root = tmp_path / "photos"
        root.mkdir()
        shallow = root / "a.png"
        nested = root / "album" / "b.png"
        keeper = root / "album" / "trip" / "c.png"
        unique = root / "unique.png"
        for path in (shallow, nested, keeper, unique):
            path.parent.mkdir(parents=True, exist_ok=True)
        for path in (shallow, nested, keeper):
            path.write_bytes(b"same image bytes")
        unique.write_bytes(b"distinct image bytes")

        class NoSimilarGroups:
            def __init__(self, threshold=None, log_cb=None):
                pass

            def find_similar_groups(self, pool):
                assert set(pool) == {str(keeper), str(unique)}
                return []

        cfg = DuplicateConfig(directory=str(root), mode="hash", use_gui=False,
                              action=action, dry_run=dry_run)
        DuplicatePipeline(cfg, hash_cls=NoSimilarGroups, out=lambda *_: None).run()

        assert keeper.exists() and unique.exists()
        assert shallow.exists() is dry_run
        assert nested.exists() is dry_run
        quarantine = root / "_quarantine"
        if action == "move" and not dry_run:
            assert (quarantine / "a.png").exists()
            assert (quarantine / "album" / "b.png").exists()
        else:
            assert not quarantine.exists()
        assert not (root / "album" / "_quarantine").exists()


    def test_library_output_arrives_through_the_pipeline_sink(self, tmp_path, capsys):
        """The real QuarantineManager's lines go to ``out``, not to stdout (5.3).

        The pipeline hands its own output sink to the library collaborators as
        their ``log_cb``, so an embedder that passed a custom ``out`` captures
        everything the CLI prints, and nothing leaks to the terminal.
        """
        a, b = _make_images(tmp_path)
        lines = _run_pipeline(tmp_path, [a, b])

        banner = "\n".join(lines)
        assert "[Quarantine] Moving 1 image(s)" in banner
        assert "Moved: b.png" in banner
        assert capsys.readouterr().out == ""


class TestPhase4ScoreMapPlumbing:
    """Phase 4 takes the GUI score map from the assessor's single scoring pass
    (``scores_out``) instead of re-running the metric set per image."""

    def test_dialog_receives_scores_filled_by_the_assessor(self, tmp_path):
        a, b = _make_images(tmp_path)
        seen = {}

        class FillingAssessor:
            def __init__(self, log_cb=None):
                pass

            def evaluate_with_notes(self, group, scores_out=None):
                if scores_out is not None:
                    scores_out.update({group[0]: 0.9, group[1]: 0.1})
                return ([group[0]], list(group[1:]), {p: "" for p in group})

            def score_image(self, path):
                raise AssertionError("pipeline must not re-score a filled map")

        def dialog(**kwargs):
            seen.update(kwargs)
            return (Decision.NOOP, kwargs["winners"], kwargs["losers"], [])

        _run_pipeline(tmp_path, [a, b], assessor_cls=FillingAssessor,
                      approval_dialog=dialog, use_gui=True)

        assert seen["scores"] == {a: 0.9, b: 0.1}
        assert Path(a).exists() and Path(b).exists()       # NOOP: nothing touched


class TestGuiUnavailableFailsClosed:
    """No display: leave unapproved images in place and report an error."""

    def test_losers_stay_put_when_the_gui_cannot_start(self, tmp_path, monkeypatch):
        a, b = _make_images(tmp_path)
        import deh.gui.duplicate as d

        def no_display():
            raise RuntimeError("could not find the Qt platform plugin 'xcb'")

        # The real show_group_approval is the pipeline's default dialog, so
        # patching its bootstrap reproduces a headless run end to end.
        monkeypatch.setattr(d, "ensure_app", no_display)
        with pytest.raises(RuntimeError, match="--no-gui"):
            _run_pipeline(tmp_path, [a, b], use_gui=True)

        assert Path(a).exists() and Path(b).exists()
        assert not (tmp_path / "_quarantine").exists()


class TestOperationErrors:
    def test_similarity_move_error_is_not_reported_as_success(self, tmp_path):
        a, b = _make_images(tmp_path)
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        (tmp_path / "_quarantine").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RuntimeError, match="file operation"):
            _run_pipeline(tmp_path, [a, b])

        assert Path(a).exists() and Path(b).exists()
        assert not (outside / "b.png").exists()

    def test_exact_prepass_error_is_not_hidden_by_early_return(self, tmp_path):
        a = tmp_path / "a.png"
        b = tmp_path / "nested" / "b.png"
        b.parent.mkdir()
        a.write_bytes(b"identical")
        b.write_bytes(b"identical")
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        (tmp_path / "_quarantine").symlink_to(outside, target_is_directory=True)

        cfg = DuplicateConfig(directory=str(tmp_path), mode="hash", use_gui=False)
        with pytest.raises(RuntimeError, match="file operation"):
            DuplicatePipeline(cfg, out=lambda *_: None).run()

        assert a.exists() and b.exists()
        assert not list(outside.iterdir())


class TestDevicePhases:
    """Each pipeline phase announces itself to ``core.device``.

    ``pick_device`` runs per hash batch / per distance matrix, so on a GPU held
    by an LLM server the red CPU-fallback banner used to repeat on every call.
    The alarm is re-armed at phase boundaries, so a run says it once per phase.
    """

    def test_each_phase_announces_its_device_phase(self, tmp_path, monkeypatch):
        from deh.duplicate import pipeline as pipeline_mod

        a, b = _make_images(tmp_path)
        phases = []
        monkeypatch.setattr(pipeline_mod, "begin_phase",
                            lambda name: phases.append(name))
        _run_pipeline(tmp_path, [a, b])
        assert phases == ["scan", "identical-dedupe", "hashing", "quality+review"]
