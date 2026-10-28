"""Tests for deh.gui.duplicate decision flow: approve / noop / quit, and image loading.

These exercise the exact bug that caused ENTER to crash (unscoped ``Qt.Key_*``
constants) by dispatching real synthetic QKeyEvents through keyPressEvent.
"""
import pytest
from PIL import Image

from deh.gui import (Decision, GroupApprovalDialog, show_group_approval,
                     ImageCard)


def _make_img(path, color=(120, 40, 200), size=(320, 240)):
    """Create a small valid image with some detail so it loads into cards."""
    img = Image.new("RGB", size, color)
    px = img.load()
    for y in range(0, size[1], 8):          # horizontal stripes -> non-degenerate
        for x in range(size[0]):
            px[x, y] = (255, 255, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)


@pytest.fixture
def group_data(tmp_path):
    a = _make_img(tmp_path / "a.jpg", (30, 90, 200))
    b = _make_img(tmp_path / "b.jpg", (200, 60, 60))
    c = _make_img(tmp_path / "c.png", (40, 160, 90))
    scores = {a: 0.9, b: 0.7, c: 0.5}
    return dict(paths=[a, b, c], winners=[a], losers=[b, c], scores=scores)


class TestDecisionEnum:
    def test_values_are_distinct(self):
        assert Decision.APPROVED.value == "approved"
        assert Decision.NOOP.value == "noop"
        assert Decision.QUIT.value == "quit"
        assert len(list(Decision)) == 3


class TestKeyboardDecisions:
    """Enter keeps the card under the cursor and approves; Esc/Backspace noop.

    The cursor starts on the scorer's winner, so a bare Enter approves exactly the
    scored partition (see TestSpaceAndCursorKeys for the keeper-at-cursor cases).
    """

    @pytest.mark.parametrize("key_name,expected", [
        ("Key_Return", Decision.APPROVED),
        ("Key_Enter", Decision.APPROVED),      # numpad Enter counts too

        ("Key_Escape", Decision.NOOP),
        ("Key_Backspace", Decision.NOOP),
    ])
    def test_keypress_sets_decision(self, qapp, group_data, key_name, expected):
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt, QEvent

        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        # Isolate the decision logic from Qt's close machinery (no event loop here).
        dlg.accept = lambda: None
        dlg.reject = lambda: None

        ev = QKeyEvent(QEvent.Type.KeyPress, getattr(Qt.Key, key_name),
                       Qt.KeyboardModifier.NoModifier)
        dlg.keyPressEvent(ev)

        assert dlg.decision is expected


class TestCloseButtonBehavior:
    def test_close_without_decision_is_quit(self, qapp, group_data):
        from PyQt6.QtGui import QCloseEvent

        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        assert dlg.decision is None
        dlg.closeEvent(QCloseEvent())
        assert dlg.decision is Decision.QUIT

    def test_existing_decision_preserved_on_close(self, qapp, group_data):
        from PyQt6.QtGui import QCloseEvent

        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        dlg.decision = Decision.APPROVED
        dlg.closeEvent(QCloseEvent())
        assert dlg.decision is Decision.APPROVED


class TestImageLoading:
    def test_all_cards_receive_nonnull_pixmap(self, qapp, group_data):
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        for i in range(len(group_data["paths"])):
            card = getattr(dlg, f"_card_{i}")
            pm = card.image_label.pixmap()
            assert pm is not None and not pm.isNull(), \
                f"card {i} failed to load its image into a pixmap"

class TestShowGroupApprovalMapping:
    def test_approved_returns_approved(self, qapp, group_data, monkeypatch):
        import deh.gui as g
        monkeypatch.setattr(g.GroupApprovalDialog, "exec",
                            lambda self: setattr(self, "decision", Decision.APPROVED))
        result = g.show_group_approval(**group_data, group_index=0)
        assert result[0] == Decision.APPROVED

    def test_noop_returns_noop(self, qapp, group_data, monkeypatch):
        import deh.gui as g
        monkeypatch.setattr(g.GroupApprovalDialog, "exec",
                            lambda self: setattr(self, "decision", Decision.NOOP))
        result = g.show_group_approval(**group_data, group_index=1)
        assert result[0] == Decision.NOOP

    def test_quit_exits_application(self, qapp, group_data, monkeypatch):
        import deh.gui as g
        monkeypatch.setattr(g.GroupApprovalDialog, "exec",
                            lambda self: setattr(self, "decision", Decision.QUIT))
        with pytest.raises(SystemExit):
            g.show_group_approval(**group_data, group_index=2)


class TestHeaderGroupProgression:
    """The top header must read exactly 'Group X of Y' when totals are known."""

    def test_header_shows_group_x_of_y(self, qapp, group_data):
        dlg = GroupApprovalDialog("t", group_data["paths"], group_data["winners"],
                                  group_data["losers"], group_data["scores"],
                                  group_number=3, total_groups=8)
        assert dlg.header_label.text() == "Group 3 of 8"

    def test_header_survives_selection_changes(self, qapp, group_data):
        d = group_data
        dlg = GroupApprovalDialog("t", d["paths"], d["winners"], d["losers"], d["scores"],
                                  group_number=3, total_groups=8)
        dlg.select_winner(d["losers"][0])     # triggers _refresh_card
        dlg.exclude_image(d["losers"][1])     # ...and again here
        assert dlg.header_label.text() == "Group 3 of 8"

    def test_fallback_wording_when_totals_unknown(self, qapp, group_data):
        dlg = GroupApprovalDialog("t", group_data["paths"], group_data["winners"],
                                  group_data["losers"], group_data["scores"])
        text = dlg.header_label.text()
        assert "image(s)" in text and "winner(s)" in text   # legacy wording preserved

    def test_show_group_approval_forwards_progression(self, qapp, group_data, monkeypatch):
        import deh.gui as g
        instances = []
        def fake_exec(self):
            self.decision = Decision.APPROVED
            instances.append(self)
        monkeypatch.setattr(g.GroupApprovalDialog, "exec", fake_exec)
        result = g.show_group_approval(**group_data, group_index=2, total_groups=9)
        assert result[0] == Decision.APPROVED
        dlg = instances[-1]
        assert dlg.group_number == 3          # index is 0-based -> display is 1-based
        assert dlg.total_groups == 9
        assert dlg.header_label.text() == "Group 3 of 9"

def _click(card, button_name):
    """Dispatch a synthetic mouse press on `card` (button_name: 'Left'/'Right')."""
    from PyQt6.QtCore import Qt, QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    btn = getattr(Qt.MouseButton, f"{button_name}Button")
    local_pos = QPointF(card.rect().center())
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, local_pos, QPointF(0, 0),
                     btn, btn, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(card, ev)


class TestMouseInteraction:
    """LMB selects the keeper; RMB excludes/restores — including all edge cases."""

    def test_left_click_makes_clicked_image_the_winner(self, qapp, group_data):
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        b = group_data["paths"][1]          # originally a loser
        _click(dlg._card_1, "Left")
        assert dlg.state[b] == "winner"
        assert dlg.state[group_data["paths"][0]] == "loser"   # old winner demoted
        assert dlg._card_1.objectName() == "winnerCard"
        assert dlg._card_0.objectName() == "loserCard"
        w, l, e = dlg.final_partition()
        assert w == [b] and len(l) == 2 and e == []
        assert "\u2605" in dlg._card_1.score_label.text()
        assert "\u2605" not in dlg._card_0.score_label.text()
        assert "Selected" in dlg.status_label.text()

    def test_left_click_on_current_winner_is_noop(self, qapp, group_data):
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        before = dict(dlg.state)
        status_before = dlg.status_label.text()
        _click(dlg._card_0, "Left")
        assert dlg.state == before                    # nothing changed at all
        assert dlg.status_label.text() == status_before
        assert dlg._card_0.objectName() == "winnerCard"

    def test_right_click_excludes_loser_with_red_border(self, qapp, group_data):
        from deh.gui import DARK_THEME_CSS
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        c = group_data["paths"][2]
        _click(dlg._card_2, "Right")
        assert dlg.state[c] == "excluded"
        card = dlg._card_2
        assert card.objectName() == "excludedCard"     # red border rule applies
        assert "#excludedcard" in DARK_THEME_CSS.lower()
        w, l, e = dlg.final_partition()
        assert c not in w and c not in l and e == [c]  # neither kept nor quarantined
        assert "Excluded" in dlg.status_label.text()

    def test_right_click_on_the_keeper_excludes_it_and_promotes_the_runner_up(self, qapp, group_data):
        """The keeper is excludable: it leaves the group and the keeper role goes to
        the highest-scoring image left, so the group is never left keeper-less."""
        a, b, c = group_data["paths"]
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        _click(dlg._card_0, "Right")                    # drop the scored keeper
        assert dlg.state[a] == "excluded"
        assert dlg._card_0.objectName() == "excludedCard"
        assert dlg.state[b] == "winner"                 # runner-up (0.7) inherited it
        assert dlg.state[c] == "loser"
        assert "\u2605" in dlg._card_1.score_label.text()      # star follows the role
        assert "\u2605" not in dlg._card_0.score_label.text()
        w, l, e = dlg.final_partition()
        assert (w, e) == ([b], [a]) and l == [c]
        assert "keeper moved to b.jpg" in dlg.status_label.text()
        assert dlg._card_0.property("keeper") is True   # box stays on the card clicked
        assert dlg._card_1.property("keeper") is False
        assert dlg.decision is None                     # Space/RMB never approves

    def test_excluding_the_last_keeper_leaves_the_group_keeperless(self, qapp, group_data):
        """Every image excluded -> no keeper at all: Enter must not approve it."""
        a, b, c = group_data["paths"]
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        _click(dlg._card_1, "Right")                    # exclude b, a keeps the role
        _click(dlg._card_2, "Right")                    # ...and c
        assert dlg.state[a] == "winner"
        _click(dlg._card_0, "Right")                    # drop the last keeper-eligible one
        assert dlg.state[a] == "excluded"
        w, l, e = dlg.final_partition()
        assert w == [] and l == [] and sorted(e) == sorted([a, b, c])
        assert "no keeper left" in dlg.status_label.text()
        _click(dlg._card_0, "Left")                     # left-click is an explicit request
        assert dlg.state[a] == "winner"                 # ...so it re-keeps the dropped one
        assert dlg.state[b] == "excluded" and dlg.state[c] == "excluded"
        w, l, e = dlg.final_partition()
        assert w == [a] and l == [] and sorted(e) == sorted([b, c])

    def test_right_click_toggles_exclusion_back_to_loser(self, qapp, group_data):
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        b = group_data["paths"][1]
        card = dlg._card_1
        _click(card, "Right")                          # loser -> excluded
        assert dlg.state[b] == "excluded" and card.objectName() == "excludedCard"
        _click(card, "Right")                          # excluded -> restored to loser
        assert dlg.state[b] == "loser" and card.objectName() == "loserCard"
        w, l, e = dlg.final_partition()
        assert b in l and e == []
        assert "Restored" in dlg.status_label.text()

    def test_left_click_on_excluded_image_promotes_it_to_winner(self, qapp, group_data):
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        c = group_data["paths"][2]
        _click(dlg._card_2, "Right")       # exclude first...
        assert dlg.state[c] == "excluded"
        _click(dlg._card_2, "Left")        # ...then select it as keeper
        assert dlg.state[c] == "winner"                # un-excluded AND promoted
        assert dlg._card_2.objectName() == "winnerCard"
        w, l, e = dlg.final_partition()
        assert w == [c] and e == []

    def test_final_partition_is_always_exact_disjoint_cover(self, qapp, group_data):
        dlg = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        paths = group_data["paths"]
        for card_idx in (1, 0, 2, 1, 2):               # mixed click sequence
            btn = "Left" if card_idx % 2 else "Right"
            _click(getattr(dlg, f"_card_{card_idx}"), btn)
            w, l, e = dlg.final_partition()
            combined = sorted(w + l + e)
            assert combined == sorted(paths), \
                "partition must cover every image exactly once"
            assert len(set(w)) == len(w) and not (set(l) & set(e))
            assert sum(len(x) for x in (w, l, e)) == len(paths)

    def test_all_losers_excluded_then_approved_quarantines_nothing(self, qapp, group_data):
        d = GroupApprovalDialog(
            "t", group_data["paths"], group_data["winners"],
            group_data["losers"], group_data["scores"])
        for i in range(1, len(group_data["paths"])):   # exclude every loser
            _click(getattr(d, f"_card_{i}"), "Right")
        d.decision = Decision.APPROVED                 # approve with zero losers left
        result = (d.decision, *d.final_partition())
        assert result[0] is Decision.APPROVED
        assert result[2] == []                          # nothing to quarantine
        assert len(result[3]) == 2                       # both kept in place via exclusion

class TestScannerStylePathObjects:
    """The real pipeline feeds pathlib.PosixPath objects into the dialog (see
    ImageScanner.scan -> filepath.resolve()). Clicks must map onto group state no
    matter what type the caller used; previously str(Path(x)) callbacks missed the
    Path-keyed state dict and a re-selected winner silently vanished."""

    def _two_img_group(self, tmp_path):
        a = _make_img(tmp_path / "a.jpg", (30, 90, 200))
        b = _make_img(tmp_path / "b.jpg", (200, 60, 60))
        return a, b

    def test_clicks_work_when_paths_are_posixpath_objects(self, qapp, tmp_path):
        from pathlib import Path as P
        a, b = self._two_img_group(tmp_path)
        pa, pb = P(a), P(b)                       # scanner-style PosixPath inputs
        dlg = GroupApprovalDialog("t", [pa, pb], [pa], [pb], {pa: 0.9, pb: 0.5})
        assert dlg._card_1.objectName() == "loserCard"   # initial mapping intact
        assert dlg._card_1.score_label.text().endswith("\u2605") is False
        _click(dlg._card_1, "Left")                      # select b as keeper
        w, l, e = dlg.final_partition()
        assert len(w) == 1 and str(w[0]) == str(pb),             f"clicked image must become THE winner, got winners={w}"
        assert sorted(str(x) for x in l + e) == [str(pa)]            # old winner demoted
        assert dlg._card_1.objectName() == "winnerCard"
        assert dlg._card_0.objectName() == "loserCard"

    def test_returned_lists_are_plain_strings(self, qapp, tmp_path):
        from pathlib import Path as P
        a, b = self._two_img_group(tmp_path)
        dlg = GroupApprovalDialog("t", [P(a), P(b)], [P(a)], [P(b)], {P(a): 0.9})
        w, l, e = dlg.final_partition()
        assert all(isinstance(x, str) for x in w + l + e)


class TestArrowNavigation:
    """← → ↑ ↓ move the keyboard cursor between candidates without changing state."""

    @staticmethod
    def _press(dlg, key_name):
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt, QEvent
        ev = QKeyEvent(QEvent.Type.KeyPress, getattr(Qt.Key, key_name),
                       Qt.KeyboardModifier.NoModifier)
        dlg.keyPressEvent(ev)

    @staticmethod
    def _dialog(data):
        return GroupApprovalDialog("t", data["paths"], data["winners"],
                                   data["losers"], data["scores"])

    def test_right_arrow_moves_cursor_forward(self, qapp, group_data):
        dlg = self._dialog(group_data)
        assert dlg._card_0.property("highlighted") is True   # starts on winner
        before = dict(dlg.state)
        status_before = dlg.status_label.text()
        self._press(dlg, "Key_Right")
        assert dlg.cursor == 1
        assert dlg._card_1.property("highlighted") is True
        assert dlg._card_0.property("highlighted") is False
        assert dlg.state == before                      # moving the cursor changes nothing
        w, l, e = dlg.final_partition()
        assert w == [group_data["paths"][0]] and len(l) == 2 and e == []
        assert dlg.status_label.text() == status_before

    def test_left_arrow_wraps_to_last_card(self, qapp, group_data):
        dlg = self._dialog(group_data)
        self._press(dlg, "Key_Left")             # from index 0 wraps around to last
        assert dlg.cursor == 2
        assert dlg._card_2.property("highlighted") is True
        assert dlg._card_0.property("highlighted") is False

    def test_down_jumps_one_row_and_up_wraps(self, qapp, tmp_path):
        paths = [_make_img(tmp_path / f"img{i}.jpg") for i in range(8)]   # 4 cols x 2 rows
        scores = {p: round(1.0 - 0.1 * i, 3) for i, p in enumerate(paths)}
        data = dict(paths=paths, winners=[paths[0]], losers=paths[1:], scores=scores)
        dlg = self._dialog(data)
        assert dlg._grid_cols == 4
        self._press(dlg, "Key_Down")
        assert dlg.cursor == 4                          # one full row down
        self._press(dlg, "Key_Up")
        assert dlg.cursor == 0                          # wrapped back to top-left

    def test_cursor_lands_on_excluded_card_too(self, qapp, group_data):
        """Excluded cards stay targetable so Space can restore them from the keyboard."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        dlg._on_card_excluded(b)                        # middle card excluded (cursor parks on it)
        assert dlg.cursor == 1
        self._press(dlg, "Key_Right")                   # moves onto c ...
        assert dlg.cursor == 2 and dlg._card_2.property("highlighted") is True
        self._press(dlg, "Key_Left")                    # ...and straight back onto excluded b
        assert dlg.cursor == 1 and dlg._card_1.property("highlighted") is True
        assert dlg.state[b] == "excluded"               # landing never restores

    def test_arrows_do_not_set_decision(self, qapp, group_data):
        dlg = self._dialog(group_data)
        for key in ("Key_Left", "Key_Right", "Key_Up", "Key_Down"):
            self._press(dlg, key)
        assert dlg.decision is None                     # navigation never approves/noops

class TestSpaceAndCursorKeys:
    """Space applies right-click semantics at the cursor; Enter keeps the keeper
    there (and approves); mouse clicks park the cursor."""

    @staticmethod
    def _press(dlg, key_name):
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt, QEvent
        ev = QKeyEvent(QEvent.Type.KeyPress, getattr(Qt.Key, key_name),
                       Qt.KeyboardModifier.NoModifier)
        dlg.keyPressEvent(ev)

    @staticmethod
    def _dialog(data):
        return GroupApprovalDialog("t", data["paths"], data["winners"],
                                   data["losers"], data["scores"])

    def test_cursor_starts_on_winner(self, qapp, group_data):
        dlg = self._dialog(group_data)
        assert dlg.cursor == 0
        assert dlg._card_0.property("highlighted") is True

    def test_space_excludes_card_under_cursor(self, qapp, group_data):
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        self._press(dlg, "Key_Right")                   # cursor onto loser b
        self._press(dlg, "Key_Space")
        assert dlg.state[b] == "excluded"               # same as right-clicking it
        assert dlg._card_1.objectName() == "excludedCard"
        assert "Excluded" in dlg.status_label.text()
        w, l, e = dlg.final_partition()
        assert (w, e) == ([a], [b]) and len(l) == 1     # keeper untouched
        assert dlg.decision is None                     # Space never approves/noops

    def test_space_toggles_restoration_back(self, qapp, group_data):
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        self._press(dlg, "Key_Right")
        self._press(dlg, "Key_Space")                   # exclude b ...
        assert dlg.state[b] == "excluded"
        self._press(dlg, "Key_Space")                   # ...cursor parked on b: restore
        assert dlg.state[b] == "loser"
        assert dlg._card_1.objectName() == "loserCard"
        assert "Restored" in dlg.status_label.text()

    def test_space_on_the_keeper_drops_it_and_hands_the_role_over(self, qapp, group_data):
        """Space on the default-selected image works: it is excluded, the keeper role
        moves to the runner-up, and the box stays where it was used."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        self._press(dlg, "Key_Space")                   # cursor starts on the winner
        assert dlg.state[a] == "excluded"
        assert dlg.state[b] == "winner"                 # role handed to the runner-up
        assert dlg._card_0.objectName() == "excludedCard"
        assert dlg.cursor == 0                          # the box did not jump
        assert dlg._card_0.property("keeper") is True
        assert dlg.decision is None                     # Space never approves
        assert "keeper moved to b.jpg" in dlg.status_label.text()

    def test_space_on_the_keeper_leaves_enter_a_noop_until_the_box_moves(self, qapp, group_data):
        """Dropping the keeper never approves: Enter on the red card is a noop, and
        only moving the box onto the new keeper makes Enter confirm it."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        accepted = []
        dlg.accept = lambda: accepted.append(True)
        self._press(dlg, "Key_Space")                   # drop keeper a
        self._press(dlg, "Key_Return")                  # box still on the red card
        assert accepted == [] and dlg.decision is None
        assert dlg.state[a] == "excluded" and dlg.state[b] == "winner"
        self._press(dlg, "Key_Right")                   # box onto the inherited keeper
        self._press(dlg, "Key_Return")
        assert accepted == [True] and dlg.decision is Decision.APPROVED
        assert dlg.final_partition() == ([b], [c], [a])

    def test_restoring_the_dropped_keeper_returns_it_as_a_loser(self, qapp, group_data):
        """Space twice: the keeper comes back, but as a plain group member - the
        runner-up it handed the role to keeps it."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        self._press(dlg, "Key_Space")                   # drop keeper a -> b inherits
        self._press(dlg, "Key_Space")                   # box parked on a: restore it
        assert dlg.state[a] == "loser"
        assert dlg.state[b] == "winner"
        assert dlg._card_0.objectName() == "loserCard"
        assert "Restored" in dlg.status_label.text()
        assert dlg.final_partition() == ([b], [a, c], [])

    def test_enter_selects_keeper_at_cursor_and_approves(self, qapp, group_data):
        """Enter is the keeper key: it selects the card under the cursor, approves
        the group and closes the dialog, quarantining everything not kept."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        accepted = []
        dlg.accept = lambda: accepted.append(True)
        self._press(dlg, "Key_Right")                   # cursor onto loser b
        self._press(dlg, "Key_Return")
        assert dlg.state[b] == "winner"                 # same as left-clicking it
        assert dlg.state[a] == "loser"                  # old winner demoted
        w, l, e = dlg.final_partition()
        assert (w, e) == ([b], []) and len(l) == 2
        assert dlg._card_1.objectName() == "winnerCard"
        assert dlg.decision is Decision.APPROVED
        assert accepted == [True]                       # Enter closes the dialog
        assert "Approved" in dlg.status_label.text()

    def test_enter_on_the_current_keeper_approves_unchanged(self, qapp, group_data):
        """The fast path: the cursor starts on the scorer's winner, so a single
        Enter keeps it and quarantines the losers exactly as scored."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        dlg.accept = lambda: None
        before = dict(dlg.state)
        self._press(dlg, "Key_Return")
        assert dlg.state == before                      # nothing re-selected
        assert dlg.decision is Decision.APPROVED
        w, l, e = dlg.final_partition()
        assert w == [a] and sorted(l) == sorted([b, c]) and e == []

    def test_enter_on_excluded_image_is_a_noop(self, qapp, group_data):
        """The red box stays put: Enter never un-excludes, never approves, never closes."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        accepted = []
        dlg.accept = lambda: accepted.append(True)
        self._press(dlg, "Key_Right")                   # cursor onto loser b
        self._press(dlg, "Key_Space")                   # exclude it -> red box
        state_before, cursor_before = dict(dlg.state), dlg.cursor
        status_before = dlg.status_label.text()
        self._press(dlg, "Key_Return")
        assert dlg.state == state_before                # b is still excluded
        assert dlg.cursor == cursor_before              # the box did not move
        assert dlg.decision is None                     # ...nothing was approved...
        assert accepted == []                           # ...and the dialog stayed open
        assert dlg.status_label.text() != status_before
        assert "excluded" in dlg.status_label.text()
        assert "Space" in dlg.status_label.text()

    def test_restore_with_space_then_enter_keeps_the_image(self, qapp, group_data):
        """Space is the only way out of an exclusion; Enter then works as usual."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        dlg.accept = lambda: None
        self._press(dlg, "Key_Right")
        self._press(dlg, "Key_Space")                   # exclude b ...
        self._press(dlg, "Key_Return")                  # ...Enter is a noop
        assert dlg.state[b] == "excluded"
        self._press(dlg, "Key_Space")                   # restore b to loser
        assert dlg.state[b] == "loser"
        self._press(dlg, "Key_Return")                  # now Enter keeps b
        assert dlg.state[b] == "winner"
        assert dlg.decision is Decision.APPROVED

    def test_s_key_is_inert(self, qapp, group_data):
        """The keeper key is Enter now; S must neither select nor approve."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        dlg.accept = lambda: None
        before = dict(dlg.state)
        self._press(dlg, "Key_Right")
        self._press(dlg, "Key_S")
        assert dlg.state == before
        assert dlg.decision is None
        assert "Selected" not in dlg.status_label.text()

    def test_enter_with_no_images_is_a_noop(self, qapp):
        """No candidates -> Enter cannot select anything, so it must not approve."""
        dlg = GroupApprovalDialog("t", [], [], [], {})
        dlg.accept = lambda: None
        dlg.reject = lambda: None
        self._press(dlg, "Key_Return")
        assert dlg.decision is None

    def test_left_click_parks_cursor(self, qapp, group_data):
        dlg = self._dialog(group_data)
        _click(dlg._card_2, "Left")         # select c as keeper
        assert dlg.cursor == 2
        assert dlg._card_2.property("highlighted") is True
        assert dlg._card_0.property("highlighted") is False

    def test_right_click_parks_cursor(self, qapp, group_data):
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        _click(dlg._card_1, "Right")        # exclude b
        assert dlg.cursor == 1
        assert dlg._card_1.property("highlighted") is True


class TestKeeperBox:
    """The arrows move the *keeper box*: the whole winner design (green border and
    green tint) rides on the cursor, so the look is never pinned to one card. The
    confirmed keeper keeps only its star and its note line, and the box burns
    brighter red when it lands on an excluded card. No third colour exists."""

    @staticmethod
    def _rule_body(css, selector):
        """Declarations of the block whose selector list contains `selector`."""
        import re
        pattern = r'[^{}]*' + re.escape(selector) + r'[^{}]*\{([^}]*)\}'
        match = re.search(pattern, css)
        assert match, f"no CSS rule for {selector}"
        return match.group(1)

    @staticmethod
    def _dialog(data):
        return GroupApprovalDialog("t", data["paths"], data["winners"],
                                   data["losers"], data["scores"])

    @staticmethod
    def _press(dlg, key_name):
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt, QEvent
        dlg.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, getattr(Qt.Key, key_name),
                                    Qt.KeyboardModifier.NoModifier))

    # ------------------------------------------------------------- CSS contract

    def test_keeper_box_carries_the_whole_winner_design(self):
        """The box is not an outline sitting on top of a state: it *is* the winner
        look - green border AND green tint."""
        from deh.gui import DARK_THEME_CSS
        from deh.gui.theme import WINNER_BG, WINNER_BORDER
        body = self._rule_body(DARK_THEME_CSS, '#loserCard[keeper="true"]')
        assert WINNER_BORDER in body and WINNER_BG in body

    def test_winner_card_carries_no_colour_of_its_own(self):
        """The green design moved off the winner: a confirmed keeper is marked by its
        star and its note line only, so the box is free to wear the green elsewhere."""
        import re
        from deh.gui import DARK_THEME_CSS
        body = self._rule_body(DARK_THEME_CSS, '#winnerCard')
        assert "border: none" in body
        assert re.findall(r"#[0-9a-fA-F]{6}", body) == []

    def test_box_on_an_excluded_card_burns_brighter_red(self):
        from deh.gui import DARK_THEME_CSS
        from deh.gui.theme import KEEPER_HOT_BORDER
        body = self._rule_body(DARK_THEME_CSS, '#excludedCard[keeper="true"]')
        assert KEEPER_HOT_BORDER in body

    def test_hot_red_is_brighter_than_the_excluded_border(self):
        """'Brighter' is pinned numerically, not by eye."""
        from deh.gui.theme import EXCLUDED_BORDER, KEEPER_HOT_BORDER

        def brightness(colour):
            return sum(int(colour[i:i + 2], 16) for i in (1, 3, 5))

        assert brightness(KEEPER_HOT_BORDER) > brightness(EXCLUDED_BORDER)

    def test_excluded_card_keeps_its_own_red_under_the_box(self):
        """The box only brightens its border: the card's red tint and its own border
        stay exactly as the excluded state draws them."""
        from deh.gui import DARK_THEME_CSS
        from deh.gui.theme import EXCLUDED_BG, EXCLUDED_BORDER
        body = self._rule_body(DARK_THEME_CSS, '#excludedCard')
        assert EXCLUDED_BORDER in body and EXCLUDED_BG in body

    def test_keeper_rules_introduce_no_new_colour(self):
        """Every keeper-box colour is a palette constant, so the box can never drift
        from the colours the card states use."""
        import re
        from deh.gui import DARK_THEME_CSS
        from deh.gui.theme import KEEPER_HOT_BORDER, WINNER_BG, WINNER_BORDER
        palette = {WINNER_BORDER, WINNER_BG, KEEPER_HOT_BORDER}
        for selector in ('#loserCard[keeper="true"]', '#winnerCard[keeper="true"]',
                         '#excludedCard[keeper="true"]'):
            colours = set(re.findall(r"#[0-9a-fA-F]{6}",
                                     self._rule_body(DARK_THEME_CSS, selector)))
            assert colours <= palette, selector

    # ----------------------------------------------------------- box behaviour

    def test_box_opens_on_the_scorers_winner(self, qapp, group_data):
        """The GUI starts with the keeper already in place on the scored winner, so a
        bare Enter confirms the scored partition."""
        dlg = self._dialog(group_data)
        assert dlg.cursor == 0
        assert dlg._card_0.property("keeper") is True
        assert dlg._card_1.property("keeper") is False
        assert dlg._card_2.property("keeper") is False

    def test_arrows_move_the_box_and_the_star_stays_behind(self, qapp, group_data):
        """Moving the box moves the keeper look, not the scored verdict: the star and
        the note line stay on the scored winner."""
        a, b, c = group_data["paths"]
        dlg = self._dialog(group_data)
        self._press(dlg, "Key_Right")
        assert dlg._card_1.property("keeper") is True
        assert dlg._card_0.property("keeper") is False
        assert "\u2605" in dlg._card_0.score_label.text()      # star stays put
        assert "\u2605" not in dlg._card_1.score_label.text()
        assert dlg.state == {a: "winner", b: "loser", c: "loser"}

    def test_box_on_an_excluded_card_is_the_hot_red_pair(self, qapp, group_data):
        """The state pairing the CSS keys off: excluded + keeper. Arrows can land on
        it, and landing never restores it."""
        b = group_data["paths"][1]
        dlg = self._dialog(group_data)
        self._press(dlg, "Key_Right")                      # box onto loser b
        self._press(dlg, "Key_Space")                      # exclude it
        card = dlg._card_1
        assert card.objectName() == "excludedCard"
        assert card.property("keeper") is True             # hot box sits on it
        self._press(dlg, "Key_Right")                      # step away ...
        assert dlg._card_1.property("keeper") is False
        self._press(dlg, "Key_Left")                       # ...and back onto the excluded card
        assert card.property("keeper") is True
        assert card.objectName() == "excludedCard"         # landing never restores
        assert dlg.state[b] == "excluded"

    def test_left_click_moves_the_box_onto_the_clicked_card(self, qapp, group_data):
        """Mouse and keyboard agree: clicking a candidate moves the whole winner
        design onto it, star included."""
        dlg = self._dialog(group_data)
        dlg._on_card_selected(group_data["paths"][2])
        assert dlg._card_2.property("keeper") is True
        assert dlg._card_0.property("keeper") is False
        assert "\u2605" in dlg._card_2.score_label.text()
        assert "\u2605" not in dlg._card_0.score_label.text()


class TestShowGroupApprovalLayout:
    """Placement/scale wiring must not break the decision flow."""

    def test_placement_is_always_applied(self, qapp, group_data, monkeypatch):
        """Every invocation enlarges and centers the dialog — no env opt-in."""
        import deh.gui as g
        import deh.gui.duplicate as d
        calls = []
        monkeypatch.delenv("DEH_GUI_SCALE", raising=False)
        monkeypatch.setattr(d, "enlarge_to_screen", lambda dlg: calls.append("enlarge"))
        monkeypatch.setattr(d, "center_on_screen", lambda dlg: calls.append("center"))
        monkeypatch.setattr(g.GroupApprovalDialog, "exec",
                            lambda self: setattr(self, "decision", Decision.APPROVED))
        result = g.show_group_approval(**group_data, group_index=0)
        assert result[0] == Decision.APPROVED
        assert calls == ["enlarge", "center"]   # ordering matters: size, then position

    def test_retired_position_env_is_ignored(self, qapp, group_data, monkeypatch):
        """The removed GUI_POSITION knob no longer changes placement."""
        import deh.gui as g
        import deh.gui.duplicate as d
        calls = []
        monkeypatch.setenv("GUI_POSITION", "default")
        monkeypatch.setattr(d, "center_on_screen", lambda dlg: calls.append("center"))
        monkeypatch.setattr(g.GroupApprovalDialog, "exec",
                            lambda self: setattr(self, "decision", Decision.APPROVED))
        g.show_group_approval(**group_data, group_index=0)
        assert calls == ["center"]

    def test_dialog_honors_scale_parameter(self, qapp, group_data):
        d1 = GroupApprovalDialog("t", group_data["paths"], group_data["winners"],
                                 group_data["losers"], group_data["scores"], scale=1.0)
        d2 = GroupApprovalDialog("t", group_data["paths"], group_data["winners"],
                                 group_data["losers"], group_data["scores"], scale=1.5)
        assert d2.scale == 1.5
        # Scale still enlarges the UI (card image footprint grows), independent of top-level min.
        c1 = d1._card_0.image_label
        c2 = d2._card_0.image_label
        assert c2.minimumWidth() > c1.minimumWidth()
        assert c2.minimumHeight() > c1.minimumHeight()

    def test_window_has_min_max_buttons(self, qapp, group_data):
        from deh.gui import _resolve_flag
        dlg = GroupApprovalDialog("t", group_data["paths"], group_data["winners"],
                                  group_data["losers"], group_data["scores"], scale=1.0)
        flags = int(dlg.windowFlags())
        mx = _resolve_flag("WindowMaximizeButtonHint")
        mn = _resolve_flag("WindowMinimizeButtonHint")
        cl = _resolve_flag("WindowCloseButtonHint")
        assert mx and bool(flags & mx), "maximize button hint missing"
        assert mn and bool(flags & mn), "minimize button hint missing"
        if cl:
            assert bool(flags & cl), "close button hint missing"


class TestDecisionNoteDisplay:
    """Cards can carry a short scorer-reason line explaining WHY an image was kept or
    quarantined despite its score (crop rule / near-tie resolution)."""

    NOTE = ("Cropped/smaller variant of other.png (~44% smaller framing)"
            " -- full-frame image preferred")

    def test_card_renders_note_when_provided(self, qapp, tmp_path):
        p = _make_img(tmp_path / "n.jpg")
        card = ImageCard(p, is_winner=True, note=self.NOTE)
        assert card.note_text == self.NOTE
        assert getattr(card, "note_label", None) is not None
        assert card.note_label.objectName() == "cardNote"
        assert card.note_label.text() == self.NOTE

    def test_card_without_note_has_no_extra_line(self, qapp, tmp_path):
        p = _make_img(tmp_path / "nn.jpg")
        card = ImageCard(p, is_winner=False, score=0.5)
        assert card.note_text is None
        assert not hasattr(card, "note_label")

    def test_theme_styles_notes(self, qapp):
        from deh.gui import DARK_THEME_CSS
        assert "#cardnote" in DARK_THEME_CSS.lower()

    def test_dialog_forwards_notes_to_cards(self, qapp, group_data):
        notes = {group_data["paths"][1]: self.NOTE}
        dlg = GroupApprovalDialog("t", group_data["paths"], group_data["winners"],
                                  group_data["losers"], group_data["scores"], scale=1.0,
                                  notes=notes)
        assert dlg._card_1.note_text == self.NOTE
        for i in (0, 2):                       # unnoted cards stay clean
            assert getattr(dlg, f"_card_{i}").note_text is None

    def test_show_group_approval_forwards_notes(self, qapp, group_data, monkeypatch):
        instances = []
        orig_init = GroupApprovalDialog.__init__

        def spy_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            instances.append(self)

        monkeypatch.setattr(GroupApprovalDialog, "__init__", spy_init)
        monkeypatch.setattr(GroupApprovalDialog, "exec",
                            lambda self: setattr(self, "decision", Decision.APPROVED))
        path = str(group_data["paths"][1])
        show_group_approval(**{**group_data, "group_index": 0}, notes={path: self.NOTE})
        dlg = instances[-1]
        card = next(getattr(dlg, f"_card_{i}") for i in range(len(group_data["paths"]))
                    if getattr(dlg, f"_card_{i}")._path_key == path)
        assert card.note_text == self.NOTE






class TestGuiUnavailableFallback:
    """A failed review dialog must not authorize file operations."""

    def test_raises_instead_of_approving(self, qapp, group_data, monkeypatch):
        import deh.gui.duplicate as d

        def no_display():
            raise RuntimeError("could not find the Qt platform plugin 'xcb'")

        monkeypatch.setattr(d, "ensure_app", no_display)
        with pytest.raises(RuntimeError, match="--no-gui"):
            d.show_group_approval(**group_data, group_index=0)
