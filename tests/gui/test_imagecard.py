"""Tests for the shared ImageCard widget (deh.gui.imagecard)."""
from PIL import Image

from deh.gui import ImageCard, GroupApprovalDialog


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


class TestImageCardPresentation:
    """Card rendering: metadata line, single winner border, and scale-driven sizing."""

    def test_card_shows_resolution_and_filesize(self, qapp, tmp_path):
        p = _make_img(tmp_path / "meta.jpg", size=(320, 240))
        card = ImageCard(p, is_winner=False, score=0.6, scale=1.0)
        assert card.resolution_text.startswith("320")
        assert "px" in card.resolution_text
        assert any(u in card.filesize_text for u in ("B", "KB", "MB"))

    def test_scale_enlarges_card_footprint(self, qapp, tmp_path):
        p = _make_img(tmp_path / "scale.jpg")
        base = ImageCard(p, is_winner=False, scale=1.0)
        big = ImageCard(p, is_winner=False, scale=1.5)
        # The image area's minimum footprint grows with GUI_SCALE.
        assert big.image_label.minimumWidth() > base.image_label.minimumWidth()
        assert big.image_label.minimumHeight() > base.image_label.minimumHeight()

    def test_single_frame_only_on_winner(self, qapp, tmp_path):
        from deh.gui import DARK_THEME_CSS
        p = _make_img(tmp_path / "w.jpg")
        winner = ImageCard(p, is_winner=True, scale=1.0)
        loser = ImageCard(p, is_winner=False, scale=1.0)
        # Winner/loser differ only by objectName, and the winner carries no outline of
        # its own - the green design belongs to the movable keeper box (see below).
        assert winner.objectName() == "winnerCard"
        assert loser.objectName() == "loserCard"
        css = DARK_THEME_CSS.lower()
        assert "#winnercard { border: none; }" in " ".join(css.split())   # the look moved to the box
        assert "background-color: #2b2b2b" in css       # forced dark background
        assert "color: #f0f0f0" in css                  # white text
        assert winner.styleSheet() == DARK_THEME_CSS   # theme applied to the card

    def test_responsive_fit_grows_with_available_space(self, qapp, tmp_path):
        from PyQt6.QtGui import QPixmap
        p = _make_img(tmp_path / "r.jpg")
        c = ImageCard(p, is_winner=False, scale=1.0)
        c.set_image_pixmap(QPixmap(800, 600))
        c.image_label.resize(250, 187); c._fit_image()
        w_small = c.image_label.pixmap().width()
        c.image_label.resize(600, 450); c._fit_image()
        w_big = c.image_label.pixmap().width()
        assert w_big > w_small   # image re-scales to fill larger area instead of staying tiny

    def test_relative_subpath_display_name(self, qapp, tmp_path):
        a = tmp_path / "samples" / "a.jpg"; b = tmp_path / "samples" / "somedir" / "b.jpg"
        for f in (a, b):
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 240), (9, 87, 200)).save(str(f))
        dlg = GroupApprovalDialog("t", [str(a), str(b)], [str(a)], [str(b)],
                                  {str(a): 0.9, str(b): 0.5})
        assert dlg._card_0.display_text == "a.jpg"
        assert dlg._card_1.display_text == "somedir/b.jpg"

    def test_cursor_highlight_is_a_separate_marker(self, qapp, tmp_path):
        """The cursor is a property on top of the card's own state, not a state."""
        from deh.gui import DARK_THEME_CSS
        p = _make_img(tmp_path / "state.jpg")
        card = ImageCard(p, is_winner=False, scale=1.0)
        css = DARK_THEME_CSS.lower()
        assert 'highlighted="true"' in css        # cursor rule exists
        card.set_highlighted(True)
        assert card.property("highlighted") is True
        card.set_highlighted(False)
        assert card.property("highlighted") is False
        assert card.objectName() == "loserCard"   # the cursor changed no state
        # the review vocabulary
        card.set_state("winner")
        assert card.objectName() == "winnerCard"
        card.set_state("excluded")
        assert card.objectName() == "excludedCard"
        card.set_state("loser")
        assert card.objectName() == "loserCard"


    def test_keeper_box_is_a_property_not_a_state(self, qapp, tmp_path):
        """The keeper box (duplicate review) carries the whole winner design and the
        arrows move it; it is a property, so it never touches the card's state."""
        from deh.gui import DARK_THEME_CSS
        p = _make_img(tmp_path / "keeper.jpg")
        card = ImageCard(p, is_winner=False, scale=1.0)
        css = DARK_THEME_CSS.lower()
        assert '#losercard[keeper="true"]' in css     # the movable winner design
        assert '#excludedcard[keeper="true"]' in css  # ...burning brighter red on a red card
        card.set_keeper(True)
        assert card.property("keeper") is True
        card.set_keeper(False)
        assert card.property("keeper") is False
        assert card.objectName() == "loserCard"       # moving the box changes no state
