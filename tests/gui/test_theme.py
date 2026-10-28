"""Tests for deh.gui.theme: the palette constants are the source of the CSS.

The stylesheet used to repeat every palette colour as a raw hex literal, so the
constants and the CSS could drift apart and nothing would fail - the dialogs
assert on CSS substrings, ``ImageCard`` applies the CSS verbatim, and the
constants were referenced by nothing. The CSS is now generated from the
constants; this file pins the exact generated string (a palette change must be
a deliberate edit here) and refuses new colours that enter the stylesheet as
untracked literals.
"""

import re

from deh.gui import DARK_THEME_CSS as FACADE_CSS
from deh.gui.theme import (BG, DARK_THEME_CSS, EXCLUDED_BG, EXCLUDED_BORDER,
                           FG, KEEPER_HOT_BORDER, LOSER_SCORE, NOTE, WINNER_BG,
                           WINNER_BORDER, WINNER_SCORE)

#: The stylesheet, byte for byte. Reviewing a palette change means reviewing
#: this string.
EXPECTED_CSS = '''
QWidget { background-color: #2b2b2b; color: #f0f0f0; }
QLabel  { background-color: transparent; border: none; padding: 0px; }
QFrame  { border: none; }
#winnerCard   { border: none; }
#scoreWinner  { color: #35e07d; font-weight: bold; }
#scoreLoser   { color: #cfcfcf; }
#excludedCard { border: 3px solid #e0454a; background-color: #3a2b2c; }
#cardNote     { color: #d9a441; font-size: 9pt; padding-top: 2px; }
QAbstractScrollArea, QScrollArea { border: none; background-color: #2b2b2b; }
QScrollBar:vertical { background: #2b2b2b; width: 12px; margin: 0px; }
QScrollBar::handle:vertical { background: #5a5a5a; min-height: 28px; border-radius: 6px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
/* Keyboard cursor on top of a card state (arrow-key navigation):
   a border-only marker, so it never claims a state the card does not have. */
#winnerCard[highlighted="true"], #loserCard[highlighted="true"] {
    border: 3px solid #17c964;
}
#excludedCard[highlighted="true"] {
    border: 3px solid #e0454a;
}
/* Duplicate review: the keeper box IS the winner design - green border AND green
   tint - and the arrows move it. The confirmed keeper keeps only its star and its
   note line, so the look is never pinned to one card; Enter confirms the box. */
#winnerCard[keeper="true"], #loserCard[keeper="true"] {
    border: 3px solid #17c964; background-color: #2e3a31;
}
/* Landing on an excluded card burns the box's border brighter red; the card's own
   red tint and border stay exactly as they are. */
#excludedCard[keeper="true"] {
    border: 3px solid #ff4d4d;
}
'''

#: Colours that live only in the stylesheet: the scrollbar chrome, which nothing
#: else uses, so it has no palette constant to drift from.
#: The keeper box is NOT one of them - it wears the palette's own kept-green and
#: burns to the palette's own hot red - so it can never drift from the card states.
CHROME_COLORS = {"#5a5a5a"}


PALETTE = [BG, FG, WINNER_BORDER, WINNER_BG, WINNER_SCORE, LOSER_SCORE,
           EXCLUDED_BORDER, EXCLUDED_BG, KEEPER_HOT_BORDER, NOTE]


class TestGeneratedStylesheet:

    def test_css_is_the_pinned_string(self):
        assert DARK_THEME_CSS == EXPECTED_CSS

    def test_gui_facade_exports_the_same_stylesheet(self):
        assert FACADE_CSS == DARK_THEME_CSS

    def test_every_palette_colour_reaches_the_css(self):
        for colour in PALETTE:
            assert colour in DARK_THEME_CSS

    def test_no_untracked_colour_in_the_css(self):
        found = set(re.findall(r"#[0-9a-fA-F]{6}", DARK_THEME_CSS))
        assert found - set(PALETTE) - CHROME_COLORS == set()

    def test_palette_constants_are_plain_hex(self):
        for colour in PALETTE:
            assert re.fullmatch(r"#[0-9a-f]{6}", colour), colour
