# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Ratan M. Kyeno
"""Dark-theme stylesheet + palette constants for the review dialogs.

Centralised so the duplicate-approval dialog (and any dialog built on the shared
card grid) shares one consistent look, and so the exact CSS rules can be
asserted in tests.
"""


# Palette (named constants so callers / tests can reference them).
BG = "#2b2b2b"
FG = "#f0f0f0"
WINNER_BORDER = "#17c964"
WINNER_BG = "#2e3a31"
WINNER_SCORE = "#35e07d"
LOSER_SCORE = "#cfcfcf"
EXCLUDED_BORDER = "#e0454a"
EXCLUDED_BG = "#3a2b2c"
KEEPER_HOT_BORDER = "#ff4d4d"   # the keeper box landing on an excluded card: brighter red
NOTE = "#d9a441"

#: The stylesheet, generated FROM the palette above. The constants are the
#: single source of truth: the CSS used to repeat every palette colour as a raw
#: literal, so a palette edit silently changed nothing (or changed the
#: constants the dialogs' tests assert on, leaving the two out of sync).
#: Scrollbar/button chrome keeps its literals - those colours exist nowhere
#: else, so there is nothing for them to drift from.
DARK_THEME_CSS = """
QWidget { background-color: %(bg)s; color: %(fg)s; }
QLabel  { background-color: transparent; border: none; padding: 0px; }
QFrame  { border: none; }
#winnerCard   { border: none; }
#scoreWinner  { color: %(win_score)s; font-weight: bold; }
#scoreLoser   { color: %(lose_score)s; }
#excludedCard { border: 3px solid %(exc_border)s; background-color: %(exc_bg)s; }
#cardNote     { color: %(note)s; font-size: 9pt; padding-top: 2px; }
QAbstractScrollArea, QScrollArea { border: none; background-color: %(bg)s; }
QScrollBar:vertical { background: %(bg)s; width: 12px; margin: 0px; }
QScrollBar::handle:vertical { background: #5a5a5a; min-height: 28px; border-radius: 6px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
/* Keyboard cursor on top of a card state (arrow-key navigation):
   a border-only marker, so it never claims a state the card does not have. */
#winnerCard[highlighted="true"], #loserCard[highlighted="true"] {
    border: 3px solid %(win_border)s;
}
#excludedCard[highlighted="true"] {
    border: 3px solid %(exc_border)s;
}
/* Duplicate review: the keeper box IS the winner design - green border AND green
   tint - and the arrows move it. The confirmed keeper keeps only its star and its
   note line, so the look is never pinned to one card; Enter confirms the box. */
#winnerCard[keeper="true"], #loserCard[keeper="true"] {
    border: 3px solid %(win_border)s; background-color: %(win_bg)s;
}
/* Landing on an excluded card burns the box's border brighter red; the card's own
   red tint and border stay exactly as they are. */
#excludedCard[keeper="true"] {
    border: 3px solid %(hot_border)s;
}
""" % {"bg": BG, "fg": FG,
       "win_border": WINNER_BORDER, "win_bg": WINNER_BG,
       "win_score": WINNER_SCORE, "lose_score": LOSER_SCORE,
       "exc_border": EXCLUDED_BORDER, "exc_bg": EXCLUDED_BG,
       "hot_border": KEEPER_HOT_BORDER,
       "note": NOTE}
