"""Shared UI toolkit for the desktop client — the workstation look and reusable building blocks.

This is a Trust Analysis workstation, not an OCR viewer: a calm, dense, dark-neutral surface with
stat tiles, titled cards, and scannable tables, so every screen answers "what is happening / what
needs my attention / why" at a glance rather than showing developer-grade empty white space. All Qt
imports live in the `clients.desktop` package only.

Nothing here holds behavior — these are presentation primitives the Center views compose. Empty
states are first-class (`empty_state`): when data genuinely does not exist yet, the operator is told
so honestly, never shown a blank canvas that looks like a failure.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# A restrained dark-neutral workstation palette. Accents carry meaning (green=agreement/health,
# amber=attention, red=conflict/failure) and are always paired with text, never colour alone.
STYLESHEET = """
QWidget { background: #12151b; color: #e6e9ef; font-family: "Segoe UI", Arial, sans-serif; font-size: 13px; }
QLabel#PageTitle { font-size: 22px; font-weight: 600; color: #f2f4f8; }
QLabel#PageSubtitle { color: #9aa4b2; font-size: 13px; }
QLabel#SectionTitle { font-size: 15px; font-weight: 600; color: #cdd4df; padding-top: 4px; }
QLabel#Muted { color: #8b95a5; }
QLabel#StatValue { font-size: 26px; font-weight: 700; color: #f2f4f8; }
QLabel#StatLabel { color: #9aa4b2; font-size: 12px; }
QFrame#Card { background: #1a1f28; border: 1px solid #262d3a; border-radius: 10px; }
QFrame#Stat { background: #1a1f28; border: 1px solid #262d3a; border-radius: 10px; }
QFrame#Stat[tone="attention"] { border: 1px solid #7a5a1e; }
QFrame#Stat[tone="danger"] { border: 1px solid #7a2e2e; }
QListWidget#Nav { background: #0d1015; border: none; border-right: 1px solid #262d3a; font-size: 14px; padding-top: 8px; }
QListWidget#Nav::item { padding: 11px 16px; color: #aeb7c4; border-radius: 6px; margin: 2px 8px; }
QListWidget#Nav::item:selected { background: #1f2632; color: #ffffff; }
QTableWidget { background: #151a22; border: 1px solid #262d3a; border-radius: 8px; gridline-color: #232a36; }
QHeaderView::section { background: #1a1f28; color: #9aa4b2; border: none; border-bottom: 1px solid #2b3340; padding: 7px; font-weight: 600; }
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected { background: #24405c; color: #ffffff; }
QPushButton { background: #232b38; color: #e6e9ef; border: 1px solid #313b4b; border-radius: 7px; padding: 8px 14px; }
QPushButton:hover { background: #2b3546; }
QPushButton:disabled { background: #171b22; color: #5b6472; border-color: #232a36; }
QPushButton#Primary { background: #2f6f4f; border-color: #3a8a62; color: #ffffff; font-weight: 600; }
QPushButton#Primary:hover { background: #37805c; }
QPushButton#Primary:disabled { background: #1d2a23; color: #5b6472; border-color: #26362c; }
QLineEdit { background: #10141b; border: 1px solid #313b4b; border-radius: 7px; padding: 8px; color: #e6e9ef; }
QScrollArea { border: none; }
"""

TONE_NEUTRAL = "neutral"
TONE_ATTENTION = "attention"
TONE_DANGER = "danger"


def page_header(title: str, subtitle: str) -> QWidget:
    """A page's title + one-line "what this screen is for" subtitle."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 8)
    t = QLabel(title)
    t.setObjectName("PageTitle")
    s = QLabel(subtitle)
    s.setObjectName("PageSubtitle")
    s.setWordWrap(True)
    layout.addWidget(t)
    layout.addWidget(s)
    return box


def section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(True)
    return label


def stat_tile(value: str, label: str, tone: str = TONE_NEUTRAL) -> QFrame:
    """A KPI tile: a large number over a small label, tone-bordered when it wants attention."""
    frame = QFrame()
    frame.setObjectName("Stat")
    frame.setProperty("tone", tone)
    frame.setMinimumWidth(150)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    v = QLabel(value)
    v.setObjectName("StatValue")
    lbl = QLabel(label)
    lbl.setObjectName("StatLabel")
    lbl.setWordWrap(True)
    layout.addWidget(v)
    layout.addWidget(lbl)
    return frame


def stat_row(tiles: list[QFrame]) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    for tile in tiles:
        layout.addWidget(tile)
    layout.addStretch(1)
    return box


def card(title: str, body: QWidget) -> QFrame:
    """A titled container for one section of content."""
    frame = QFrame()
    frame.setObjectName("Card")
    frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.addWidget(section_title(title))
    layout.addWidget(body)
    return frame


def table(headers: list[str], rows: list[list[str]]) -> QWidget:
    """A read-only, scannable table. Falls back to an honest empty state when there are no rows."""
    if not rows:
        return empty_state("Nothing to show yet", "No data of this kind has been recorded so far.")
    widget = QTableWidget(len(rows), len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().setVisible(False)
    widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    widget.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    widget.horizontalHeader().setStretchLastSection(True)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            item = QTableWidgetItem(value)
            if c > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            widget.setItem(r, c, item)
    widget.resizeColumnsToContents()
    widget.horizontalHeader().setStretchLastSection(True)
    return widget


def scroll_page(sections: list[QWidget]) -> QWidget:
    """Stack `sections` vertically inside a vertical scroll area, so a dense Center never clips and
    the page body never scrolls horizontally."""
    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.setContentsMargins(24, 20, 24, 24)
    layout.setSpacing(16)
    for section in sections:
        layout.addWidget(section)
    layout.addStretch(1)

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(inner)
    return area


def empty_state(title: str, detail: str) -> QWidget:
    """Shown when data genuinely does not exist yet — never a blank area (the operator must never
    wonder whether something failed)."""
    box = QFrame()
    box.setObjectName("Card")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(20, 24, 20, 24)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    t = QLabel(title)
    t.setObjectName("SectionTitle")
    t.setAlignment(Qt.AlignmentFlag.AlignCenter)
    d = muted(detail)
    d.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(t)
    layout.addWidget(d)
    return box
