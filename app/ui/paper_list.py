from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QListView,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
)


@dataclass(frozen=True)
class Paper:
    title: str
    abstract: str


class PaperListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._papers: list[Paper] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._papers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        paper = self._papers[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return paper.title
        if role == Qt.ItemDataRole.UserRole:
            return paper
        return None

    def populate(self, papers: list[Paper]) -> None:
        self.beginResetModel()
        self._papers = papers
        self.endResetModel()


class WordWrapDelegate(QStyledItemDelegate):
    def __init__(self, view: "PaperListView", parent=None):
        super().__init__(parent)
        self._view = view

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        style = opt.widget.style() if opt.widget else QApplication.style()
        painter.save()

        # Draw background (handles selection, alternating rows, hover, etc.)
        style.drawPrimitive(
            QStyle.PrimitiveElement.PE_PanelItemViewItem, opt, painter, opt.widget
        )

        # Draw text with word wrap
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        text_rect = opt.rect.adjusted(4, 2, -4, -2)
        color = (
            opt.palette.highlightedText().color()
            if opt.state & QStyle.StateFlag.State_Selected
            else opt.palette.text().color()
        )
        painter.setPen(color)
        painter.drawText(
            text_rect,
            Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            text,
        )
        painter.restore()

    def sizeHint(self, option, index):
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        width = self._view.viewport().width()
        rect = option.fontMetrics.boundingRect(
            QRect(0, 0, max(width - 8, 1), 10000),
            Qt.TextFlag.TextWordWrap,
            text,
        )
        return QSize(width, rect.height() + 8)


class PaperListView(QListView):
    _COLOR_BASE = QColor("#f0f0f0")
    _COLOR_ALT = QColor("#e0e0e0")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, self._COLOR_BASE)
        palette.setColor(QPalette.ColorRole.AlternateBase, self._COLOR_ALT)
        self.setPalette(palette)
        self.setItemDelegate(WordWrapDelegate(self, parent=self))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setResizeMode(QListView.ResizeMode.Adjust)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scheduleDelayedItemsLayout()
