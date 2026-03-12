from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRect, QSize, Qt
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
    tldr: str = "n/a"


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

        style.drawPrimitive(
            QStyle.PrimitiveElement.PE_PanelItemViewItem, opt, painter, opt.widget
        )

        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        text_rect = opt.rect.adjusted(8, 6, -8, -6)
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
        painter.setPen(opt.palette.mid().color())
        painter.drawLine(opt.rect.bottomLeft(), opt.rect.bottomRight())
        painter.restore()

    def sizeHint(self, option, index):
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        width = self._view.viewport().width()
        rect = option.fontMetrics.boundingRect(
            QRect(0, 0, max(width - 16, 1), 10000),
            Qt.TextFlag.TextWordWrap,
            text,
        )
        return QSize(width, rect.height() + 12)


class PaperListView(QListView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(WordWrapDelegate(self, parent=self))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setResizeMode(QListView.ResizeMode.Adjust)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scheduleDelayedItemsLayout()
