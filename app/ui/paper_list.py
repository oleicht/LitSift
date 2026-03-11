from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QListView


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
        if not index.isValid() or not (0 <= index.row() < len(self._papers)):
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


class PaperListView(QListView):
    paper_activated = Signal(Paper)

    _COLOR_BASE = QColor("#f0f0f0")
    _COLOR_ALT = QColor("#e0e0e0")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, self._COLOR_BASE)
        palette.setColor(QPalette.ColorRole.AlternateBase, self._COLOR_ALT)
        self.setPalette(palette)
        self.doubleClicked.connect(self._emit_paper)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._emit_paper(self.currentIndex())
        else:
            super().keyPressEvent(event)

    def _emit_paper(self, index: QModelIndex) -> None:
        paper = self.model().data(index, Qt.ItemDataRole.UserRole)
        if paper is not None:
            self.paper_activated.emit(paper)
