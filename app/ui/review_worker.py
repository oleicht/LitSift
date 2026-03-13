from PySide6.QtCore import QThread, Signal

from app.ui.paper import Paper


class ReviewWorker(QThread):
    results_ready = Signal(list)
    error = Signal(str)

    def __init__(self, paper: Paper, parent=None):
        super().__init__(parent)
        self._paper = paper

    def run(self):
        try:
            from app.backend import get_reviews
            self.results_ready.emit(get_reviews(self._paper.id))
        except Exception as e:
            self.error.emit(str(e))
