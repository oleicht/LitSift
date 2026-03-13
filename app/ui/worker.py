from PySide6.QtCore import QThread, Signal


class RankingWorker(QThread):
    results_ready = Signal(list)
    error = Signal(str)

    def __init__(self, query: str, parent=None):
        super().__init__(parent)
        self._query = query

    def run(self):
        try:
            from app.backend import get_rankings
            from app.ui.paper import Paper
            self.results_ready.emit([
                Paper(title, abstract, tldr, authors, name, score, id)
                for title, abstract, tldr, authors, name, score, id in get_rankings(self._query)
            ])
        except Exception as e:
            self.error.emit(str(e))
