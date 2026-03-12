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
            from app.ui.paper_list import Paper
            self.results_ready.emit([
                Paper(title, abstract, tldr)
                for title, abstract, tldr in get_rankings(self._query)
            ])
        except Exception as e:
            self.error.emit(str(e))
