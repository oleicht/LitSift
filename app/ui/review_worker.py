from PySide6.QtCore import QThread, Signal

from app.ui.paper_list import Paper


class ReviewWorker(QThread):
    results_ready = Signal(list)
    error = Signal(str)

    def __init__(self, paper: Paper, parent=None):
        super().__init__(parent)
        self._paper = paper

    def run(self):
        try:
            from app.ranking import get_data, get_reviews
            data = get_data()
            match = data[data["title"] == self._paper.title]
            if match.empty:
                self.error.emit("Paper not found in database.")
                return
            self.results_ready.emit(get_reviews(match["id"].iloc[0]))
        except Exception as e:
            self.error.emit(str(e))
