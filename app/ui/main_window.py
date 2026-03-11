from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.paper_detail import PaperDetailDialog
from app.ui.paper_list import Paper, PaperListModel, PaperListView
from app.ui.worker import RankingWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LitSift")
        self.resize(720, 800)
        self._worker: RankingWorker | None = None
        self._model = PaperListModel()
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("Enter your query:"))

        query_row = QHBoxLayout()
        self._query_input = QLineEdit()
        self._query_input.setPlaceholderText("e.g. diffusion models for protein design")
        self._query_input.returnPressed.connect(self._on_submit)
        query_row.addWidget(self._query_input)

        self._submit_btn = QPushButton("Submit")
        self._submit_btn.clicked.connect(self._on_submit)
        query_row.addWidget(self._submit_btn)
        layout.addLayout(query_row)

        self._view = PaperListView()
        self._view.setModel(self._model)
        self._view.paper_activated.connect(self._on_paper_activated)
        layout.addWidget(self._view)

        self.statusBar().showMessage("Ready")

    def _on_submit(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        query = self._query_input.text().strip()
        if not query:
            return

        self._submit_btn.setEnabled(False)
        self.statusBar().showMessage("Ranking…")

        self._worker = RankingWorker(query, parent=self)
        self._worker.results_ready.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda: self._submit_btn.setEnabled(True))
        self._worker.start()

    def _on_results(self, results: list[tuple[str, str]]) -> None:
        papers = [Paper(title, abstract) for title, abstract in results]
        self._model.populate(papers)
        self.statusBar().showMessage(f"{len(papers)} papers ranked", 5000)

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Error: {msg}", 8000)

    def _on_paper_activated(self, paper: Paper) -> None:
        PaperDetailDialog(paper, parent=self).exec()

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        super().closeEvent(event)
