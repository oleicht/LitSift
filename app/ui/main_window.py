from PySide6.QtCore import Qt
from PySide6.QtGui import QTextBlockFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui.paper_list import Paper, PaperListModel, PaperListView
from app.ui.review_dialog import ReviewDialog
from app.ui.review_worker import ReviewWorker
from app.ui.worker import RankingWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LitSift")
        self.resize(1200, 700)
        self._worker: RankingWorker | None = None
        self._review_worker: ReviewWorker | None = None
        self._model = PaperListModel()
        self._current_paper: Paper | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self._view = PaperListView()
        self._view.setModel(self._model)
        self._view.selectionModel().currentChanged.connect(self._on_selection_changed)
        self._model.modelReset.connect(self._on_model_reset)
        splitter.addWidget(self._view)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        right_layout.addWidget(QLabel("Enter your query:"))
        query_row = QHBoxLayout()
        self._query_input = QLineEdit()
        self._query_input.setPlaceholderText("e.g. diffusion models for protein design")
        self._query_input.returnPressed.connect(self._on_submit)
        query_row.addWidget(self._query_input)
        self._submit_btn = QPushButton("Submit")
        self._submit_btn.clicked.connect(self._on_submit)
        query_row.addWidget(self._submit_btn)
        right_layout.addLayout(query_row)

        right_layout.addWidget(QLabel("Abstract:"))
        self._abstract_view = QTextEdit()
        self._abstract_view.setReadOnly(True)
        self._abstract_view.document().setDocumentMargin(10)
        right_layout.addWidget(self._abstract_view)

        self._download_btn = QPushButton("Download")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._on_download)
        right_layout.addWidget(self._download_btn)

        self._reviews_btn = QPushButton("Show Reviews")
        self._reviews_btn.setEnabled(False)
        self._reviews_btn.clicked.connect(self._on_reviews)
        right_layout.addWidget(self._reviews_btn)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        font = self.font()
        font.setPointSize(font.pointSize() + 1)
        self._view.setFont(font)
        self._abstract_view.setFont(font)

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

    def _on_selection_changed(self, current, _) -> None:
        paper = self._model.data(current, Qt.ItemDataRole.UserRole)
        self._current_paper = paper
        if paper is not None:
            self._abstract_view.setPlainText(paper.abstract)
            fmt = QTextBlockFormat()
            fmt.setLineHeight(140.0, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            cursor = self._abstract_view.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.mergeBlockFormat(fmt)
            self._download_btn.setEnabled(True)
            self._reviews_btn.setEnabled(True)
        else:
            self._abstract_view.clear()
            self._download_btn.setEnabled(False)
            self._reviews_btn.setEnabled(False)

    def _on_model_reset(self) -> None:
        self._current_paper = None
        self._abstract_view.clear()
        self._download_btn.setEnabled(False)
        self._reviews_btn.setEnabled(False)

    def _on_reviews(self) -> None:
        if self._review_worker and self._review_worker.isRunning():
            return
        self._reviews_btn.setEnabled(False)
        self.statusBar().showMessage("Loading reviews…")
        self._review_worker = ReviewWorker(self._current_paper, parent=self)
        self._review_worker.results_ready.connect(self._on_review_results)
        self._review_worker.error.connect(self._on_review_error)
        self._review_worker.finished.connect(
            lambda: self._reviews_btn.setEnabled(self._current_paper is not None)
        )
        self._review_worker.start()

    def _on_review_results(self, notes: list) -> None:
        self.statusBar().clearMessage()
        ReviewDialog(self._current_paper, notes, parent=self).exec()

    def _on_review_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Error loading reviews: {msg}", 8000)

    def _on_download(self) -> None:
        from app.ranking import download
        try:
            download(self._current_paper.title)
            QMessageBox.information(self, "Download", "Downloaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Download Error", str(e))

    def closeEvent(self, event) -> None:
        for worker in (self._worker, self._review_worker):
            if worker and worker.isRunning():
                worker.quit()
                worker.wait()
        super().closeEvent(event)
