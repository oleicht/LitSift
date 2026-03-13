from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut, QTextBlockFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui.paper import Paper
from app.ui.review_dialog import ReviewDialog
from app.ui.review_worker import ReviewWorker
from app.ui.worker import RankingWorker

_MAX_AUTHORS = 4


def _format_authors(authors: tuple[str, ...]) -> str:
    if not authors:
        return "Unknown authors"
    if len(authors) <= _MAX_AUTHORS:
        return ", ".join(authors)
    return ", ".join(authors[:_MAX_AUTHORS]) + " et al."


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LitSift")
        self.resize(900, 700)
        self._worker: RankingWorker | None = None
        self._review_worker: ReviewWorker | None = None
        self._papers: list[Paper] = []
        self._idx: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Query row
        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("Query:"))
        self._query_input = QLineEdit()
        self._query_input.setPlaceholderText("e.g. diffusion models for protein design")
        self._query_input.returnPressed.connect(self._on_submit)
        query_row.addWidget(self._query_input)
        self._submit_btn = QPushButton("Submit")
        self._submit_btn.clicked.connect(self._on_submit)
        query_row.addWidget(self._submit_btn)
        layout.addLayout(query_row)

        # Navigation row
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("←")
        self._prev_btn.setFixedWidth(40)
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._on_prev)
        nav_row.addWidget(self._prev_btn)

        self._rank_label = QLabel("")
        self._rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self._rank_label, 1)

        self._next_btn = QPushButton("→")
        self._next_btn.setFixedWidth(40)
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next)
        nav_row.addWidget(self._next_btn)

        nav_row.addSpacing(20)
        self._score_label = QLabel("")
        nav_row.addWidget(self._score_label)
        layout.addLayout(nav_row)

        # Title
        self._title_label = QLabel("")
        self._title_label.setWordWrap(True)
        title_font = self._title_label.font()
        title_font.setPointSize(title_font.pointSize() + 3)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)

        # Authors + venue
        self._authors_label = QLabel("")
        self._authors_label.setWordWrap(True)
        layout.addWidget(self._authors_label)

        self._venue_label = QLabel("")
        layout.addWidget(self._venue_label)

        # TLDR
        layout.addWidget(QLabel("TLDR:"))
        self._tldr_view = QTextEdit()
        self._tldr_view.setReadOnly(True)
        self._tldr_view.document().setDocumentMargin(10)
        self._tldr_view.setMaximumHeight(80)
        layout.addWidget(self._tldr_view)

        # Abstract
        layout.addWidget(QLabel("Abstract:"))
        self._abstract_view = QTextEdit()
        self._abstract_view.setReadOnly(True)
        self._abstract_view.document().setDocumentMargin(10)
        layout.addWidget(self._abstract_view)

        # Action buttons
        btn_row = QHBoxLayout()
        self._download_btn = QPushButton("Download")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)

        self._reviews_btn = QPushButton("Show Reviews")
        self._reviews_btn.setEnabled(False)
        self._reviews_btn.clicked.connect(self._on_reviews)
        btn_row.addWidget(self._reviews_btn)

        btn_row.addStretch()

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.clicked.connect(self._on_settings)
        btn_row.addWidget(self._settings_btn)
        layout.addLayout(btn_row)

        base_font = self.font()
        base_font.setPointSize(base_font.pointSize() + 1)
        self._tldr_view.setFont(base_font)
        self._abstract_view.setFont(base_font)

        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self._on_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._on_next)
        QShortcut(QKeySequence("R"), self).activated.connect(self._on_reviews)
        QShortcut(QKeySequence("D"), self).activated.connect(self._on_download)
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(self._query_input.setFocus)
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self._query_input)
        esc.setContext(Qt.ShortcutContext.WidgetShortcut)
        esc.activated.connect(self.setFocus)

        self.statusBar().showMessage("Ready")

    def _show_paper(self, idx: int) -> None:
        paper = self._papers[idx]
        n = len(self._papers)

        self._rank_label.setText(f"Paper {idx + 1} of {n}")
        self._score_label.setText(f"Score: {paper.score:.2f}")
        self._title_label.setText(paper.title)
        self._authors_label.setText(_format_authors(paper.authors))
        self._venue_label.setText(paper.name)
        self._tldr_view.setPlainText(paper.tldr)
        self._abstract_view.setPlainText(paper.abstract)

        for view in (self._tldr_view, self._abstract_view):
            fmt = QTextBlockFormat()
            fmt.setLineHeight(140.0, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            cursor = view.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.mergeBlockFormat(fmt)

        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < n - 1)
        self._download_btn.setEnabled(True)
        self._reviews_btn.setEnabled(True)

    def _on_prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._show_paper(self._idx)

    def _on_next(self) -> None:
        if self._idx < len(self._papers) - 1:
            self._idx += 1
            self._show_paper(self._idx)

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

    def _on_results(self, papers: list[Paper]) -> None:
        self._papers = papers
        self._idx = 0
        if papers:
            self._show_paper(0)
        self.statusBar().showMessage(f"{len(papers)} papers ranked", 5000)

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Error: {msg}", 8000)

    def _on_reviews(self) -> None:
        if not self._papers:
            return
        if self._review_worker and self._review_worker.isRunning():
            return
        self._reviews_btn.setEnabled(False)
        self.statusBar().showMessage("Loading reviews…")
        paper = self._papers[self._idx]
        self._review_worker = ReviewWorker(paper, parent=self)
        self._review_worker.results_ready.connect(self._on_review_results)
        self._review_worker.error.connect(self._on_review_error)
        self._review_worker.finished.connect(
            lambda: self._reviews_btn.setEnabled(bool(self._papers))
        )
        self._review_worker.start()

    def _on_review_results(self, paper: Paper, notes: list) -> None:
        self.statusBar().clearMessage()
        ReviewDialog(paper, notes, parent=self).exec()

    def _on_review_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Error loading reviews: {msg}", 8000)

    def _on_settings(self) -> None:
        from app.ui.settings_dialog import SettingsDialog
        SettingsDialog(parent=self).exec()

    def _on_download(self) -> None:
        if not self._papers:
            return
        from app.backend import download
        try:
            download(self._papers[self._idx].title)
            QMessageBox.information(self, "Download", "Downloaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Download Error", str(e))

    def closeEvent(self, event) -> None:
        for worker in (self._worker, self._review_worker):
            if worker and worker.isRunning():
                worker.quit()
                worker.wait()
        super().closeEvent(event)
