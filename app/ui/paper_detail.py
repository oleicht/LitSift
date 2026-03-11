from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from app.ui.paper_list import Paper


class PaperDetailDialog(QDialog):
    def __init__(self, paper: Paper, parent=None):
        super().__init__(parent)
        self._paper = paper
        self.setWindowTitle(paper.title)
        self.resize(700, 500)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        download_btn = QPushButton("Download")
        download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(download_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        text = QTextEdit()
        text.setPlainText(self._paper.abstract)
        text.setReadOnly(True)
        layout.addWidget(text)

    def _on_download(self) -> None:
        from app.ranking import download
        try:
            download(self._paper.title)
            QMessageBox.information(self, "Download", "Downloaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Download Error", str(e))
