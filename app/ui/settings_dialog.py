from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(480, 500)
        self._build_ui()

    def _build_ui(self):
        from app.backend import get_available_venues, get_selected_venues

        layout = QVBoxLayout(self)

        venue_group = QGroupBox("Venues")
        venue_layout = QVBoxLayout(venue_group)

        select_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        select_row.addWidget(select_all_btn)
        select_row.addStretch()
        venue_layout.addLayout(select_row)

        self._venue_list = QListWidget()
        selected = set(get_selected_venues())
        for venue, year, track, count in get_available_venues():
            item = QListWidgetItem(f"{venue}  {year}  —  {track}  ({count:,} papers)")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if (venue, year, track) in selected
                else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, (venue, year, track))
            self._venue_list.addItem(item)
        venue_layout.addWidget(self._venue_list)
        layout.addWidget(venue_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self):
        for i in range(self._venue_list.count()):
            self._venue_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _save(self):
        from app.backend import set_selected_venues

        selected = []
        for i in range(self._venue_list.count()):
            item = self._venue_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.data(Qt.ItemDataRole.UserRole))
        set_selected_venues(selected)
        self.accept()
