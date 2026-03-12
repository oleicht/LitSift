import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
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
        from app.backend import config, get_available_venues

        layout = QVBoxLayout(self)

        venue_group = QGroupBox("Venues")
        venue_layout = QVBoxLayout(venue_group)
        self._venue_list = QListWidget()
        selected = {
            (v["venue"], v["year"], v["track"])
            for v in config["openreview"]["venues"]
        }
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

        model_group = QGroupBox("Embedding Model")
        model_layout = QVBoxLayout(model_group)
        model_layout.addWidget(QLabel(config["ranking"]["model"]))
        layout.addWidget(model_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self):
        from app.backend import config, reload_config

        selected = []
        for i in range(self._venue_list.count()):
            item = self._venue_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                venue, year, track = item.data(Qt.ItemDataRole.UserRole)
                selected.append({"venue": venue, "year": year, "track": track})

        config["openreview"]["venues"] = selected
        user_json_path = Path(__file__).parent.parent / "user.json"
        with open(user_json_path, "w") as f:
            json.dump(config, f, indent=4)

        reload_config()
        self.accept()
