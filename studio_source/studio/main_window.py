from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from evivn import EviVNInputs

from . import __version__
from .data_io import DatasetSummary, load_npz, make_demo_dataset, save_npz_template
from .exporter import export_run
from .pages import ConfigurationPage, DataPage, TrainingPage
from .runner import CompletedRun, TrainingSettings, train_evivn


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base / relative


class TrainingWorker(QObject):
    log = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        inputs: EviVNInputs,
        dataset: DatasetSummary,
        settings: TrainingSettings,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.dataset = dataset
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            result = train_evivn(
                self.inputs,
                self.dataset,
                self.settings,
                log=self.log.emit,
            )
            self.completed.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EviVN Studio")
        self.setMinimumSize(1080, 720)
        icon_path = resource_path("assets/evivn.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.inputs: EviVNInputs | None = None
        self.dataset: DatasetSummary | None = None
        self.completed_run: CompletedRun | None = None
        self._thread: QThread | None = None
        self._worker: TrainingWorker | None = None

        root = QWidget()
        root.setObjectName("AppRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_sidebar())

        self.pages = QStackedWidget()
        self.data_page = DataPage()
        self.configuration_page = ConfigurationPage()
        self.training_page = TrainingPage()
        self.pages.addWidget(self.data_page)
        self.pages.addWidget(self.configuration_page)
        self.pages.addWidget(self.training_page)
        root_layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        self.data_page.import_requested.connect(self._import_dataset)
        self.data_page.demo_requested.connect(self.load_demo)
        self.data_page.save_demo_requested.connect(self._save_demo)
        self.training_page.start_requested.connect(self._start_training)
        self.training_page.export_requested.connect(self._export_run)
        self._restore_window_state()

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(216)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 20, 14, 16)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        mark = QLabel("EV")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(42, 42)
        names = QVBoxLayout()
        name = QLabel("EviVN Studio")
        name.setObjectName("BrandName")
        version = QLabel(f"Version {__version__}")
        version.setObjectName("BrandVersion")
        names.addWidget(name)
        names.addWidget(version)
        brand.addWidget(mark)
        brand.addLayout(names)
        layout.addLayout(brand)
        layout.addSpacing(22)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        entries = (
            ("Dataset", QStyle.SP_DirOpenIcon),
            ("Configuration", QStyle.SP_ComputerIcon),
            ("Training", QStyle.SP_MediaPlay),
        )
        for index, (label, icon_type) in enumerate(entries):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setIcon(self.style().standardIcon(icon_type))
            button.clicked.connect(lambda _checked=False, page=index: self.pages.setCurrentIndex(page))
            self.nav_group.addButton(button, index)
            layout.addWidget(button)
            if index == 0:
                button.setChecked(True)
        layout.addStretch()
        self.sidebar_status = QLabel("No dataset")
        self.sidebar_status.setObjectName("SidebarStatus")
        self.sidebar_status.setWordWrap(True)
        layout.addWidget(self.sidebar_status)
        return sidebar

    @Slot()
    def _import_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Evi-VN dataset",
            str(Path.home()),
            "NumPy archive (*.npz)",
        )
        if not path:
            return
        try:
            inputs, summary = load_npz(path)
            self._set_dataset(inputs, summary)
        except Exception as error:
            QMessageBox.critical(self, "Dataset validation failed", str(error))

    @Slot()
    def load_demo(self) -> None:
        inputs, summary = make_demo_dataset()
        self._set_dataset(inputs, summary)

    @Slot()
    def _save_demo(self) -> None:
        inputs, _ = make_demo_dataset()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save demo dataset",
            str(Path.home() / "evivn_demo.npz"),
            "NumPy archive (*.npz)",
        )
        if path:
            save_npz_template(path, inputs)

    def _set_dataset(self, inputs: EviVNInputs, summary: DatasetSummary) -> None:
        self.inputs = inputs
        self.dataset = summary
        self.completed_run = None
        self.data_page.show_dataset(summary)
        self.training_page.dataset_ready(summary.name)
        self.sidebar_status.setText(
            f"{summary.name}\n{summary.nodes:,} nodes / {summary.edges:,} edges"
        )

    @Slot()
    def _start_training(self) -> None:
        if self.inputs is None or self.dataset is None:
            QMessageBox.warning(self, "Dataset required", "Load a valid dataset first.")
            return
        try:
            settings = self.configuration_page.settings()
        except Exception as error:
            QMessageBox.warning(self, "Invalid configuration", str(error))
            return

        self.completed_run = None
        self.pages.setCurrentIndex(2)
        for button in self.nav_group.buttons():
            if self.nav_group.id(button) == 2:
                button.setChecked(True)
        self.training_page.set_running(True)
        self._set_navigation_enabled(False)

        self._thread = QThread(self)
        self._worker = TrainingWorker(self.inputs, self.dataset, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.training_page.append_log)
        self._worker.completed.connect(self._training_completed)
        self._worker.failed.connect(self._training_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._training_thread_finished)
        self._thread.start()

    @Slot(object)
    def _training_completed(self, run: CompletedRun) -> None:
        self.completed_run = run
        self.training_page.set_running(False)
        self.training_page.show_run(run)

    @Slot(str)
    def _training_failed(self, message: str) -> None:
        self.training_page.set_running(False)
        self.training_page.show_failure(message)
        QMessageBox.critical(self, "Evi-VN training failed", message.splitlines()[-1])

    @Slot()
    def _training_thread_finished(self) -> None:
        self._set_navigation_enabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    @Slot()
    def _export_run(self) -> None:
        if self.inputs is None or self.completed_run is None:
            return
        default_name = f"evivn-run-{self.completed_run.run_id}.zip"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Evi-VN run",
            str(Path.home() / default_name),
            "Evi-VN run package (*.zip)",
        )
        if not path:
            return
        try:
            output = export_run(path, self.inputs, self.completed_run)
            QMessageBox.information(self, "Run exported", str(output))
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))

    def _set_navigation_enabled(self, enabled: bool) -> None:
        for button in self.nav_group.buttons():
            button.setEnabled(enabled or self.nav_group.id(button) == 2)

    def _restore_window_state(self) -> None:
        settings = QSettings("EviVN", "EviVN Studio")
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1240, 800)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(
                self,
                "Training in progress",
                "Wait for the current training run to finish before closing.",
            )
            event.ignore()
            return
        QSettings("EviVN", "EviVN Studio").setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
