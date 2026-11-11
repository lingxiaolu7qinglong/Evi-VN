from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .data_io import DatasetSummary
from .registry import (
    ESTIMATORS,
    LLM_PRIOR_KEY,
    LLM_PRIOR_LABEL,
    backbone_entries,
)
from .runner import CompletedRun, TrainingSettings
from .widgets import MetricTile, page_heading


class DataPage(QWidget):
    import_requested = Signal()
    demo_requested = Signal()
    save_demo_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)
        title, subtitle = page_heading(
            "Dataset",
            "Feature-isolated graph and evidence arrays with fixed train, development, and test splits.",
        )
        layout.addWidget(title)
        layout.addWidget(subtitle)

        toolbar = QFrame()
        toolbar.setObjectName("Toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 10, 12, 10)
        self.import_button = QPushButton("Import NPZ")
        self.import_button.setObjectName("PrimaryButton")
        self.demo_button = QPushButton("Load demo")
        self.save_demo_button = QPushButton("Save demo NPZ")
        self.source_label = QLabel("No dataset loaded")
        self.source_label.setObjectName("PageSubtitle")
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        toolbar_layout.addWidget(self.import_button)
        toolbar_layout.addWidget(self.demo_button)
        toolbar_layout.addWidget(self.save_demo_button)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(self.source_label, 1)
        layout.addWidget(toolbar)

        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.node_tile = MetricTile("NODES")
        self.edge_tile = MetricTile("EDGES")
        self.graph_tile = MetricTile("GNN FEATURES")
        self.evidence_tile = MetricTile("EVIDENCE FEATURES")
        self.rate_tile = MetricTile("POSITIVE RATE")
        for tile in (
            self.node_tile,
            self.edge_tile,
            self.graph_tile,
            self.evidence_tile,
            self.rate_tile,
        ):
            metrics.addWidget(tile, 1)
        layout.addLayout(metrics)

        section = QLabel("Split and isolation checks")
        section.setObjectName("SectionTitle")
        layout.addWidget(section)
        self.table = QTableWidget(5, 2)
        self.table.setHorizontalHeaderLabels(("Check", "Value"))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table, 1)

        self.import_button.clicked.connect(self.import_requested)
        self.demo_button.clicked.connect(self.demo_requested)
        self.save_demo_button.clicked.connect(self.save_demo_requested)

    def show_dataset(self, summary: DatasetSummary) -> None:
        self.source_label.setText(summary.source)
        self.node_tile.set_value(f"{summary.nodes:,}")
        self.edge_tile.set_value(f"{summary.edges:,}")
        self.graph_tile.set_value(str(summary.graph_dimensions))
        self.evidence_tile.set_value(str(summary.evidence_dimensions))
        self.rate_tile.set_value(f"{summary.positive_rate:.1%}")
        rows = (
            ("Training split", f"{summary.train_nodes:,} nodes; labels available"),
            ("Development split", f"{summary.dev_nodes:,} nodes; selection only"),
            ("Test split", f"{summary.test_nodes:,} nodes; reporting only"),
            ("Feature isolation", "GNN-side and evidence-side matrices are separate"),
            ("Archive safety", "Loaded with allow_pickle=False"),
        )
        for row, (name, value) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(value))
        self.table.resizeRowsToContents()


class ConfigurationPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content.setObjectName("Page")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)
        title, subtitle = page_heading(
            "Configuration",
            "Target backbone, OOF discovery, routed evidence branch, and virtual-node training.",
        )
        layout.addWidget(title)
        layout.addWidget(subtitle)

        columns = QHBoxLayout()
        columns.setSpacing(24)
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        target_group = QGroupBox("Target and discovery backbones")
        target_layout = QFormLayout(target_group)
        self.target_combo = QComboBox()
        self.discovery_list = QListWidget()
        self.discovery_list.setMinimumHeight(276)
        for key, label in backbone_entries():
            self.target_combo.addItem(label, key)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.discovery_list.addItem(item)
        self.target_combo.setCurrentIndex(1)
        target_layout.addRow("Target backbone", self.target_combo)
        target_layout.addRow("Discovery set", self.discovery_list)
        left.addWidget(target_group)

        evidence_group = QGroupBox("Evidence branch")
        evidence_layout = QFormLayout(evidence_group)
        self.router_combo = QComboBox()
        self.prior_combo = QComboBox()
        for entry in ESTIMATORS:
            self.router_combo.addItem(entry.label, entry.key)
            self.prior_combo.addItem(entry.label, entry.key)
        self.prior_combo.addItem(LLM_PRIOR_LABEL, LLM_PRIOR_KEY)
        self.route_threshold = self._double_spin(0.0, 1.0, 0.07, 0.01)
        self.uncertainty = self._double_spin(0.0, 0.50, 0.20, 0.05)
        evidence_layout.addRow("Router", self.router_combo)
        evidence_layout.addRow("Prior generator", self.prior_combo)
        evidence_layout.addRow("Fixed router threshold", self.route_threshold)
        evidence_layout.addRow("Uncertainty delta", self.uncertainty)
        left.addWidget(evidence_group)

        self.llm_group = QGroupBox("LLM prior connection")
        llm_layout = QFormLayout(self.llm_group)
        self.llm_model = QLineEdit()
        self.llm_model.setPlaceholderText("Model identifier")
        self.llm_endpoint = QLineEdit("https://api.openai.com/v1")
        self.llm_endpoint.setPlaceholderText("API base URL")
        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_api_key.setPlaceholderText("API key")
        self.llm_show_key = QCheckBox("Show API key")
        self.llm_share_confirm = QCheckBox("Allow evidence to be sent to this endpoint")
        self.llm_group.setToolTip(
            "The API key stays in memory and is excluded from exported run packages."
        )
        llm_layout.addRow("Model", self.llm_model)
        llm_layout.addRow("API endpoint", self.llm_endpoint)
        llm_layout.addRow("API key", self.llm_api_key)
        llm_layout.addRow("Key visibility", self.llm_show_key)
        llm_layout.addRow("External processing", self.llm_share_confirm)
        left.addWidget(self.llm_group)
        self.prior_combo.currentIndexChanged.connect(self._sync_llm_controls)
        self.llm_show_key.toggled.connect(
            lambda visible: self.llm_api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
            )
        )
        self._sync_llm_controls()
        left.addStretch()

        training_group = QGroupBox("Training protocol")
        training_layout = QFormLayout(training_group)
        self.fold_count = self._spin(2, 10, 3)
        self.hidden_dim = self._spin(16, 512, 64, 16)
        self.max_epochs = self._spin(1, 1000, 80)
        self.oof_epochs = self._spin(1, 1000, 55)
        self.patience = self._spin(1, 200, 20)
        self.seed = self._spin(0, 999999, 42)
        training_layout.addRow("OOF folds", self.fold_count)
        training_layout.addRow("Hidden dimensions", self.hidden_dim)
        training_layout.addRow("Full epochs", self.max_epochs)
        training_layout.addRow("OOF epochs", self.oof_epochs)
        training_layout.addRow("Patience", self.patience)
        training_layout.addRow("Random seed", self.seed)
        right.addWidget(training_group)

        virtual_group = QGroupBox("Virtual-node injection")
        virtual_layout = QFormLayout(virtual_group)
        self.tau_negative = self._double_spin(0.05, 0.49, 0.40, 0.05)
        self.tau_positive = self._double_spin(0.51, 0.95, 0.60, 0.05)
        self.edge_scale = self._double_spin(0.25, 12.0, 2.0, 0.25)
        self.compact_tuning = QCheckBox("Tune on dev Accuracy")
        self.compact_tuning.setChecked(True)
        self.compact_tuning.setToolTip(
            "Search router thresholds 0.01--0.99 using final development accuracy."
        )
        self.route_threshold.setEnabled(False)
        self.compact_tuning.toggled.connect(
            lambda enabled: self.route_threshold.setEnabled(not enabled)
        )
        virtual_layout.addRow("Negative threshold", self.tau_negative)
        virtual_layout.addRow("Positive threshold", self.tau_positive)
        virtual_layout.addRow("Edge scale", self.edge_scale)
        virtual_layout.addRow("Model selection", self.compact_tuning)
        right.addWidget(virtual_group)
        right.addStretch()

        layout.addLayout(columns)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def settings(self) -> TrainingSettings:
        discovery = tuple(
            self.discovery_list.item(index).data(Qt.UserRole)
            for index in range(self.discovery_list.count())
            if self.discovery_list.item(index).checkState() == Qt.Checked
        )
        if not discovery:
            raise ValueError("Select at least one discovery backbone")
        if self.target_combo.currentData() == "gat" and self.hidden_dim.value() % 4:
            raise ValueError("GAT hidden dimensions must be divisible by four")
        prior = str(self.prior_combo.currentData())
        if prior == LLM_PRIOR_KEY:
            if not self.llm_model.text().strip():
                raise ValueError("Enter the LLM model identifier")
            if not self.llm_endpoint.text().strip():
                raise ValueError("Enter the LLM API endpoint")
            if not self.llm_share_confirm.isChecked():
                raise ValueError("Confirm evidence sharing for the configured LLM endpoint")
        return TrainingSettings(
            target_backbone=str(self.target_combo.currentData()),
            discovery_backbones=discovery,
            router=str(self.router_combo.currentData()),
            prior=prior,
            folds=self.fold_count.value(),
            seed=self.seed.value(),
            hidden_dim=self.hidden_dim.value(),
            max_epochs=self.max_epochs.value(),
            patience=self.patience.value(),
            oof_epochs=self.oof_epochs.value(),
            oof_patience=max(2, self.patience.value() // 2),
            route_threshold=self.route_threshold.value(),
            uncertainty_delta=self.uncertainty.value(),
            tau_negative=self.tau_negative.value(),
            tau_positive=self.tau_positive.value(),
            edge_scale=self.edge_scale.value(),
            compact_tuning=self.compact_tuning.isChecked(),
            llm_model=self.llm_model.text().strip(),
            llm_endpoint=self.llm_endpoint.text().strip(),
            llm_api_key=self.llm_api_key.text(),
            llm_data_sharing_confirmed=self.llm_share_confirm.isChecked(),
        )

    def _sync_llm_controls(self) -> None:
        self.llm_group.setVisible(self.prior_combo.currentData() == LLM_PRIOR_KEY)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setSingleStep(step)
        return control

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, value: float, step: float
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setSingleStep(step)
        control.setDecimals(2)
        return control


class TrainingPage(QWidget):
    start_requested = Signal()
    export_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)
        title, subtitle = page_heading(
            "Training and evaluation",
            "OOF hard-region discovery, routed evidence priors, virtual-node training, and locked test reporting.",
        )
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("Toolbar")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        self.start_button = QPushButton("Start Evi-VN")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setEnabled(False)
        self.export_button = QPushButton("Export run")
        self.export_button.setEnabled(False)
        self.status = QLabel("Load a dataset to begin")
        self.status.setObjectName("PageSubtitle")
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.status, 1)
        layout.addWidget(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        metric_layout = QHBoxLayout()
        metric_layout.setSpacing(10)
        self.accuracy = MetricTile("TEST ACCURACY")
        self.auc = MetricTile("TEST AUC")
        self.f1 = MetricTile("TEST F1+")
        self.hard = MetricTile("TRAIN HARD RATE")
        self.route = MetricTile("TEST ROUTE RATE")
        for tile in (self.accuracy, self.auc, self.f1, self.hard, self.route):
            metric_layout.addWidget(tile, 1)
        layout.addLayout(metric_layout)

        lower = QHBoxLayout()
        lower.setSpacing(16)
        result_panel = QFrame()
        result_panel.setObjectName("Panel")
        result_layout = QVBoxLayout(result_panel)
        result_title = QLabel("Selected configuration")
        result_title.setObjectName("SectionTitle")
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(("Component", "Selection"))
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionMode(QTableWidget.NoSelection)
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_table)

        log_panel = QFrame()
        log_panel.setObjectName("Panel")
        log_layout = QVBoxLayout(log_panel)
        log_title = QLabel("Run log")
        log_title.setObjectName("SectionTitle")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log)
        lower.addWidget(result_panel, 1)
        lower.addWidget(log_panel, 2)
        layout.addLayout(lower, 1)

        self.start_button.clicked.connect(self.start_requested)
        self.export_button.clicked.connect(self.export_requested)

    def dataset_ready(self, name: str) -> None:
        self.start_button.setEnabled(True)
        self.status.setText(f"Ready: {name}")

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.export_button.setEnabled(False if running else self.export_button.isEnabled())
        if running:
            self.progress.setRange(0, 0)
            self.status.setText("Training")
            self.log.clear()
            self._clear_metrics()
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)

    def append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def show_run(self, run: CompletedRun) -> None:
        metrics = run.test_metrics
        self.accuracy.set_value(f"{metrics['accuracy']:.4f}")
        self.auc.set_value(f"{metrics['auc']:.4f}")
        self.f1.set_value(f"{metrics['f1']:.4f}")
        self.hard.set_value(f"{run.hard_rate:.1%}")
        self.route.set_value(f"{run.test_route_rate:.1%}")
        self.result_table.setRowCount(len(run.selected_configuration) + 2)
        rows = list(run.selected_configuration.items()) + [
            ("target_backbone", run.settings.target_backbone),
            ("elapsed", f"{run.elapsed_seconds:.1f} s"),
        ]
        for row, (key, value) in enumerate(rows):
            self.result_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.result_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.result_table.resizeRowsToContents()
        self.status.setText(f"Completed: {run.run_id}")
        self.export_button.setEnabled(True)

    def show_failure(self, message: str) -> None:
        self.status.setText("Training failed")
        self.append_log(message)

    def _clear_metrics(self) -> None:
        for tile in (self.accuracy, self.auc, self.f1, self.hard, self.route):
            tile.set_value("--")
