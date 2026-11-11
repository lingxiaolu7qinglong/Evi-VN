from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _self_test(output_path: str) -> int:
    from studio.data_io import make_demo_dataset
    from studio.runner import TrainingSettings, train_evivn

    inputs, summary = make_demo_dataset(seed=7, node_count=48)
    settings = TrainingSettings(
        target_backbone="gcn",
        discovery_backbones=("mlp", "gcn"),
        router="logistic",
        prior="logistic",
        folds=2,
        hidden_dim=16,
        max_epochs=2,
        patience=1,
        oof_epochs=2,
        oof_patience=1,
        compact_tuning=False,
    )
    run = train_evivn(inputs, summary, settings)
    report = {
        "status": "ok",
        "accuracy": run.test_metrics["accuracy"],
        "auc": run.test_metrics["auc"],
        "f1": run.test_metrics["f1"],
        "hard_rate": run.hard_rate,
    }
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        position = sys.argv.index("--self-test")
        if position + 1 >= len(sys.argv):
            return 2
        output_path = sys.argv[position + 1]
        try:
            return _self_test(output_path)
        except Exception as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            Path(output_path).write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return 1

    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    from studio.main_window import MainWindow
    from studio.styles import APP_STYLE

    app = QApplication(sys.argv)
    app.setApplicationName("EviVN Studio")
    app.setOrganizationName("EviVN")
    app.setStyle("Fusion")
    for font_path in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(font_path).exists() and QFontDatabase.addApplicationFont(font_path) >= 0:
            break
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
