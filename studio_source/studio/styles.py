APP_STYLE = """
QWidget {
    color: #18222d;
    font-family: "Segoe UI";
    font-size: 10pt;
    letter-spacing: 0px;
}
QMainWindow, QWidget#AppRoot, QStackedWidget {
    background: #f4f6f7;
}
QFrame#Sidebar {
    background: #17212b;
    border: 0;
}
QLabel#BrandMark {
    background: #0d8a7a;
    color: white;
    border-radius: 4px;
    font-size: 15pt;
    font-weight: 700;
}
QLabel#BrandName {
    color: white;
    font-size: 14pt;
    font-weight: 650;
}
QLabel#BrandVersion, QLabel#SidebarStatus {
    color: #9fb0bf;
    font-size: 9pt;
}
QPushButton#NavButton {
    color: #c6d1da;
    background: transparent;
    border: 0;
    border-left: 3px solid transparent;
    padding: 10px 14px;
    text-align: left;
    min-height: 24px;
}
QPushButton#NavButton:hover {
    background: #22313f;
    color: white;
}
QPushButton#NavButton:checked {
    background: #203b3b;
    color: white;
    border-left-color: #31b7a5;
}
QWidget#Page {
    background: #f4f6f7;
}
QLabel#PageTitle {
    color: #101820;
    font-size: 20pt;
    font-weight: 650;
}
QLabel#PageSubtitle {
    color: #66727d;
    font-size: 10pt;
}
QLabel#SectionTitle {
    color: #25313d;
    font-size: 11pt;
    font-weight: 650;
}
QFrame#Toolbar, QFrame#MetricTile, QFrame#Panel {
    background: white;
    border: 1px solid #dce2e5;
    border-radius: 6px;
}
QFrame#MetricTile {
    min-height: 74px;
}
QLabel#MetricLabel {
    color: #69757f;
    font-size: 9pt;
    font-weight: 600;
}
QLabel#MetricValue {
    color: #14202a;
    font-size: 18pt;
    font-weight: 700;
}
QPushButton {
    background: white;
    border: 1px solid #cbd4d9;
    border-radius: 4px;
    padding: 7px 12px;
    min-height: 20px;
}
QPushButton:hover {
    border-color: #0d8a7a;
    color: #086f63;
}
QPushButton:disabled {
    background: #eef1f2;
    border-color: #dde2e5;
    color: #9aa4ab;
}
QPushButton#PrimaryButton {
    background: #0d8a7a;
    border-color: #0d8a7a;
    color: white;
    font-weight: 650;
}
QPushButton#PrimaryButton:hover {
    background: #087466;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QPlainTextEdit, QTableWidget {
    background: white;
    border: 1px solid #cfd7dc;
    border-radius: 4px;
    padding: 5px;
    selection-background-color: #b9e6df;
    selection-color: #101820;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus {
    border-color: #0d8a7a;
}
QGroupBox {
    background: transparent;
    border: 0;
    margin-top: 20px;
    font-weight: 650;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 0;
    color: #25313d;
}
QHeaderView::section {
    background: #e9edef;
    color: #43515d;
    border: 0;
    border-bottom: 1px solid #ccd5da;
    padding: 7px;
    font-weight: 650;
}
QTableWidget {
    gridline-color: #e6eaec;
}
QProgressBar {
    border: 1px solid #cfd7dc;
    border-radius: 3px;
    background: #edf1f2;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #0d8a7a;
}
QLabel#StatusGood {
    color: #087466;
    font-weight: 650;
}
QLabel#StatusBad {
    color: #bf3d3d;
    font-weight: 650;
}
"""
