"""PySide6 desktop GUI for the experimental first-page rule discovery lab.

No production actions. No secret persistence. External text uploads require
both an enabled Jev backend, per-run consent and a second confirmation dialog.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import fitz
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTabWidget, QTableView, QVBoxLayout, QWidget,
)

from .demo import make_demo
from .gui_service import Experiment, Request, export_experiment, run_experiment
from .jev import resolve_api_key

STYLESHEET = """
QMainWindow, QWidget#root { background: #111821; color: #edf3f9; }
QLabel { color: #edf3f9; }
QLabel[muted="true"] { color: #9faebe; }
QLabel[section="true"] { color: #56c2b2; font-weight: bold; font-size: 11px; }
QFrame#panel, QFrame#metric { background: #1b2633; border: 1px solid #2b394a; border-radius: 11px; }
QGroupBox { background: transparent; border: 0; margin-top: 7px; padding-top: 15px;
  color: #56c2b2; font-weight: 700; font-size: 11px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; }
QLineEdit, QSpinBox, QComboBox {
  background: #121d28; border: 1px solid #334456; border-radius: 7px;
  color: #f2f6fa; padding: 8px; selection-background-color: #237e79;
}
QLineEdit:disabled, QSpinBox:disabled { color: #778999; }
QComboBox QAbstractItemView { background: #17222f; color: white; selection-background-color: #237e79; }
QPushButton {
  background: #29394a; border: 1px solid #3d5264; border-radius: 7px;
  color: #f3f7fb; font-weight: 600; padding: 9px 13px;
}
QPushButton:hover { background: #345069; }
QPushButton:disabled { background: #1f2b37; color: #748698; border-color: #304051; }
QPushButton[primary="true"] { background: #20a895; border-color: #20a895; color: #061d1a; }
QPushButton[primary="true"]:hover { background: #49c9b7; }
QPushButton[primary="true"]:disabled { background: #315e5b; color: #98b6b2; }
QCheckBox { color: #c2cdda; spacing: 7px; }
QTabWidget::pane { background: #1b2633; border: 1px solid #2b394a;
 border-radius: 8px; }
QTabBar::tab { background: #172330; color: #a9b8c8;
 padding: 10px 13px; border: 1px solid #2b394a; border-bottom: none; }
QTabBar::tab:selected { background: #245850; color: #e8fffa; }
QTableView { background: #172330; color: #e4edf5; alternate-background-color: #1b2a37;
 gridline-color: #273a4b; border: 0; selection-background-color: #275c5a;
 selection-color: white; }
QHeaderView::section { background: #223344; color: #bed4dd; padding: 8px;
 border: 0; border-right: 1px solid #34495b; font-weight: 700; }
QProgressBar { background: #1d2b38; color: #e4f6f0; border: 1px solid #3b5962; border-radius: 5px; }
QProgressBar::chunk { background: #20a895; }
QScrollArea { border: 0; background: transparent; }
QScrollBar:vertical { background: #15212d; width: 11px; }
QScrollBar::handle:vertical { background: #436070; border-radius: 4px; min-height: 20px; }
"""


class RowsModel(QAbstractTableModel):
    def __init__(self, headers: list[str], parent=None):
        super().__init__(parent)
        self.headers = headers
        self.rows: list[list[str]] = []

    def set_rows(self, rows: list[list[str]]) -> None:
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.rows[index.row()][index.column()])
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() == 0:
            return Qt.AlignmentFlag.AlignCenter
        if role == Qt.ItemDataRole.ForegroundRole and index.column() == len(self.headers)-1:
            value = str(self.rows[index.row()][index.column()])
            if value in ("First?", "EXPERIMENTAL"):
                return QColor("#68d6c1")
            if value in ("Uncertain", "NO GROUND TRUTH"):
                return QColor("#e6b66d")
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return self.headers[section] if orientation == Qt.Orientation.Horizontal else str(section+1)


class AnalysisWorker(QThread):
    stage = Signal(str)
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, request: Request, parent=None):
        super().__init__(parent)
        self.request = request

    def run(self):
        try:
            result = run_experiment(self.request, progress=self.stage.emit)
        except Exception as error:
            # Never print remote provider error body or PDF extracted text.
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.result_ready.emit(result)


def _small_label(text: str, *, muted: bool = False) -> QLabel:
    label = QLabel(text)
    if muted:
        label.setProperty("muted", True)
    return label


def _button(text: str, *, primary=False) -> QPushButton:
    item = QPushButton(text)
    if primary:
        item.setProperty("primary", True)
    item.setCursor(Qt.CursorShape.PointingHandCursor)
    return item


def _table(headers: list[str]) -> tuple[QTableView, RowsModel]:
    model = RowsModel(headers)
    table = QTableView()
    table.setModel(model)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
    table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    table.verticalHeader().hide()
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.verticalHeader().setDefaultSectionSize(32)
    return table, model


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF FirstPage · Jev Lab — Research Only")
        self.setMinimumSize(990, 670)
        self.resize(1310, 835)
        self.result: Experiment | None = None
        self._worker: AnalysisWorker | None = None
        self._demo_root: str | None = None
        self._current_page = 1
        self._build_ui()
        self._update_backend()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 15)
        layout.setSpacing(14)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        brand = QLabel("PDF FirstPage   /   Jev Lab")
        brand.setStyleSheet("font-size: 24px; font-weight: 750; color: #effffc;")
        titles.addWidget(brand)
        titles.addWidget(_small_label("AI-assisted first-page rule discovery • Text-layer PDFs only", muted=True))
        header.addLayout(titles, stretch=1)
        badge = QLabel("EXPERIMENTAL   •   NOT FOR PRODUCTION")
        badge.setStyleSheet("background:#3e3231; color:#f4c6a6; border:1px solid #815a4b;"
                            "border-radius:8px; padding:10px; font-weight:700;")
        header.addWidget(badge)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(465)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QFrame()
        left.setObjectName("panel")
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(18, 16, 18, 18)
        left_col.setSpacing(10)

        source_box = QGroupBox("01   SOURCE PDF")
        source_col = QVBoxLayout(source_box)
        self.pdf_line = QLineEdit()
        self.pdf_line.setPlaceholderText("選擇可抽取文字嘅 PDF…")
        self.pdf_line.setReadOnly(True)
        source_col.addWidget(self.pdf_line)
        browse = _button("Open PDF…")
        browse.clicked.connect(self.choose_pdf)
        self.demo_button = _button("Create 20-page demo")
        self.demo_button.clicked.connect(self.create_demo)
        source_buttons = QHBoxLayout()
        source_buttons.addWidget(browse)
        source_buttons.addWidget(self.demo_button)
        source_col.addLayout(source_buttons)
        self.truth_line = QLineEdit()
        self.truth_line.setReadOnly(True)
        self.truth_line.setPlaceholderText("Optional ground-truth StartPage CSV")
        source_col.addWidget(self.truth_line)
        self.truth_button = _button("Load known StartPage CSV…")
        self.truth_button.clicked.connect(self.choose_truth)
        source_col.addWidget(self.truth_button)
        left_col.addWidget(source_box)

        sample_box = QGroupBox("02   SAMPLE STRATEGY")
        sample_col = QVBoxLayout(sample_box)
        self.sample_combo = QComboBox()
        self.sample_combo.addItem("First N pages", "first")
        self.sample_combo.addItem("Distributed across PDF", "distributed")
        self.sample_combo.addItem("Custom page ranges", "custom")
        sample_col.addWidget(self.sample_combo)
        self.sample_combo.currentIndexChanged.connect(self._update_sampling)
        spin_form = QFormLayout()
        self.sample_size = QSpinBox()
        self.sample_size.setRange(3, 200)
        self.sample_size.setValue(20)
        spin_form.addRow("Sample pages:", self.sample_size)
        sample_col.addLayout(spin_form)
        self.range_input = QLineEdit()
        self.range_input.setPlaceholderText("1-20,35,50-60")
        sample_col.addWidget(self.range_input)
        self.range_input.hide()
        sample_col.addWidget(_small_label("Jev labels only the sample. Rules scan the entire PDF.", muted=True))
        left_col.addWidget(sample_box)

        ai_box = QGroupBox("03   DECISION BACKEND")
        ai_col = QVBoxLayout(ai_box)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Mock • Fully offline", "mock")
        self.backend_combo.addItem("Jev • TypeSafe external API", "jev")
        self.backend_combo.addItem("OpenRouter • Jev 1.14 external API", "openrouter")
        self.backend_combo.currentIndexChanged.connect(self._update_backend)
        ai_col.addWidget(self.backend_combo)
        self.api_status = _small_label("", muted=True)
        self.api_status.setWordWrap(True)
        ai_col.addWidget(self.api_status)
        self.consent = QCheckBox("I have permission to send redacted PDF-derived text to the selected external API.")
        self.consent.setChecked(False)
        ai_col.addWidget(self.consent)
        self.security_note = _small_label(
            "Best-effort redaction is NOT a data-leak safeguard. Never use confidential "
            "customer documents without information-security approval.", muted=True)
        self.security_note.setWordWrap(True)
        ai_col.addWidget(self.security_note)
        left_col.addWidget(ai_box)

        self.run_button = _button("▶   Analyze PDF", primary=True)
        self.run_button.clicked.connect(self.start_analysis)
        left_col.addWidget(self.run_button)
        self.export_button = _button("Export CSV / JSON reports")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        left_col.addWidget(self.export_button)
        left_col.addStretch()
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right = QWidget()
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(5, 0, 0, 0)
        right_col.setSpacing(12)

        stats = QHBoxLayout()
        self.metrics: dict[str, QLabel] = {}
        for title, key in [("PDF PAGES", "pages"), ("SAMPLED", "sampled"),
                           ("PROPOSED ENVELOPES", "envelopes"), ("VERIFICATION", "validation")]:
            metric_frame = QFrame()
            metric_frame.setObjectName("metric")
            inner = QVBoxLayout(metric_frame)
            inner.setContentsMargins(13, 12, 13, 12)
            caption = _small_label(title, muted=True)
            caption.setStyleSheet("font-size:10px; color:#9faebe; font-weight:700;")
            inner.addWidget(caption)
            value = QLabel("—")
            value.setStyleSheet("color:#edfdfa; font-size:21px; font-weight:750;")
            inner.addWidget(value)
            stats.addWidget(metric_frame, stretch=1)
            self.metrics[key] = value
        right_col.addLayout(stats)

        self.tabs = QTabWidget()
        self.pred_table, self.pred_model = _table(["PDF page", "First page probability", "Interpretation"])
        self.rule_table, self.rule_model = _table(
            ["Rank", "Rule", "Marker / Region", "Positive matches", "False-positive pseudo-labels", "Status"])
        self.env_table, self.env_model = _table(["Envelope", "StartPage", "EndPage", "Pages", "Status"])
        self.pred_table.clicked.connect(self._preview_prediction_row)
        self.env_table.clicked.connect(self._preview_envelope_row)
        self.tabs.addTab(self.pred_table, "Page decisions")
        self.tabs.addTab(self.rule_table, "Discovered rules")
        self.tabs.addTab(self.env_table, "Envelope index")

        preview_box = QWidget()
        preview_col = QVBoxLayout(preview_box)
        preview_top = QHBoxLayout()
        self.preview_heading = QLabel("Select a PDF and run analysis to preview a page")
        self.preview_heading.setStyleSheet("font-weight:700; color:#e0f7f0;")
        preview_top.addWidget(self.preview_heading, 1)
        self.prev_button = _button("◀")
        self.next_button = _button("▶")
        self.prev_button.clicked.connect(lambda: self.preview_page(self._current_page - 1))
        self.next_button.clicked.connect(lambda: self.preview_page(self._current_page + 1))
        preview_top.addWidget(self.prev_button)
        preview_top.addWidget(self.next_button)
        preview_col.addLayout(preview_top)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.preview_image = QLabel("PDF page preview")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setStyleSheet("background:#f5f6f7; color:#536374; border-radius:6px;")
        self.preview_image.setMinimumSize(350, 400)
        scroll.setWidget(self.preview_image)
        preview_col.addWidget(scroll, 1)
        self.tabs.addTab(preview_box, "PDF preview")
        right_col.addWidget(self.tabs, 1)
        self.result_hint = _small_label(
            "Start with Create 20-page demo → Analyze PDF. No external API is used in Mock mode.",
            muted=True)
        self.result_hint.setWordWrap(True)
        right_col.addWidget(self.result_hint)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 900])
        self._browse_button = browse

        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(150)
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.status = _small_label("Ready · Local offline mock selected", muted=True)
        footer.addWidget(self.progress)
        footer.addWidget(self.status, 1)
        layout.addLayout(footer)
        self._update_sampling()

    def _update_sampling(self, *_args):
        custom = self.sample_combo.currentData() == "custom"
        self.range_input.setVisible(custom)
        self.sample_size.setEnabled(not custom)
        self._invalidate()

    def _update_backend(self, *_args):
        backend = self.backend_combo.currentData()
        live = backend != "mock"
        self.consent.setVisible(live)
        self.security_note.setVisible(live)
        self.consent.setChecked(False)
        if not live:
            self.api_status.setText("Mock is a deterministic fixture heuristic, not actual Jev predictions.")
        elif backend == "jev":
            self.api_status.setText(
                "Jev API key detected in environment; key never displayed or saved."
                if resolve_api_key()
                else "TYPESAFE_API_KEY (or OPENCODE_API_KEY) missing. Set one in the shell before launching.")
        else:
            self.api_status.setText(
                "OpenRouter API key detected in environment; key never displayed or saved."
                if os.getenv("OPENROUTER_API_KEY")
                else "OPENROUTER_API_KEY missing. Set it in the shell before launching.")
        self._invalidate()

    def _invalidate(self):
        if not hasattr(self, "result"):
            return
        self.result = None
        if not hasattr(self, "export_button"):
            return
        self.export_button.setEnabled(False)
        self.pred_model.set_rows([])
        self.rule_model.set_rows([])
        self.env_model.set_rows([])
        for value in self.metrics.values():
            value.setText("—")

    def set_source(self, path: str, *, truth: str = ""):
        if self._worker is not None:
            return
        self.pdf_line.setText(str(Path(path).resolve()))
        self.pdf_line.setToolTip(self.pdf_line.text())
        self.truth_line.setText(truth)
        self.consent.setChecked(False)
        self._invalidate()
        self.status.setText(f"Selected local file: {Path(path).name}")
        self.preview_page(1)

    def choose_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a text-layer PDF", "", "PDF files (*.pdf)")
        if path:
            self.set_source(path)

    def create_demo(self):
        try:
            self._demo_root = tempfile.mkdtemp(prefix="firstpage-jev-lab-")
            pdf, starts = make_demo(Path(self._demo_root) / "synthetic.pdf")
            truth = pdf.with_name("synthetic_truth.csv")
            import csv
            with truth.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["StartPage"])
                writer.writerows([[n] for n in starts])
            self.set_source(str(pdf), truth=str(truth))
            self.status.setText("Synthetic 20-page PDF loaded with independently generated ground truth.")
        except Exception as error:
            QMessageBox.critical(self, "Unable to create demo", str(error))

    def choose_truth(self):
        path, _ = QFileDialog.getOpenFileName(self, "Independent ground-truth StartPage CSV", "", "CSV (*.csv)")
        if path:
            self.truth_line.setText(path)
            self._invalidate()

    def _request(self) -> Request:
        path = self.pdf_line.text().strip()
        if not path or not Path(path).is_file():
            raise ValueError("Select a valid PDF first, or create the synthetic demo.")
        backend = self.backend_combo.currentData()
        if backend != "mock":
            if backend == "jev" and not resolve_api_key():
                raise ValueError("Set TYPESAFE_API_KEY or OPENCODE_API_KEY in the environment, then restart this GUI.")
            if backend == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
                raise ValueError("Set OPENROUTER_API_KEY in the environment, then restart this GUI.")
            if not self.consent.isChecked():
                raise ValueError("External text transfer requires the explicit approval checkbox.")
        return Request(
            pdf=path, backend=backend, sampling=self.sample_combo.currentData(),
            sample_count=self.sample_size.value(), custom_ranges=self.range_input.text().strip(),
            ground_truth=self.truth_line.text().strip(),
            allow_external_text=backend != "mock" and self.consent.isChecked())

    def start_analysis(self):
        if self._worker is not None:
            return
        try:
            request = self._request()
        except ValueError as error:
            QMessageBox.warning(self, "Check experiment settings", str(error))
            return
        if request.backend == "jev":
            confirmation = QMessageBox.warning(
                self, "Transmit PDF-derived text outside this computer?",
                "The selected PDF's sample text will be sent to TypeSafe's Jev API.\n\n"
                "Best-effort redaction is NOT guaranteed to remove customer data. "
                "Only approved or synthetic documents are permitted.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if confirmation != QMessageBox.StandardButton.Yes:
                return
        self._invalidate()
        self._worker = AnalysisWorker(request)  # not a child of the QWidget
        self._worker.stage.connect(self.status.setText)
        self._worker.result_ready.connect(self._render)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._run_finished)
        self._set_busy(True)
        self._worker.start()

    def _set_busy(self, busy: bool):
        for element in (self._browse_button, self.demo_button, self.truth_button,
                        self.sample_combo, self.sample_size, self.range_input,
                        self.backend_combo, self.consent, self.run_button):
            element.setEnabled(not busy)
        self.progress.setVisible(busy)

    def _run_finished(self):
        worker = self._worker
        self._worker = None
        self._set_busy(False)
        if worker is not None:
            worker.wait()  # Qt finished has been emitted; join before queued disposal
            worker.deleteLater()

    def _failed(self, message: str):
        self.status.setText("Experiment stopped: " + message)
        QMessageBox.warning(self, "Analysis failed", message)

    def _render(self, result: Experiment):
        self.result = result
        self.metrics["pages"].setText(f"{result.page_count:,}")
        self.metrics["sampled"].setText(str(len(result.sampled)))
        self.metrics["envelopes"].setText(f"{len(result.groups):,}")
        exact = result.ground_truth and result.ground_truth["exact_match"]
        self.metrics["validation"].setText("MATCH" if exact else "MISMATCH" if result.ground_truth else "UNVERIFIED")
        self.metrics["validation"].setStyleSheet(
            "font-weight:750; font-size:17px; color:" + ("#71dbc7" if exact else "#f0bc79") + ";")
        self.pred_model.set_rows([
            [number, f"{result.probabilities[number]*100:.1f}%",
             "First?" if result.probabilities[number] >= 0.8 else
             "Continue?" if result.probabilities[number] <= 0.2 else "Uncertain"]
            for number in result.sampled])
        self.rule_model.set_rows([
            [i, c.rule.kind, c.rule.term or
             (str(tuple(round(x, 1) for x in c.rule.region_mm)) if c.rule.region_mm else "Auto page counter"),
             f"{c.tp}/{c.tp+c.fn}", str(c.fp), "ACTIVE" if i == result.selected_rank + 1 else "CANDIDATE"]
            for i, c in enumerate(result.candidates, 1)])
        self.env_model.set_rows([
            [i, start, end, end-start+1, "EXPERIMENTAL"]
            for i, (start, end) in enumerate(result.groups, 1)])
        self.export_button.setEnabled(True)
        self.result_hint.setText(
            "Mock probabilities are synthetic heuristics; model scores are uncalibrated. "
            "Rule metrics compare pseudo-labels only. Even a ground-truth match here is a lab test, "
            "not a production authorization.")
        if result.sampled:
            self.preview_page(result.sampled[0])
        summary = ("MATCHED independent StartPage CSV" if exact else
                   "DISAGREES with independent StartPage CSV" if result.ground_truth else
                   "no independent ground truth available")
        self.status.setText(f"Finished: {len(result.groups):,} candidate envelopes; {summary}. NOT production approved.")

    def export(self):
        if self.result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose directory for experimental CSV / JSON")
        if not folder:
            return
        try:
            destination = export_experiment(self.result, folder)
        except OSError as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self.status.setText(f"Exported experimental reports to {destination}")
        QMessageBox.information(self, "Export completed",
                                f"Unverified experiment files saved to:\n{destination}")

    def _preview_prediction_row(self, index: QModelIndex):
        if index.isValid() and self.result:
            self.preview_page(self.result.sampled[index.row()])

    def _preview_envelope_row(self, index: QModelIndex):
        if index.isValid() and self.result:
            self.preview_page(self.result.groups[index.row()][0])

    def preview_page(self, number: int):
        path = self.pdf_line.text().strip()
        if not path or not Path(path).is_file() or number < 1:
            return
        try:
            with fitz.open(path) as pdf:
                if number > pdf.page_count:
                    return
                page = pdf[number - 1]
                scale = min(1.4, 640 / max(page.rect.width, page.rect.height))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = QImage(pix.samples, pix.width, pix.height, pix.stride,
                               QImage.Format.Format_RGB888).copy()
            self.preview_image.setPixmap(QPixmap.fromImage(image))
            self.preview_heading.setText(f"PDF source page {number}  •  local preview only")
            self._current_page = number
        except Exception as error:
            self.preview_heading.setText(f"Preview unavailable: {error}")

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Analysis in progress",
                                    "Finish the current analysis before closing, to avoid corrupting an experiment.")
            event.ignore()
            return
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Arial", 10))
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
