"""Synthetic, headless GUI tests; never call TypeSafe or use customer PDFs."""
import csv
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from firstpage_jev_lab.demo import make_demo
from firstpage_jev_lab.gui import MainWindow
from firstpage_jev_lab.gui_service import Request, export_experiment, parse_page_ranges, run_experiment
from firstpage_jev_lab.jev import MockClassifier


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    app.quit()


def dispose_window(window, app):
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture
def synthetic(tmp_path):
    pdf, starts = make_demo(tmp_path / "demo.pdf")
    truth = tmp_path / "truth.csv"
    with truth.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["StartPage"])
        writer.writerows([[n] for n in starts])
    return pdf, starts, truth


def test_custom_ranges_and_rejections():
    assert parse_page_ranges("1-3,8,11-13", 20) == [1, 2, 3, 8, 11, 12, 13]
    for value in ("", "1", "1,1,3", "0-5", "10-8", "1-999", "abc", "1-3,4-", "1,,3"):
        with pytest.raises(ValueError):
            parse_page_ranges(value, 20)


def test_service_mock_full_scan_and_exports(synthetic, tmp_path):
    pdf, starts, truth = synthetic
    request = Request(str(pdf), backend="mock", sampling="custom",
                      custom_ranges="1-20", ground_truth=str(truth))
    stages = []
    result = run_experiment(request, stages.append)
    assert "Classifying" in " ".join(stages)
    assert result.sampled == list(range(1, 21))
    assert result.starts == starts
    assert result.ground_truth["exact_match"] is True
    assert result.report()["status"] == "EXPERIMENTAL_NOT_PRODUCTION_APPROVED"
    out = export_experiment(result, tmp_path / "results")
    assert (out / "candidate_index.csv").is_file()
    assert (out / "sample_decisions.csv").is_file()
    assert (out / "rule_candidates.csv").is_file()
    assert (out / "candidate_detection_config.json").is_file()
    assert (out / "report.json").is_file()


def test_service_fail_closed_before_api(synthetic):
    pdf, _, _ = synthetic
    with pytest.raises(ValueError, match="approval"):
        run_experiment(Request(str(pdf), backend="jev", allow_external_text=False),
                       classifier=MockClassifier())


def test_window_creation_selection_and_model(app, synthetic):
    pdf, _, truth = synthetic
    window = MainWindow()
    try:
        assert window.windowTitle().startswith("PDF FirstPage")
        assert window.backend_combo.currentData() == "mock"
        window.set_source(str(pdf), truth=str(truth))
        assert window._current_page == 1
        assert window.export_button.isEnabled() is False
        result = run_experiment(Request(str(pdf), ground_truth=str(truth)))
        window._render(result)
        assert window.metrics["pages"].text() == "20"
        assert window.metrics["envelopes"].text() == "7"
        assert window.metrics["validation"].text() == "MATCH"
        assert window.pred_model.rowCount() == 20
        assert window.rule_model.rowCount() > 0
        assert window.env_model.rowCount() == 7
        assert window.export_button.isEnabled()
        window.preview_page(4)
        assert window._current_page == 4
        window.set_source(str(pdf))
        assert window.result is None
        assert not window.export_button.isEnabled()
    finally:
        dispose_window(window, app)


def test_background_thread_keeps_gui_responsive(app, synthetic):
    pdf, _, truth = synthetic
    window = MainWindow()
    try:
        window.set_source(str(pdf), truth=str(truth))
        window.start_analysis()
        assert window._worker is not None
        deadline = time.monotonic() + 12
        while window._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.015)
        app.processEvents()
        assert window._worker is None
        assert window.result is not None
        assert window.result.ground_truth["exact_match"]
        assert window.run_button.isEnabled()
    finally:
        dispose_window(window, app)


def test_jev_requires_key_and_user_approval(app, synthetic, monkeypatch):
    pdf, _, _ = synthetic
    window = MainWindow()
    try:
        window.set_source(str(pdf))
        window.backend_combo.setCurrentIndex(1)
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            window._request()
        monkeypatch.setenv("TYPESAFE_API_KEY", "SYNTHETIC_TEST_ONLY")
        with pytest.raises(ValueError, match="approval"):
            window._request()
        window.consent.setChecked(True)
        assert window._request().allow_external_text is True
        window.backend_combo.setCurrentIndex(0)
        assert not window.consent.isChecked()
        assert window._request().allow_external_text is False
    finally:
        dispose_window(window, app)


def test_openrouter_requires_key_and_user_approval(app, synthetic, monkeypatch):
    pdf, _, _ = synthetic
    window = MainWindow()
    try:
        window.set_source(str(pdf))
        window.backend_combo.setCurrentIndex(2)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            window._request()
        monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_TEST_ONLY")
        with pytest.raises(ValueError, match="approval"):
            window._request()
        window.consent.setChecked(True)
        request = window._request()
        assert request.backend == "openrouter"
        assert request.allow_external_text is True
    finally:
        dispose_window(window, app)
