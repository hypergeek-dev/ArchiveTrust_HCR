"""One real, headless smoke test for the GUI window itself -- confirms it actually constructs
without crashing, using Qt's `offscreen` platform plugin (no real display needed, the same mechanism
CI-safe Qt testing normally relies on). Everything else about GUI behavior is covered by
`test_gui_commands.py`'s pure command-construction tests, which need no Qt at all.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


def test_main_window_constructs_without_crashing():
    from PySide6.QtWidgets import QApplication

    from archivetrust.htr.training.full_run.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.windowTitle() == "Loghi Full-Corpus Training -- Operator Console"
        assert window._run_dir_edit.text() == ""
        assert window._hours_spin.value() == 5.0
        assert window._batch_size_spin.value() == 16
        assert window._force_kill_btn.isEnabled() is False
    finally:
        window.close()


def test_status_panel_has_a_label_for_every_required_field():
    from PySide6.QtWidgets import QApplication

    from archivetrust.htr.training.full_run.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        required = {
            "status", "monitoring_state", "current_epoch", "current_global_step",
            "latest_val_cer", "best_val_cer", "epochs_since_improvement",
            "latest_checkpoint", "best_checkpoint", "resumable", "stop_reason",
        }
        assert required.issubset(window._status_labels.keys())
    finally:
        window.close()


def test_refresh_status_does_not_crash_with_no_run_directory_selected():
    from PySide6.QtWidgets import QApplication

    from archivetrust.htr.training.full_run.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window._refresh_status()  # must not raise -- no run directory set yet
    finally:
        window.close()


def test_refresh_status_reflects_a_real_run_state(tmp_path):
    from PySide6.QtWidgets import QApplication

    from archivetrust.htr.training.full_run.gui.app import MainWindow
    from archivetrust.htr.training.full_run.run_state import create_initial_run_state, heartbeat, mark_running, save_run_state

    run_dir = tmp_path / "myrun"
    state = heartbeat(
        mark_running(create_initial_run_state(run_id="r1", configuration_hash="h1")),
        current_epoch=5, latest_metrics={"val_cer": 0.25}, best_metrics={"val_cer": 0.22},
    )
    save_run_state(run_dir / "run-state", state)

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window._run_dir_edit.setText(str(run_dir))
        window._refresh_status()
        assert window._status_labels["current_epoch"].text() == "5"
        assert window._status_labels["latest_val_cer"].text() == "0.25"
        assert window._status_labels["best_val_cer"].text() == "0.22"
    finally:
        window.close()
