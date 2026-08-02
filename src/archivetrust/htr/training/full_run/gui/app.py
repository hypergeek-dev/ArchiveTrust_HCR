"""PySide6 GUI for preparing, starting, monitoring, stopping, and resuming a full-corpus training
run -- without opening VS Code. Requires the optional `gui` extra (`pip install -e ".[gui]"`).

Run with `python -m archivetrust.htr.training.full_run gui`.

**Read-only monitoring, subprocess-driven control.** Every action button launches a real
`python -m archivetrust.htr.training.full_run <subcommand>` child process via `QProcess` -- this
window never imports or calls `orchestrator.py`/`training_session.py` directly, so a GUI crash or
close can never leave a training container in an inconsistent state the GUI itself was holding open.
The live status panel polls `run_state.json`/the monitoring config on a `QTimer`, never by parsing
the log pane's text (WP6's "do not rely solely on parsing console text for monitoring").
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from archivetrust.htr.training.full_run.gui.commands import (
    build_analyze_pilot_command,
    build_prepare_command,
    build_preflight_command,
    build_resume_command,
    build_start_command,
    build_stop_command,
)
from archivetrust.htr.training.full_run.monitoring_config import load_monitoring_config
from archivetrust.htr.training.full_run.monitoring_state import classify_monitoring_state
from archivetrust.htr.training.full_run.run_state import display_status, load_run_state

STATUS_POLL_INTERVAL_MS = 3000


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Loghi Full-Corpus Training -- Operator Console")
        self.resize(1000, 800)

        self._process: QProcess | None = None
        self._monitoring_config = None

        root = QWidget()
        layout = QVBoxLayout(root)

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_actions_group())
        layout.addWidget(QLabel("Command about to run:"))
        self._command_preview = QLineEdit()
        self._command_preview.setReadOnly(True)
        layout.addWidget(self._command_preview)

        tabs = QTabWidget()
        tabs.addTab(self._build_status_tab(), "Live status")
        tabs.addTab(self._build_log_tab(), "Log")
        layout.addWidget(tabs)

        self.setCentralWidget(root)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(STATUS_POLL_INTERVAL_MS)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    # -- Layout construction ------------------------------------------------------------------

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Paths")
        form = QFormLayout(group)

        self._training_root_edit = QLineEdit(str(Path("training").resolve()))
        form.addRow("Training root:", self._browse_row(self._training_root_edit, directory=True))

        self._pilot_run_edit = QLineEdit(str((Path("training") / "loghi-swedish-v1").resolve()))
        form.addRow("Pilot run:", self._browse_row(self._pilot_run_edit, directory=True))

        self._monitoring_config_edit = QLineEdit(
            str((Path("training") / "config" / "pilot_derived_monitoring.json").resolve())
        )
        form.addRow("Monitoring config:", self._browse_row(self._monitoring_config_edit, directory=False))

        self._run_dir_edit = QLineEdit()
        form.addRow("Run directory:", self._browse_row(self._run_dir_edit, directory=True))

        return group

    def _browse_row(self, line_edit: QLineEdit, *, directory: bool) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        button = QPushButton("Browse...")

        def _browse() -> None:
            if directory:
                path = QFileDialog.getExistingDirectory(self, "Select directory", line_edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Select file", line_edit.text())
            if path:
                line_edit.setText(path)

        button.clicked.connect(_browse)
        layout.addWidget(button)
        return row

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("Actions")
        layout = QHBoxLayout(group)

        analyze_btn = QPushButton("Analyze pilot")
        analyze_btn.clicked.connect(self._on_analyze_pilot)
        layout.addWidget(analyze_btn)

        preflight_btn = QPushButton("Run preflight")
        preflight_btn.clicked.connect(self._on_preflight)
        layout.addWidget(preflight_btn)

        self._run_name_edit = QLineEdit()
        self._run_name_edit.setPlaceholderText("run name (blank = auto-generated)")
        layout.addWidget(self._run_name_edit)

        prepare_btn = QPushButton("Prepare fresh run")
        prepare_btn.clicked.connect(self._on_prepare)
        layout.addWidget(prepare_btn)

        self._hours_spin = QDoubleSpinBox()
        self._hours_spin.setRange(0.1, 168.0)
        self._hours_spin.setValue(5.0)
        self._hours_spin.setSuffix(" h")
        layout.addWidget(self._hours_spin)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 256)
        self._batch_size_spin.setValue(16)
        layout.addWidget(self._batch_size_spin)

        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self._on_start)
        layout.addWidget(start_btn)

        resume_btn = QPushButton("Resume")
        resume_btn.clicked.connect(self._on_resume)
        layout.addWidget(resume_btn)

        stop_btn = QPushButton("Stop (graceful)")
        stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(stop_btn)

        self._force_kill_btn = QPushButton("Force kill")
        self._force_kill_btn.setEnabled(False)
        self._force_kill_btn.clicked.connect(self._on_force_kill)
        layout.addWidget(self._force_kill_btn)

        open_dir_btn = QPushButton("Open run directory")
        open_dir_btn.clicked.connect(self._on_open_run_directory)
        layout.addWidget(open_dir_btn)

        return group

    def _build_status_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        self._status_labels: dict[str, QLabel] = {}
        for key, label in [
            ("status", "Run status:"), ("monitoring_state", "Monitoring state:"),
            ("epoch_position", "Epoch position:"), ("global_shards", "Shards completed:"),
            ("current_global_step", "Global step:"),
            ("latest_val_cer", "Latest val CER:"), ("best_val_cer", "Best val CER:"),
            ("latest_train_loss", "Latest train loss:"), ("latest_val_loss", "Latest val loss:"),
            ("epochs_since_improvement", "Epochs since improvement:"),
            ("elapsed", "Cumulative training time:"), ("last_heartbeat", "Last heartbeat:"),
            ("latest_checkpoint", "Latest checkpoint:"), ("best_checkpoint", "Best checkpoint:"),
            ("resumable", "Resumable:"), ("stop_reason", "Stop reason:"),
        ]:
            value_label = QLabel("--")
            self._status_labels[key] = value_label
            form.addRow(label, value_label)
        return widget

    def _build_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(self.font())
        layout.addWidget(self._log)
        return widget

    # -- Process management ---------------------------------------------------------------------

    def _launch(self, argv: list[str]) -> None:
        self._command_preview.setText(" ".join(argv))
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Busy", "A command is already running -- wait for it to finish.")
            return

        process = QProcess(self)
        process.setProgram(argv[0])
        process.setArguments(argv[1:])
        process.readyReadStandardOutput.connect(lambda: self._append_log(process.readAllStandardOutput()))
        process.readyReadStandardError.connect(lambda: self._append_log(process.readAllStandardError()))
        process.finished.connect(lambda code, status: self._on_process_finished(code))
        self._process = process
        self._force_kill_btn.setEnabled(True)
        self._log.appendPlainText(f"$ {' '.join(argv)}")
        process.start()

    def _append_log(self, data) -> None:
        text = bytes(data).decode("utf-8", errors="replace")
        self._log.appendPlainText(text.rstrip("\n"))

    def _on_process_finished(self, exit_code: int) -> None:
        self._log.appendPlainText(f"[process exited with code {exit_code}]")
        self._force_kill_btn.setEnabled(False)

    # -- Button handlers ---------------------------------------------------------------------

    def _on_analyze_pilot(self) -> None:
        argv = build_analyze_pilot_command(pilot_run=self._pilot_run_edit.text())
        self._launch(argv)

    def _on_preflight(self) -> None:
        argv = build_preflight_command(
            config_path=self._monitoring_config_edit.text(), run_dir=self._run_dir_edit.text() or None,
            batch_size=self._batch_size_spin.value(),
        )
        self._launch(argv)

    def _on_prepare(self) -> None:
        argv = build_prepare_command(
            config_path=self._monitoring_config_edit.text(), run_name=self._run_name_edit.text() or None,
        )
        self._launch(argv)

    def _on_start(self) -> None:
        """Deliberately does not build or launch a real `start` command. Real full-corpus training
        must only ever begin from an operator explicitly typing `--confirm-full-corpus-run` on the
        real CLI -- this button exists so the operator can see the exact command to copy, never so a
        single click can launch it. `commands.build_start_command` never includes the confirmation
        flag, so even a modified build here could never succeed against the launch guard; this handler
        additionally never invokes it at all, so there is no click path from this window to a running
        training container."""
        if not self._run_dir_edit.text():
            QMessageBox.warning(self, "Missing run directory", "Select a prepared run directory first.")
            return
        argv = build_start_command(
            run_dir=self._run_dir_edit.text(), hours=self._hours_spin.value(), batch_size=self._batch_size_spin.value(),
        )
        self._command_preview.setText(" ".join(argv) + " --confirm-full-corpus-run")
        QMessageBox.information(
            self, "Not launched from here",
            "This GUI never starts real full-corpus training itself. Copy the command shown above, "
            "add --confirm-full-corpus-run, and run it manually in a terminal when you are ready.",
        )

    def _on_resume(self) -> None:
        """See `_on_start` -- resuming also performs real optimizer steps, so it is gated the same
        way: shown, never launched, from this window."""
        if not self._run_dir_edit.text():
            QMessageBox.warning(self, "Missing run directory", "Select a run directory first.")
            return
        argv = build_resume_command(
            run_dir=self._run_dir_edit.text(), hours=self._hours_spin.value(), batch_size=self._batch_size_spin.value(),
        )
        self._command_preview.setText(" ".join(argv) + " --confirm-full-corpus-run")
        QMessageBox.information(
            self, "Not launched from here",
            "This GUI never resumes real full-corpus training itself. Copy the command shown above, "
            "add --confirm-full-corpus-run, and run it manually in a terminal when you are ready.",
        )

    def _on_stop(self) -> None:
        """Graceful stop -- writes the STOP_REQUESTED sentinel via a quick, separate `stop`
        invocation. Never kills the running process directly; that is `_on_force_kill`'s job,
        gated behind its own explicit second action."""
        if not self._run_dir_edit.text():
            return
        import subprocess

        argv = build_stop_command(run_dir=self._run_dir_edit.text())
        subprocess.run(argv, capture_output=True, timeout=30, check=False)
        self._log.appendPlainText(f"$ {' '.join(argv)}")
        self._log.appendPlainText("[graceful stop requested -- the run will stop after the current shard]")

    def _on_force_kill(self) -> None:
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            return
        confirmed = QMessageBox.question(
            self, "Force kill?",
            "This immediately terminates the training subprocess without waiting for the current "
            "shard to finish. The last verified checkpoint is preserved, but this session's progress "
            "since then is lost. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirmed == QMessageBox.StandardButton.Yes:
            self._process.kill()
            self._log.appendPlainText("[force-killed by operator]")

    def _on_open_run_directory(self) -> None:
        path = self._run_dir_edit.text()
        if not path:
            return
        if sys.platform == "win32":
            import os

            os.startfile(path)  # noqa: S606 -- operator-selected local path, not user-controlled input
        else:
            import subprocess

            subprocess.run(["xdg-open", path], check=False)

    # -- Status polling ---------------------------------------------------------------------

    def _refresh_status(self) -> None:
        run_dir = self._run_dir_edit.text()
        if not run_dir:
            return
        state = load_run_state(Path(run_dir) / "run-state")
        if state is None:
            return

        labels = self._status_labels
        labels["status"].setText(display_status(state))
        labels["epoch_position"].setText(
            f"epochs_completed={state.epochs_completed} "
            f"| shard {state.shards_completed_in_current_epoch}/{state.shards_per_epoch or '?'} "
            f"| {state.epoch_progress:.1%} of epoch")
        labels["global_shards"].setText(str(state.global_shards_completed or state.current_epoch))
        labels["current_global_step"].setText(str(state.current_global_step))
        labels["latest_val_cer"].setText(str(state.latest_metrics.get("val_cer")))
        labels["best_val_cer"].setText(str(state.best_metrics.get("val_cer")))
        labels["latest_train_loss"].setText(str(state.latest_metrics.get("train_loss")))
        labels["latest_val_loss"].setText(str(state.latest_metrics.get("val_loss")))
        labels["epochs_since_improvement"].setText(str(state.epochs_since_improvement))
        labels["last_heartbeat"].setText(state.last_heartbeat_at or "--")
        labels["latest_checkpoint"].setText(state.latest_checkpoint or "--")
        labels["best_checkpoint"].setText(state.best_checkpoint or "--")
        labels["resumable"].setText(str(state.latest_checkpoint is not None))
        labels["stop_reason"].setText(state.stop_reason or "--")

        config_path = Path(run_dir) / "config" / "full_run_monitoring.json"
        if config_path.exists():
            try:
                config = load_monitoring_config(config_path)
                labels["monitoring_state"].setText(classify_monitoring_state(run_state=state, monitoring_config=config))
            except Exception:  # noqa: BLE001 -- a stale/partial config must not break the status panel
                labels["monitoring_state"].setText("unknown")


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
