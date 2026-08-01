"""Read-only training dashboard ViewModel (docs/methods/loghi-training-dashboard.md).

Framework-independent, like every other ViewModel in this package (`htr_methods_viewmodel.py`'s
convention) -- zero Streamlit import, so every method here is directly unit-testable and the actual
Streamlit script (`scripts/train_loghi_swedish_dashboard.py`) contains only widget-rendering calls
into this class. Discovers runs by walking `training_root` for `*/run-state/training_identity.json`
-- never assumes there is exactly one run, so a future full-corpus run appears here automatically once
it exists.

**Never controls a training process.** Every method here only reads files
(`run_status.py`/`run_health.py`/`checkpoint_index.py`/`training_session.py`'s own loaders) -- there
is no method on this class that could start, stop, or otherwise touch a training run.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.checkpoint_index import load_index
from archivetrust.htr.training.incremental_jsonl_reader import JsonlReadResult, read_new_lines
from archivetrust.htr.training.run_health import HealthAssessment, compute_health
from archivetrust.htr.training.run_profile import (
    full_corpus_profile,
    pilot_profile,
    cap_shorter_than_one_epoch_warning,
)
from archivetrust.htr.training.run_status import RunStatus, compute_run_status
from archivetrust.htr.training.training_session import load_session_state

_DEFAULT_TRAINING_ROOT = Path(__file__).resolve().parents[3] / "training"

_PROFILE_BY_NAME = {"pilot": pilot_profile, "full_corpus": lambda: full_corpus_profile(data_prepared=True)}


class RunSummaryRow(BaseModel):
    """One row in the run list / comparison table (brief §9 "Run comparison")."""

    model_config = ConfigDict(frozen=True)

    run_dir: str
    run_id: str
    run_profile: str
    status: str
    health_level: str
    dataset_name: str
    training_line_count: int | None
    base_model: str
    batch_size: int | None
    learning_rate_policy: str | None = None
    max_epochs: int | None
    best_val_cer: float | None
    best_epoch: int | None
    cumulative_epoch: int
    cumulative_training_seconds: float
    stop_reason: str | None
    best_checkpoint_path: str | None
    test_cer: str | None = None
    """Always `None` this phase -- the reserved test set is never evaluated by this dashboard (the
    original brief's execution boundary). Present as a field, not silently omitted, so a comparison
    table column exists and visibly reads "not evaluated" rather than looking like the column was
    forgotten."""


class RunDetailView(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: RunStatus
    health: HealthAssessment
    validation_history: tuple[dict, ...]
    cap_warning: str | None


class ChartSeries(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: tuple[float, ...]
    y: tuple[float | None, ...]
    label: str


class ValCerChartSeries(BaseModel):
    model_config = ConfigDict(frozen=True)

    epochs: tuple[int, ...]
    val_cer: tuple[float | None, ...]
    best_epoch: int | None
    best_val_cer: float | None
    early_stopping_patience: int | None
    epochs_since_improvement: int


class TrainingDashboardViewModel:
    def __init__(self, *, training_root: str | Path | None = None) -> None:
        self._training_root = Path(training_root) if training_root is not None else _DEFAULT_TRAINING_ROOT

    def discover_run_dirs(self) -> tuple[Path, ...]:
        if not self._training_root.exists():
            return ()
        found = []
        for candidate in sorted(self._training_root.iterdir()):
            if not candidate.is_dir():
                continue
            if (candidate / "run-state" / "training_identity.json").exists():
                found.append(candidate)
        return tuple(found)

    def run_status(self, run_dir: str | Path) -> RunStatus:
        return compute_run_status(run_dir)

    def run_health(self, run_dir: str | Path) -> HealthAssessment:
        run_dir = Path(run_dir)
        status = compute_run_status(run_dir)
        state = load_session_state(run_dir / "run-state")
        validation_history = state.validation_history if state else ()
        entries = load_index(run_dir / "run-state" / "checkpoint_index.json")

        cap_warning = None
        if status.wall_clock_safety_cap_hours is not None:
            profile_factory = _PROFILE_BY_NAME.get(status.run_profile)
            if profile_factory is not None:
                cap_warning = cap_shorter_than_one_epoch_warning(
                    profile=profile_factory(), wall_clock_hours=status.wall_clock_safety_cap_hours
                )

        return compute_health(
            status=status,
            validation_history=validation_history,
            checkpoint_entries=tuple(entries),
            cap_warning=cap_warning,
        )

    def run_detail(self, run_dir: str | Path) -> RunDetailView:
        run_dir = Path(run_dir)
        status = compute_run_status(run_dir)
        health = self.run_health(run_dir)
        state = load_session_state(run_dir / "run-state")
        validation_history = state.validation_history if state else ()

        cap_warning = None
        if status.wall_clock_safety_cap_hours is not None:
            profile_factory = _PROFILE_BY_NAME.get(status.run_profile)
            if profile_factory is not None:
                cap_warning = cap_shorter_than_one_epoch_warning(
                    profile=profile_factory(), wall_clock_hours=status.wall_clock_safety_cap_hours
                )

        return RunDetailView(
            status=status, health=health, validation_history=validation_history, cap_warning=cap_warning
        )

    def run_summary_rows(self) -> tuple[RunSummaryRow, ...]:
        rows = []
        for run_dir in self.discover_run_dirs():
            status = compute_run_status(run_dir)
            health = self.run_health(run_dir)
            rows.append(
                RunSummaryRow(
                    run_dir=str(run_dir),
                    run_id=status.run_id,
                    run_profile=status.run_profile,
                    status=status.status,
                    health_level=health.level,
                    dataset_name=status.dataset_name,
                    training_line_count=status.training_line_count,
                    base_model=status.base_model,
                    batch_size=status.batch_size,
                    max_epochs=status.max_epochs,
                    best_val_cer=status.best_val_cer,
                    best_epoch=status.best_epoch,
                    cumulative_epoch=status.cumulative_epoch,
                    cumulative_training_seconds=status.cumulative_training_seconds,
                    stop_reason=status.stop_reason,
                    best_checkpoint_path=status.best_checkpoint_path,
                )
            )
        return tuple(rows)

    def read_new_gpu_samples(self, run_dir: str | Path, *, since_offset: int = 0) -> JsonlReadResult:
        """Thin wrapper for the app layer's own incremental cache (brief §"Performance
        requirements") -- the app keeps `(offset, accumulated_rows)` across refreshes (e.g. in
        Streamlit `session_state`) and passes the accumulated rows into `chart_series` below; this
        ViewModel itself holds no cross-call state."""
        path = Path(run_dir) / "run-state" / "telemetry" / "gpu_samples.jsonl"
        return read_new_lines(path, since_offset=since_offset)

    def epoch_metric_series(self, run_dir: str | Path, *, metric: str, label: str) -> ChartSeries:
        """`metric` in `{"train_loss", "val_loss", "duration_seconds"}` (or any key
        `validation_history` entries carry) -- x is epoch number."""
        state = load_session_state(Path(run_dir) / "run-state")
        history = state.validation_history if state else ()
        x = tuple(float(entry["epoch"]) for entry in history)
        y = tuple(entry.get(metric) for entry in history)
        return ChartSeries(x=x, y=y, label=label)

    def throughput_series(self, run_dir: str | Path, *, lines_per_epoch: int | None) -> ChartSeries:
        """Lines processed per second, per epoch -- `lines_per_epoch / duration_seconds`. `None`
        where `lines_per_epoch` is not known (never a fabricated throughput)."""
        state = load_session_state(Path(run_dir) / "run-state")
        history = state.validation_history if state else ()
        x = tuple(float(entry["epoch"]) for entry in history)
        y = tuple(
            (lines_per_epoch / entry["duration_seconds"])
            if lines_per_epoch and entry.get("duration_seconds")
            else None
            for entry in history
        )
        return ChartSeries(x=x, y=y, label="Lines/sec")

    def val_cer_series(self, run_dir: str | Path) -> ValCerChartSeries:
        run_dir = Path(run_dir)
        state = load_session_state(run_dir / "run-state")
        status = compute_run_status(run_dir)
        history = state.validation_history if state else ()
        epochs = tuple(int(entry["epoch"]) for entry in history)
        val_cer = tuple(entry.get("val_cer") for entry in history)
        return ValCerChartSeries(
            epochs=epochs, val_cer=val_cer, best_epoch=status.best_epoch, best_val_cer=status.best_val_cer,
            early_stopping_patience=status.early_stopping_patience,
            epochs_since_improvement=status.epochs_since_improvement,
        )

    def gpu_utilization_series(self, gpu_samples: tuple[dict, ...]) -> ChartSeries:
        return self._gpu_series(gpu_samples, path=("gpu", "utilization_pct"), label="GPU utilization %")

    def vram_series(self, gpu_samples: tuple[dict, ...]) -> ChartSeries:
        return self._gpu_series(gpu_samples, path=("gpu", "memory_used_mb"), label="VRAM used (MB)")

    def gpu_temperature_series(self, gpu_samples: tuple[dict, ...]) -> ChartSeries:
        return self._gpu_series(gpu_samples, path=("gpu", "temperature_c"), label="GPU temperature (C)")

    @staticmethod
    def _gpu_series(gpu_samples: tuple[dict, ...], *, path: tuple[str, str], label: str) -> ChartSeries:
        x = tuple(float(i) for i in range(len(gpu_samples)))
        outer, inner = path
        y = tuple(sample.get(outer, {}).get(inner) for sample in gpu_samples)
        return ChartSeries(x=x, y=y, label=label)
