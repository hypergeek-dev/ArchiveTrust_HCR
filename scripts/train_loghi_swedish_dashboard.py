#!/usr/bin/env python
"""Read-only training dashboard for `loghi_swedish_finetuned_v1`
(docs/methods/loghi-training-dashboard.md).

**Read-only, by construction.** Every widget on this page renders data returned by
`presentation/training_dashboard_viewmodel.py::TrainingDashboardViewModel` -- there is no button,
form, or code path anywhere in this file that starts, stops, pauses, or resumes a training process.
The trainer (`scripts/train_loghi_swedish.py`) and this dashboard read and write completely
independent files; killing or restarting either one never affects the other.

Run (after `pip install -e .[dashboard]`):

    streamlit run scripts/train_loghi_swedish_dashboard.py

Auto-refreshes every 5-15s (configurable in the sidebar) via `time.sleep` + `st.rerun()` -- the
standard no-extra-dependency Streamlit polling pattern, deliberately not using `st_autorefresh` or
any JS-timer library.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from archivetrust.presentation.training_dashboard_viewmodel import (  # noqa: E402
    TrainingDashboardViewModel,
)

st.set_page_config(page_title="Loghi Swedish Fine-Tuning Dashboard", layout="wide")

_STOP_REASON_MEANINGS = {
    "time_budget_reached": "The session's configured wall-clock cap was reached.",
    "stop_requested": "A safe stop was requested via the STOP_REQUESTED sentinel file.",
    "epoch_would_not_fit": "The next epoch's estimated duration would not fit the remaining budget.",
    "epoch_failed": "The container/process for one epoch failed before producing a verified checkpoint.",
    "target_epochs_reached": "The session's configured epoch cap was reached.",
    "no_val_cer_improvement": "Validation CER did not improve for the configured patience.",
}

_HEALTH_COLOR = {"healthy": "🟢", "completed": "🔵", "warning": "🟡", "stalled": "🟠", "failed": "🔴"}


@st.cache_resource
def _viewmodel() -> TrainingDashboardViewModel:
    return TrainingDashboardViewModel()


def _render_run_header(status) -> None:
    cols = st.columns(6)
    cols[0].metric("Run ID", status.run_id[-12:])
    cols[1].metric("Profile", status.run_profile)
    cols[2].metric("Dataset size", status.training_line_count or "unknown")
    cols[3].metric("Base model", status.base_model.split("@")[0])
    cols[4].metric("Started", status.start_time or "unknown")
    cols[5].metric("Last update", status.last_update_time or "unknown")


def _render_health_summary(health) -> None:
    st.subheader(f"{_HEALTH_COLOR.get(health.level, '⚪')} Health: {health.level}")
    if health.findings:
        for finding in health.findings:
            st.warning(f"**{finding.reason}** -- {finding.message}")
    else:
        st.caption("No warnings.")


def _render_progress(status) -> None:
    cols = st.columns(4)
    cols[0].metric("Epoch", f"{status.cumulative_epoch}" + (f" / {status.max_epochs}" if status.max_epochs else ""))
    cols[1].metric("Cumulative training time", f"{status.cumulative_training_seconds / 60.0:.1f} min")
    cols[2].metric("Sessions completed", status.session_count)
    cols[3].metric("Status", status.status)


def _render_charts(vm: TrainingDashboardViewModel, run_dir: Path) -> None:
    val_series = vm.val_cer_series(run_dir)
    if val_series.epochs:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=val_series.epochs, y=val_series.val_cer, mode="lines+markers", name="val_CER"))
        if val_series.best_epoch is not None and val_series.best_val_cer is not None:
            fig.add_trace(go.Scatter(
                x=[val_series.best_epoch], y=[val_series.best_val_cer], mode="markers",
                marker=dict(size=14, symbol="star", color="gold"), name="Best",
            ))
        if val_series.early_stopping_patience is not None and val_series.epochs:
            fig.add_annotation(
                x=val_series.epochs[-1], y=val_series.val_cer[-1] or 0,
                text=f"epochs since improvement: {val_series.epochs_since_improvement} / patience "
                     f"{val_series.early_stopping_patience}",
                showarrow=True,
            )
        fig.update_layout(title="Validation CER by epoch", xaxis_title="Epoch", yaxis_title="val_CER")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No completed epochs yet -- no validation CER to chart.")

    col1, col2 = st.columns(2)
    with col1:
        loss_series = vm.epoch_metric_series(run_dir, metric="train_loss", label="Train loss")
        if loss_series.x:
            fig = go.Figure(go.Scatter(x=loss_series.x, y=loss_series.y, mode="lines+markers"))
            fig.update_layout(title="Training loss by epoch", xaxis_title="Epoch", yaxis_title="loss")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.caption("Throughput chart requires a known lines-per-epoch count (not tracked this phase).")

    gpu_samples = st.session_state.get(f"gpu_samples::{run_dir}", [])
    if gpu_samples:
        for label, series_fn in (
            ("GPU utilization %", vm.gpu_utilization_series),
            ("VRAM used (MB)", vm.vram_series),
            ("GPU temperature (C)", vm.gpu_temperature_series),
        ):
            series = series_fn(tuple(gpu_samples))
            if any(y is not None for y in series.y):
                fig = go.Figure(go.Scatter(x=series.x, y=series.y, mode="lines"))
                fig.update_layout(title=label, xaxis_title="Sample #")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No GPU telemetry samples yet for this run.")


def _render_early_stopping_panel(status) -> None:
    st.subheader("Early stopping")
    cols = st.columns(4)
    cols[0].metric("Patience", status.early_stopping_patience or "not configured")
    cols[1].metric("Epochs since improvement", status.epochs_since_improvement)
    cols[2].metric("Best val_CER", status.best_val_cer)
    cols[3].metric("Best epoch", status.best_epoch)
    if status.early_stopping_patience is not None:
        armed = status.cumulative_epoch > 0
        st.caption(f"Armed: {armed}. Enough completed epochs to evaluate: {status.cumulative_epoch > 0}.")
    else:
        st.caption("Early stopping was not configured for this session.")


def _render_safety_cap_panel(vm: TrainingDashboardViewModel, run_dir: Path, status, cap_warning: str | None) -> None:
    st.subheader("Wall-clock safety cap")
    cols = st.columns(3)
    cols[0].metric("Configured cap", f"{status.wall_clock_safety_cap_hours:.2f}h" if status.wall_clock_safety_cap_hours else "not configured")
    cols[1].metric("Elapsed (cumulative)", f"{status.cumulative_training_seconds / 3600.0:.2f}h")
    remaining = (
        status.wall_clock_safety_cap_hours - status.cumulative_training_seconds / 3600.0
        if status.wall_clock_safety_cap_hours is not None else None
    )
    cols[2].metric("Remaining", f"{remaining:.2f}h" if remaining is not None else "unknown")
    if cap_warning:
        st.error(cap_warning)
    st.caption("A graceful stop only ever occurs at an epoch/checkpoint boundary -- there is no mid-epoch stop capability by design.")


def _render_checkpoint_panel(status) -> None:
    st.subheader("Checkpoints")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Latest**")
        st.code(status.latest_checkpoint_path or "none yet")
    with cols[1]:
        st.markdown("**Best (val_CER)**")
        st.code(status.best_checkpoint_path or "none yet")
        if status.best_val_cer is not None:
            st.caption(f"val_CER={status.best_val_cer:.4f} at epoch {status.best_epoch}")
    st.metric("Resumable", status.resumable)


def _render_stop_summary(status) -> None:
    if status.status not in ("completed", "failed", "interrupted"):
        return
    st.subheader("Stop summary")
    st.write(f"**Stop reason:** `{status.stop_reason}` -- {_STOP_REASON_MEANINGS.get(status.stop_reason, 'unknown reason')}")
    st.write(f"**Stopped mid-epoch:** {status.stopped_mid_epoch}")
    st.write(f"**Final epoch:** {status.cumulative_epoch}  **Best epoch:** {status.best_epoch}  "
             f"**Best val_CER:** {status.best_val_cer}")
    st.warning(
        "A falling training loss or CER here is evidence the training mechanics work, not a quality "
        "claim. This checkpoint has not been evaluated on the reserved test set or against the "
        "sealed Lion-vs-Loghi benchmark."
    )


def _render_run_detail_tab(vm: TrainingDashboardViewModel) -> None:
    run_dirs = vm.discover_run_dirs()
    if not run_dirs:
        st.info(f"No training runs found. Expected run directories under `{REPO_ROOT / 'training'}`.")
        return

    labels = [d.name for d in run_dirs]
    selected = st.selectbox("Run", labels, index=0)
    run_dir = run_dirs[labels.index(selected)]

    detail = vm.run_detail(run_dir)
    status = detail.status

    cache_key = f"gpu_samples::{run_dir}"
    offset_key = f"gpu_offset::{run_dir}"
    cached = st.session_state.get(cache_key, [])
    since = st.session_state.get(offset_key, 0)
    result = vm.read_new_gpu_samples(run_dir, since_offset=since)
    cached.extend(result.rows)
    st.session_state[cache_key] = cached
    st.session_state[offset_key] = result.new_offset

    _render_run_header(status)
    _render_health_summary(detail.health)
    _render_progress(status)
    _render_charts(vm, run_dir)
    _render_early_stopping_panel(status)
    _render_safety_cap_panel(vm, run_dir, status, detail.cap_warning)
    _render_checkpoint_panel(status)
    _render_stop_summary(status)


def _render_compare_tab(vm: TrainingDashboardViewModel) -> None:
    rows = vm.run_summary_rows()
    if not rows:
        st.info("No runs to compare yet.")
        return
    selected_ids = st.multiselect("Runs to compare", [r.run_id for r in rows], default=[r.run_id for r in rows])
    selected_rows = [r for r in rows if r.run_id in selected_ids]
    st.dataframe(
        [
            {
                "run_id": r.run_id[-12:],
                "profile": r.run_profile,
                "status": r.status,
                "dataset_size": r.training_line_count,
                "base_model": r.base_model.split("@")[0],
                "batch_size": r.batch_size,
                "max_epochs": r.max_epochs,
                "best_val_CER": r.best_val_cer,
                "best_epoch": r.best_epoch,
                "runtime_min": round(r.cumulative_training_seconds / 60.0, 1),
                "stop_reason": r.stop_reason,
                "best_checkpoint": r.best_checkpoint_path,
                "test_CER": r.test_cer,  # always "None" this phase -- never mixed with val_CER above
            }
            for r in selected_rows
        ]
    )
    st.caption(
        "`test_CER` is always empty in this phase -- the reserved test set is never evaluated by "
        "this dashboard, and it is never conflated with `best_val_CER` above."
    )


def main() -> None:
    st.title("Loghi Swedish Fine-Tuning -- Training Dashboard (read-only)")
    vm = _viewmodel()

    with st.sidebar:
        refresh_seconds = st.slider("Refresh interval (seconds)", min_value=5, max_value=15, value=10)
        st.caption("This page never starts, stops, pauses, or resumes training. Use "
                   "`scripts/train_loghi_swedish.py` to control a session.")

    tab_detail, tab_compare = st.tabs(["Run detail", "Compare runs"])
    with tab_detail:
        _render_run_detail_tab(vm)
    with tab_compare:
        _render_compare_tab(vm)

    time.sleep(refresh_seconds)
    st.rerun()


if __name__ == "__main__":
    main()
