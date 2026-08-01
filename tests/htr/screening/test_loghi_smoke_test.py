"""`htr/screening/loghi_smoke_test.py` -- proves the smoke-test mechanics end to end against a fake
facade: real `LoghiAdapter.recognize()` calls, real telemetry recorded, real report produced. No
Docker/subprocess is invoked."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.screening.loghi_smoke_test import SmokeTestPageRef, run_loghi_smoke_test
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.providers.loghi.adapter import LoghiAdapter
from archivetrust.providers.loghi.facade import FakeLoghiWorkerFacade
from archivetrust.providers.loghi.models import (
    LoghiComponentVersions,
    LoghiEnvironmentReport,
    LoghiExecutionMode,
)
from archivetrust.providers.loghi.stage_results import LoghiPipelineResult, LoghiStageResult

_REAL_PINS = LoghiComponentVersions(
    loghi_repo_commit="a" * 40,
    submodule_commits={},
    laypa_commit="b" * 40,
    loghi_tooling_commit="c" * 40,
    loghi_htr_commit="d" * 40,
    model_checkpoint_id="dutch-v1",
    model_checkpoint_hash="e" * 64,
    docker_image_tag="ghcr.io/knaw-huc/loghi:v2",
    docker_image_digest="sha256:" + "f" * 64,
    inference_script_version="1.0",
    beam_width=1,
    reading_order_settings="default",
    language_detection_settings="auto",
    gpu_selection="none",
    container_runtime_version="29.6.1",
)

_PAGE_XML = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageWidth="10" imageHeight="10">
    <TextRegion id="r1"><TextLine id="l1">
      <TextEquiv conf="0.8"><Unicode>smoke test text</Unicode></TextEquiv>
    </TextLine></TextRegion>
  </Page>
</PcGts>"""


@pytest.fixture()
def env_report() -> LoghiEnvironmentReport:
    return LoghiEnvironmentReport(
        host_os="Linux", docker_cli_present=True, docker_version="x", wsl_present=False,
        wsl_distros=(), linux_distribution="Ubuntu", nvidia_toolkit_version=None,
        cuda_visible_devices=None, execution_mode=LoghiExecutionMode.DOCKER_LINUX,
    )


def _adapter_with_fake_success(monkeypatch, env_report) -> LoghiAdapter:
    from archivetrust.providers.loghi import adapter as adapter_module

    monkeypatch.setattr(adapter_module, "probe_loghi_environment", lambda: env_report)
    facade = FakeLoghiWorkerFacade(
        result=LoghiPipelineResult(
            ok=True,
            stages=(LoghiStageResult(stage_name="loghi_pipeline", started_at="t0", completed_at="t1", ok=True),),
            final_page_xml=_PAGE_XML,
            total_duration_ms=42.0,
        )
    )
    return LoghiAdapter(component_versions=_REAL_PINS, facade=facade)


def test_smoke_test_reports_success_for_every_page_against_fake_facade(monkeypatch, env_report) -> None:
    adapter = _adapter_with_fake_success(monkeypatch, env_report)
    swedish = (SmokeTestPageRef(page_id="sv1", image_path="/tmp/sv1.png", corpus_language="sv"),)
    dutch = (SmokeTestPageRef(page_id="nl1", image_path="/tmp/nl1.jpg", corpus_language="nl"),)

    report = run_loghi_smoke_test(adapter=adapter, swedish_pages=swedish, dutch_pages=dutch, store=None)

    assert report.environment_valid is True
    assert report.succeeded_count() == 2
    assert report.failed_count() == 0
    assert report.success_rate_for("sv") == 1.0
    assert report.success_rate_for("nl") == 1.0
    assert report.success_rate_for("de") is None  # empty pool, never a fabricated 0.0


def test_smoke_test_records_real_telemetry_when_store_supplied(monkeypatch, env_report) -> None:
    adapter = _adapter_with_fake_success(monkeypatch, env_report)
    with tempfile.TemporaryDirectory() as d:
        sink = FileTelemetrySink(Path(d) / "events.jsonl")
        store = DurableHtrResearchStore(sink)
        run_loghi_smoke_test(
            adapter=adapter,
            swedish_pages=(SmokeTestPageRef(page_id="p1", image_path="/tmp/p1.png", corpus_language="sv"),),
            dutch_pages=(),
            store=store,
        )
        events = sink.all_events()
        kinds = {e.kind.value for e in events}
        assert "LoghiEnvironmentValidated" in kinds
        assert "LoghiPipelineStarted" in kinds
        assert "LoghiStageCompleted" in kinds
        assert "LoghiPageXmlGenerated" in kinds


def test_smoke_test_never_fabricates_lion_execution_events(monkeypatch, env_report) -> None:
    """This module only ever runs Loghi -- it must never emit a Lion-related event, fabricated or
    otherwise."""
    adapter = _adapter_with_fake_success(monkeypatch, env_report)
    with tempfile.TemporaryDirectory() as d:
        sink = FileTelemetrySink(Path(d) / "events.jsonl")
        store = DurableHtrResearchStore(sink)
        run_loghi_smoke_test(
            adapter=adapter,
            swedish_pages=(SmokeTestPageRef(page_id="p1", image_path="/tmp/p1.png", corpus_language="sv"),),
            dutch_pages=(),
            store=store,
        )
        for event in sink.all_events():
            assert "lion" not in event.kind.value.lower()


def test_smoke_test_reports_report_reconstruction_without_rerunning(monkeypatch, env_report) -> None:
    """Brief: "Demonstrate that completed Loghi results can be reconstructed and reported without
    rerunning the containers." The report is a plain, serializable model -- round-trips through JSON
    with no adapter/facade needed to read it back."""
    adapter = _adapter_with_fake_success(monkeypatch, env_report)
    report = run_loghi_smoke_test(
        adapter=adapter,
        swedish_pages=(SmokeTestPageRef(page_id="p1", image_path="/tmp/p1.png", corpus_language="sv"),),
        dutch_pages=(),
        store=None,
    )
    from archivetrust.htr.screening.loghi_smoke_test import LoghiSmokeTestReport

    reconstructed = LoghiSmokeTestReport.model_validate_json(report.model_dump_json())
    assert reconstructed == report


def test_smoke_test_classifies_failures_never_excludes_after_one_bad_page(monkeypatch, env_report) -> None:
    """A failing page must still produce a full outcome, and other pages must still be attempted."""
    from archivetrust.providers.loghi import adapter as adapter_module

    monkeypatch.setattr(adapter_module, "probe_loghi_environment", lambda: env_report)
    failing_facade = FakeLoghiWorkerFacade(
        result=LoghiPipelineResult(
            ok=False,
            stages=(
                LoghiStageResult(
                    stage_name="laypa", started_at="t0", completed_at="t1", ok=False,
                    errors=("layout analysis crashed",),
                ),
            ),
        )
    )
    adapter = LoghiAdapter(component_versions=_REAL_PINS, facade=failing_facade)
    report = run_loghi_smoke_test(
        adapter=adapter,
        swedish_pages=(
            SmokeTestPageRef(page_id="p1", image_path="/tmp/p1.png", corpus_language="sv"),
            SmokeTestPageRef(page_id="p2", image_path="/tmp/p2.png", corpus_language="sv"),
        ),
        dutch_pages=(),
        store=None,
    )
    assert report.failed_count() == 2
    assert len(report.page_outcomes) == 2  # both attempted, neither skipped
    assert report.failure_categories()  # classified, not silently dropped
