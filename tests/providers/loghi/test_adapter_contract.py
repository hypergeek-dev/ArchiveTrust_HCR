"""Fast Loghi adapter tests -- protocol conformance, capability honesty, and failure-path unit tests,
all using `FakeLoghiWorkerFacade`. No Docker, no subprocess, no GPU required -- these run in the
default `pytest tests -q` sweep.
"""

from __future__ import annotations

from archivetrust.providers.htr_adapter import HtrMethodAdapter, RecognitionInput
from archivetrust.providers.loghi.adapter import (
    LoghiAdapter,
    build_evidence,
    build_failure_record,
    build_observation_payloads,
    normalize_transcription,
)
from archivetrust.providers.loghi.facade import FakeLoghiWorkerFacade
from archivetrust.providers.loghi.models import LoghiComponentVersions
from archivetrust.providers.loghi.stage_results import LoghiPipelineResult, LoghiStageResult

_VALID_PAGE_XML = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Metadata><Creator>loghi-htr v1.2</Creator></Metadata>
  <Page imageWidth="100" imageHeight="200">
    <TextRegion id="r1">
      <TextLine id="l1">
        <TextEquiv conf="0.87"><Unicode>hello world</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>"""

_REAL_PINS = LoghiComponentVersions(
    loghi_repo_commit="a" * 40,
    submodule_commits={"laypa": "b" * 40, "loghi-htr": "c" * 40, "loghi-tooling": "d" * 40},
    laypa_commit="b" * 40,
    loghi_tooling_commit="d" * 40,
    loghi_htr_commit="c" * 40,
    model_checkpoint_id="dutch-generic-v1",
    model_checkpoint_hash="e" * 64,
    docker_image_tag="ghcr.io/knaw-huc/loghi:v2.1",
    docker_image_digest="sha256:" + "f" * 64,
    inference_script_version="1.0.0",
    beam_width=3,
    reading_order_settings="default",
    language_detection_settings="nl",
    gpu_selection="all",
    container_runtime_version="29.6.1",
)


def _successful_pipeline_result(page_xml: str = _VALID_PAGE_XML) -> LoghiPipelineResult:
    return LoghiPipelineResult(
        ok=True,
        stages=(
            LoghiStageResult(stage_name="loghi_pipeline", started_at="t0", completed_at="t1", ok=True),
        ),
        final_page_xml=page_xml,
        total_duration_ms=456.0,
    )


def test_implements_htr_method_adapter_protocol() -> None:
    assert isinstance(LoghiAdapter(), HtrMethodAdapter)


def test_metadata_and_capabilities_are_honest() -> None:
    adapter = LoghiAdapter(component_versions=_REAL_PINS)
    metadata = adapter.get_metadata()
    assert metadata.method_id == "loghi"
    assert "UNPINNED" not in metadata.model_revision

    capabilities = adapter.get_capabilities()
    assert capabilities.page_level_supported is True
    assert capabilities.line_level_supported is False, "internal line-cutting must not earn this flag"
    assert capabilities.local_execution_supported is True
    assert capabilities.external_upload_required is False
    assert capabilities.image_color_normalization_required is True
    assert capabilities.geometry_supported is True
    assert capabilities.confidence_supported is True


def test_validate_environment_refuses_placeholder_pins() -> None:
    """`CURRENT_PINNED_VERSIONS` is real now (2026-08-01 Swedish fine-tuning environment install) --
    this test constructs an explicit placeholder instance rather than relying on `LoghiAdapter()`'s
    default, which would otherwise silently stop testing the refusal path it exists to cover."""
    from archivetrust.providers.loghi.pinned_versions import PLACEHOLDER_SENTINEL

    placeholder_pins = _REAL_PINS.model_copy(update={"loghi_repo_commit": PLACEHOLDER_SENTINEL})
    adapter = LoghiAdapter(component_versions=placeholder_pins)
    validation = adapter.validate_environment()
    assert validation.valid is False
    assert any("UNPINNED" in m or "placeholder" in m for m in validation.messages)


def test_validate_environment_accepts_the_real_pinned_versions() -> None:
    """The counterpart -- proves `CURRENT_PINNED_VERSIONS` itself is no longer a placeholder and that
    `validate_environment()` reports it as such (Docker/WSL2 execution mode permitting, which this
    machine has -- see `docs/methods/loghi-swedish-finetuning.md`)."""
    from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

    assert CURRENT_PINNED_VERSIONS.is_placeholder() is False
    adapter = LoghiAdapter()
    validation = adapter.validate_environment()
    assert not any("UNPINNED" in m or "placeholder" in m for m in validation.messages)


def test_recognize_rejects_missing_image_reference() -> None:
    adapter = LoghiAdapter(component_versions=_REAL_PINS, facade=FakeLoghiWorkerFacade(result=_successful_pipeline_result()))
    result = adapter.recognize(RecognitionInput())
    assert result.text is None
    assert result.raw_response["category"] == "malformed_input"


def test_recognize_succeeds_with_real_pins_and_fake_facade(monkeypatch) -> None:
    from archivetrust.providers.loghi import adapter as adapter_module
    from archivetrust.providers.loghi.models import LoghiEnvironmentReport, LoghiExecutionMode

    monkeypatch.setattr(
        adapter_module,
        "probe_loghi_environment",
        lambda: LoghiEnvironmentReport(
            host_os="Windows",
            docker_cli_present=True,
            docker_version="Docker version 29.6.1",
            wsl_present=True,
            wsl_distros=("docker-desktop",),
            linux_distribution="Docker Desktop",
            nvidia_toolkit_version=None,
            cuda_visible_devices=None,
            execution_mode=LoghiExecutionMode.DOCKER_WSL2,
        ),
    )
    facade = FakeLoghiWorkerFacade(result=_successful_pipeline_result())
    adapter = LoghiAdapter(component_versions=_REAL_PINS, facade=facade)
    result = adapter.recognize(RecognitionInput(page_image_ref="/tmp/page.jpg"))

    assert result.text == "hello world"
    assert result.confidence == 0.87
    assert len(facade.calls) == 1
    assert facade.calls[0]["input_image_path"] == "/tmp/page.jpg"
    assert facade.calls[0]["component_versions"] is _REAL_PINS


def test_recognize_reports_empty_output_honestly(monkeypatch) -> None:
    from archivetrust.providers.loghi import adapter as adapter_module
    from archivetrust.providers.loghi.models import LoghiEnvironmentReport, LoghiExecutionMode

    monkeypatch.setattr(
        adapter_module,
        "probe_loghi_environment",
        lambda: LoghiEnvironmentReport(
            host_os="Linux",
            docker_cli_present=True,
            docker_version="x",
            wsl_present=False,
            wsl_distros=(),
            linux_distribution="Ubuntu",
            nvidia_toolkit_version=None,
            cuda_visible_devices=None,
            execution_mode=LoghiExecutionMode.DOCKER_LINUX,
        ),
    )
    empty_xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageWidth="10" imageHeight="10"></Page>
</PcGts>"""
    facade = FakeLoghiWorkerFacade(result=_successful_pipeline_result(empty_xml))
    adapter = LoghiAdapter(component_versions=_REAL_PINS, facade=facade)
    result = adapter.recognize(RecognitionInput(page_image_ref="/tmp/blank.jpg"))
    assert result.text is None
    assert result.raw_response["category"] == "empty_output"


def test_build_evidence_and_failure_record_round_trip(monkeypatch) -> None:
    from archivetrust.providers.loghi import adapter as adapter_module
    from archivetrust.providers.loghi.models import LoghiEnvironmentReport, LoghiExecutionMode

    monkeypatch.setattr(
        adapter_module,
        "probe_loghi_environment",
        lambda: LoghiEnvironmentReport(
            host_os="Linux", docker_cli_present=True, docker_version="x", wsl_present=False,
            wsl_distros=(), linux_distribution="Ubuntu", nvidia_toolkit_version=None,
            cuda_visible_devices=None, execution_mode=LoghiExecutionMode.DOCKER_LINUX,
        ),
    )
    facade = FakeLoghiWorkerFacade(result=_successful_pipeline_result())
    adapter = LoghiAdapter(component_versions=_REAL_PINS, facade=facade)
    result = adapter.recognize(RecognitionInput(page_image_ref="/tmp/page.jpg"))

    evidence = build_evidence(result)
    assert evidence is not None
    assert evidence.raw_output == _VALID_PAGE_XML
    assert build_failure_record(result, method_run_id="mr_1") is None

    payloads = build_observation_payloads(result)
    assert payloads is not None
    raw, parsed, normalized = payloads
    assert raw.text == parsed.text == "hello world"
    assert normalized.text == normalize_transcription("hello world")


def test_health_check_reflects_placeholder_pins() -> None:
    from archivetrust.providers.loghi.pinned_versions import PLACEHOLDER_SENTINEL

    placeholder_pins = _REAL_PINS.model_copy(update={"loghi_repo_commit": PLACEHOLDER_SENTINEL})
    adapter = LoghiAdapter(component_versions=placeholder_pins)
    health = adapter.health_check()
    assert health.healthy is False
    assert "pins_placeholder=True" in health.message


def test_health_check_reflects_the_real_pinned_versions() -> None:
    adapter = LoghiAdapter()  # default == CURRENT_PINNED_VERSIONS, real since 2026-08-01
    health = adapter.health_check()
    assert "pins_placeholder=False" in health.message
