from pathlib import Path


ROOT = Path("src/archivetrust")


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_exporters_cannot_traverse_historical_canonical_events_independently() -> None:
    for relative in ("application/export/json_export.py", "application/export/interop.py"):
        source = _source(relative)
        assert "CurrentStateService" in source
        assert ".all_events(" not in source
        assert "CanonicalDocumentCreated" not in source


def test_evaluation_selects_current_canonical_through_authoritative_service() -> None:
    source = _source("evaluation/evaluate.py")
    assert "CurrentStateService" in source
    assert "history[-1]" not in source
    assert "canonical_observation_history(slot)" not in source


def test_current_evidence_view_cannot_derive_its_own_latest_correction() -> None:
    source = _source("presentation/evidence_explorer_viewmodel.py")
    assert "CurrentStateService" in source
    assert "canonical_observation_history" not in source
    assert "history[-1]" not in source


def test_service_and_processing_views_share_the_authoritative_projection() -> None:
    read_model = _source("presentation/read_model/facade.py")
    processing = _source("presentation/operations_viewmodel.py")
    service = _source("core/service.py")
    assert "CurrentStateService" in read_model
    assert "self._current_state.documents()" in processing
    assert "self._read_model.current_document" in service
