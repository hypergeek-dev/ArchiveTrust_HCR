from __future__ import annotations

from pathlib import Path

from archivetrust.domain.telemetry.events import TelemetryEventKind


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_telemetry_standard_names_every_production_event_kind() -> None:
    standard = read("docs/TELEMETRY_STANDARD_V1.md")

    for kind in TelemetryEventKind:
        assert f"`{kind.value}`" in standard


def test_public_telemetry_standard_declares_current_production_boundary() -> None:
    standard = read("docs/TELEMETRY_STANDARD_V1.md")

    assert "Current production behavior profile" in standard
    assert "does not" in standard
    assert "every aspiration" in standard
    assert "Unknown `kind` values must not be silently accepted" in standard


def test_public_telemetry_standard_covers_f3_f4_and_c3_behavior() -> None:
    standard = read("docs/TELEMETRY_STANDARD_V1.md")

    assert "CandidateExcludedBatch" in standard
    assert "Raw Output Blob Indirection" in standard
    assert "ReviewPacketDispatched" in standard
    assert "ReviewPacketClosed" in standard
    assert "<stream>.chain.json" in standard
    assert "<export>.integrity.json" in standard


def test_public_telemetry_standard_covers_f5_scope_exclusion_code() -> None:
    standard = read("docs/TELEMETRY_STANDARD_V1.md")
    checklist = read("docs/TELEMETRY_STANDARD_CONFORMANCE_CHECKLIST.md")

    assert "SCOPE_MISMATCH" in standard
    assert "scope-compatibility policy gate" in standard
    assert "SCOPE_MISMATCH" in checklist
    assert "policy exclusion" in checklist


def test_mapping_doc_names_prov_openlineage_and_extension_boundary() -> None:
    mapping = read("docs/TELEMETRY_STANDARD_PROV_OPENLINEAGE_MAPPING.md")

    assert "PROV-O" in mapping
    assert "OpenLineage" in mapping
    assert "not an implemented exporter" in mapping
    assert "custom facets" in mapping


def test_conformance_checklist_has_replay_integrity_and_review_closure_sections() -> None:
    checklist = read("docs/TELEMETRY_STANDARD_CONFORMANCE_CHECKLIST.md")

    assert "## 3. Replay" in checklist
    assert "## 6. Review Closure" in checklist
    assert "Detects bit flips, truncation, and reorder" in checklist
    assert "flattening review outcome and packet closure" in checklist
