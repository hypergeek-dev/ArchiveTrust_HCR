"""Phase 9's acceptance criterion, proven directly: "list every audit that examined
clustering.py's geometry-precision guard, in order" returns the RCDP finding, then the Adversarial
Audit that falsifies it, with the supersession explicit -- reconstructed from the checked-in
registry (`benchmarks/research_telemetry.jsonl`), never by reading the source markdown documents.
"""

from __future__ import annotations

from pathlib import Path

from archivetrust.domain.research.events import AuditConducted, BenchmarkExecuted
from archivetrust.infrastructure.storage.research_telemetry_sink import FileResearchTelemetrySink

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "benchmarks" / "research_telemetry.jsonl"


def _load_audits() -> tuple[AuditConducted, ...]:
    sink = FileResearchTelemetrySink(REGISTRY_PATH)
    return tuple(e for e in sink.all_events() if isinstance(e, AuditConducted))


def test_registry_file_exists_and_is_reproducible_from_the_script():
    import subprocess
    import sys

    assert REGISTRY_PATH.exists(), (
        "benchmarks/research_telemetry.jsonl is missing -- run "
        "scripts/register_research_telemetry.py"
    )
    repo_root = REGISTRY_PATH.parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "register_research_telemetry.py")],
        cwd=repo_root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Wrote 10 events" in result.stdout


def test_ten_events_registered_eight_audits_two_benchmarks():
    sink = FileResearchTelemetrySink(REGISTRY_PATH)
    all_events = list(sink.all_events())
    audits = [e for e in all_events if isinstance(e, AuditConducted)]
    benchmarks = [e for e in all_events if isinstance(e, BenchmarkExecuted)]
    assert len(audits) == 8
    assert len(benchmarks) == 2
    assert len(all_events) == 10


def test_query_reproduces_rcdp_to_adversarial_audit_supersession_in_order():
    # The exact acceptance criterion: "list every audit that examined clustering.py's
    # geometry-precision guard, in order" -- both RCDP and the Adversarial Audit mention it in
    # their claim text; this query finds them by that content, in registration (append) order.
    audits = _load_audits()
    examined_the_guard = [a for a in audits if "geometry" in a.claim.lower() or "geometry" in a.verdict.lower()]

    assert len(examined_the_guard) == 2
    rcdp, adversarial = examined_the_guard
    assert rcdp.audit_id == "rcdp-remaining-clustering-defect-2026-07-14"
    assert adversarial.audit_id == "adversarial-architecture-2026-07-14"
    assert adversarial.supersedes_audit_id == rcdp.audit_id


def test_rcdp_and_adversarial_claims_state_the_counterfactual_explicitly():
    # Constitution Article 34/S1.1: the registration itself must not repeat the conflation the
    # 2026-07-14 correction fixed -- both claims must say "counterfactual", never imply shipped code.
    audits = {a.audit_id: a for a in _load_audits()}
    rcdp = audits["rcdp-remaining-clustering-defect-2026-07-14"]
    adversarial = audits["adversarial-architecture-2026-07-14"]
    for audit in (rcdp, adversarial):
        assert "counterfactual" in audit.claim.lower()
        assert "no file under src/ was modified" in audit.claim.lower() or "no file under src" in audit.claim.lower()


def test_supersession_chain_covers_all_seven_audits_in_conducted_order():
    audits = {a.audit_id: a for a in _load_audits()}
    chain = [
        "review-packet-integrity-2026-07-14",
        "semantic-alignment-2026-07-14",
        "human-reviewability-2026-07-14",
        "semantic-contract-2026-07-14",
        "observation-typing-prevalence-2026-07-14",
        "rcdp-remaining-clustering-defect-2026-07-14",
        "adversarial-architecture-2026-07-14",
    ]
    for earlier, later in zip(chain, chain[1:]):
        assert audits[later].supersedes_audit_id == earlier, f"{later} should supersede {earlier}"


def test_benchmark_1_is_registered_with_document_count():
    sink = FileResearchTelemetrySink(REGISTRY_PATH)
    benchmark_1 = next(
        e for e in sink.all_events()
        if isinstance(e, BenchmarkExecuted) and e.benchmark_id == "benchmark-1-2026-07-13"
    )
    assert benchmark_1.document_count == 478


def test_phase_13_closing_validation_is_registered_and_does_not_supersede_any_prior_finding():
    # Phase 13's own deliverable: "a final validation report, structured as its own AuditConducted
    # event". It validates the initiative, it doesn't invalidate any prior finding's claim, so it
    # deliberately carries no supersedes_audit_id (distinct from the RCDP->Adversarial chain, which
    # names a real ordinal fallback improvement).
    audits = {a.audit_id: a for a in _load_audits()}
    closing = audits["telemetry-standard-closing-validation-2026-07-14"]
    assert closing.supersedes_audit_id is None
    assert "819 passed" in closing.verdict
    assert "+2.0%" in closing.verdict
