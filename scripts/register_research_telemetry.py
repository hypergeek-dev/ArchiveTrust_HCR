"""Retroactively registers the seven 2026-07-14 audit-like findings, two benchmarks, and (Phase 13)
this initiative's own closing validation as Research Telemetry (Constitution Article 29/34) -- the
one place backfilling is appropriate (`ARCHITECTURE_TELEMETRY_STANDARD.md` §9), since the goal is
specifically to make *existing* findings queryable, not to preserve a "never backfilled" purity that
would defeat this phase's purpose (`ROADMAP_TELEMETRY_STANDARD.md` Phase 9).

Writes only to `benchmarks/research_telemetry.jsonl` -- a repository-local, version-controlled
file -- never to any `archivetrust_data/workspaces/*/telemetry` directory, which may hold live
deployment state this script has no authority to mutate (Article 34, explicit direction 2026-07-14).

Idempotent: re-running this script against a fresh (or reset) output file always produces the same
ten events, in the same order -- it is not consulted at runtime by any production code path, only
by this repository's own regression tests (`tests/domain/research/test_registered_corpus.py`).
"""

from __future__ import annotations

from pathlib import Path

from archivetrust.domain.research.events import AuditConducted, BenchmarkExecuted

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "research_telemetry.jsonl"

# corpus_ref: the real Workspace this corpus's audits examined (per RCDP_REMAINING_CLUSTERING_
# DEFECT_2026-07-14.md's own script constants) -- an identifier, never fabricated.
_CORPUS_REF = "workspace:46e94f15d73347c2be79f1527e646c11"


def _audit(
    audit_id: str,
    *,
    title: str,
    claim: str,
    verdict: str,
    population_size: int | None = None,
    sample_basis: str | None = None,
    supersedes_audit_id: str | None = None,
) -> AuditConducted:
    return AuditConducted(
        event_id=f"event_{audit_id}",
        corpus_ref=_CORPUS_REF,
        audit_id=audit_id,
        title=title,
        claim=claim,
        verdict=verdict,
        population_size=population_size,
        sample_basis=sample_basis,
        supersedes_audit_id=supersedes_audit_id,
    )


def build_events() -> tuple[AuditConducted | BenchmarkExecuted, ...]:
    """The nine events this phase registers, in the order they were actually conducted."""
    review_packet_integrity = _audit(
        "review-packet-integrity-2026-07-14",
        title="Review Packet Integrity Audit",
        claim=(
            "Nonsensical review items in the production review queue trace to a specific root "
            "cause in domain/comparison/clustering.py's ordinal fallback."
        ),
        verdict=(
            "Confirmed: the ordinal fallback used a hash-sort as a spatial proxy, producing "
            "84-93% of contested packets against the real corpus. Fixed and confirmed."
        ),
        population_size=None,
        sample_basis="full production corpus, 2026-07-13 benchmark run",
    )

    semantic_alignment = _audit(
        "semantic-alignment-2026-07-14",
        title="Semantic Alignment Audit",
        claim=(
            "Classify all 522 review packets remaining after the Review Packet Integrity Audit's "
            "fix, to determine what mechanism produces each."
        ),
        verdict=(
            "67% (350/522) are a real, near-threshold IoU merge -- a different, not-yet-fixed "
            "mechanism than the hash-sort defect. Calibration Corpus #1 built from the "
            "classification."
        ),
        population_size=522,
        supersedes_audit_id=review_packet_integrity.audit_id,
    )

    human_reviewability = _audit(
        "human-reviewability-2026-07-14",
        title="Human Reviewability Audit",
        claim=(
            "Are the review packets the Semantic Alignment Audit classified actually reviewable "
            "by a human -- does an Accept/Reject/Edit decision have a well-formed meaning?"
        ),
        verdict=(
            "4 of 6 sampled packets were fundamentally unreviewable (no ReviewAction means 'both "
            "candidates are correct, describing different things'). Accept never narrows "
            "contributing_observations provenance. Found one mislabeled sample in the Semantic "
            "Alignment Audit's own prior classification."
        ),
        population_size=6,
        sample_basis="hand-selected sample across classification categories",
        supersedes_audit_id=semantic_alignment.audit_id,
    )

    semantic_contract = _audit(
        "semantic-contract-2026-07-14",
        title="Semantic Contract Audit",
        claim=(
            "Meta-audit: do the three prior audits' findings hold, or does an earlier break in "
            "the pipeline explain some of what they attribute to clustering/comparison?"
        ),
        verdict=(
            "Found an EARLIER break than all three prior audits: tesseract_layoutparser's "
            "'Text' native label is unconditionally mapped to PARAGRAPH, with no telemetry "
            "visibility. Downgrades the prior audits' 'Comparison Engine layer' recommendations "
            "to the provider-adapter layer instead."
        ),
        supersedes_audit_id=human_reviewability.audit_id,
    )

    observation_typing_prevalence = _audit(
        "observation-typing-prevalence-2026-07-14",
        title="Observation Typing Prevalence Audit",
        claim=(
            "Full-corpus measurement of the Semantic Contract Audit's Text->PARAGRAPH finding: "
            "how prevalent is it, and what would fixing it change?"
        ),
        verdict=(
            "100% prevalence of the mapping; 85.82% of contested packets touch it. Counterfactual "
            "fix: 522 -> 332 contested packets. Both H1 (bug) and H2 (real disagreement) are "
            "supported for different segments of the corpus."
        ),
        population_size=522,
        supersedes_audit_id=semantic_contract.audit_id,
    )

    rcdp = _audit(
        "rcdp-remaining-clustering-defect-2026-07-14",
        title="RCDP: remaining_clustering_defect",
        claim=(
            "Of the 332 packets remaining after the typing fix, the largest category "
            "(remaining_clustering_defect, 126) is explained by clustering.py's ordinal fallback "
            "conflating missing geometry with coarse geometry. EVALUATED ONLY AS A COUNTERFACTUAL "
            "(scripts/rcdp_ordinal_geometry_counterfactual.py's alternate AlignmentService) -- "
            "no file under src/ was modified; this claim describes what a proposed fix would do, "
            "never code merged into production (ARCHITECTURE_TELEMETRY_STANDARD.md S1.1)."
        ),
        verdict=(
            "Mechanism confirmed (100% prevalence in the 126-packet category). The proposed "
            "counterfactual fix would reduce 332 contested packets to 0 for this mechanism -- "
            "but see the Adversarial Architecture Audit, which falsifies this as a general-"
            "purpose fix."
        ),
        population_size=126,
        supersedes_audit_id=observation_typing_prevalence.audit_id,
    )

    adversarial = _audit(
        "adversarial-architecture-2026-07-14",
        title="Adversarial Architecture Audit: What Did We Accidentally Remove?",
        claim=(
            "Falsification attempt against the RCDP finding's proposed geometry-precision guard: "
            "does that counterfactual fix remove incorrect comparisons, or remove comparisons "
            "entirely? EVALUATED ONLY AS A COUNTERFACTUAL, same alternate AlignmentService as "
            "RCDP -- no file under src/ was modified by either audit "
            "(ARCHITECTURE_TELEMETRY_STANDARD.md S1.1)."
        ),
        verdict=(
            "FALSIFIED as a general-purpose fix: it removes the entire Docling<->PaddleOCR-VL "
            "comparison capability structurally, not a narrow subset. 60% of a random sample "
            "(seed=42, n=30) carried unique, high-value PaddleOCR-VL content that would become "
            "permanently unreachable. Should not be shipped unmodified."
        ),
        population_size=332,
        sample_basis="random sample, seed=42, n=30, real text inspected not inferred",
        supersedes_audit_id=rcdp.audit_id,
    )

    benchmark_1 = BenchmarkExecuted(
        event_id="event_benchmark-1-2026-07-13",
        corpus_ref=_CORPUS_REF,
        benchmark_id="benchmark-1-2026-07-13",
        title="Benchmark #1 (2026-07-13 run)",
        document_count=478,
        metrics_ref="benchmarks/BENCHMARK_1_REPORT.md",
    )

    calibration_sufficiency = BenchmarkExecuted(
        event_id="event_calibration-sufficiency-2026-07-14",
        corpus_ref=_CORPUS_REF,
        benchmark_id="calibration-sufficiency-2026-07-14",
        title="Confidence Calibration Sufficiency Analysis",
        metrics_ref="benchmarks/CONFIDENCE_CALIBRATION_SUFFICIENCY_REPORT.md",
        supersedes_benchmark_id=benchmark_1.benchmark_id,
    )

    closing_validation = _audit(
        "telemetry-standard-closing-validation-2026-07-14",
        title="Telemetry Constitution & Architecture Initiative: Closing Validation (Phases 0-13)",
        claim=(
            "docs/ARCHITECTURE_TELEMETRY_STANDARD.md and docs/ROADMAP_TELEMETRY_STANDARD.md's "
            "14-phase initiative (Articles 26-34) holds against the real corpus and the six prior "
            "2026-07-14 audits that motivated it: each audit is reproducible by one named query "
            "(src/archivetrust/research/queries.py), the Adversarial Audit's falsification is a "
            "permanent CI-gated regression (Article 31), and the new telemetry does not "
            "materially degrade processing throughput."
        ),
        verdict=(
            "Confirmed. Phase 11: all six audits reproduced by named query against this same "
            "registry (tests/research/test_queries.py, 14 tests). Phase 12: measured +2.0% "
            "Journal.replay overhead per document (before/after this initiative's own code, via "
            "git stash around the real corpus's telemetry, since this session made no commits), "
            "within the 5% tolerance -- no optimization applied. Phase 13: the Adversarial Audit's "
            "falsification re-expressed as tests/domain/alignment/test_adversarial_audit_"
            "regression.py, asserting today's production ClusteringAlignmentService still clusters "
            "(never structurally excludes) a Docling/PaddleOCR-VL pair shaped like the real "
            "eliminated packets -- this fails the day the unshipped RCDP clustering-exclusion "
            "mechanism ships without satisfying Article 31 first. Full regression suite: 819 passed, 13 "
            "pre-existing GPU-VRAM-exhaustion failures confirmed unrelated (dev-machine resource "
            "constraint, not a code defect)."
        ),
        sample_basis="all prior 2026-07-14 audits/benchmarks in this registry, plus this session's own test suite",
    )

    return (
        review_packet_integrity,
        semantic_alignment,
        human_reviewability,
        semantic_contract,
        observation_typing_prevalence,
        rcdp,
        adversarial,
        benchmark_1,
        calibration_sufficiency,
        closing_validation,
    )


def main() -> None:
    from archivetrust.infrastructure.storage.research_telemetry_sink import (
        FileResearchTelemetrySink,
    )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()  # idempotent re-run: rebuild from scratch, never append duplicates
    sink = FileResearchTelemetrySink(OUTPUT_PATH)
    for event in build_events():
        sink.append(event)
    print(f"Wrote {len(list(sink.all_events()))} events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
