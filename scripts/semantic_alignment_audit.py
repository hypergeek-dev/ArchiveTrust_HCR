"""Semantic Alignment Audit (2026-07-14 follow-up to the Review Packet Integrity Audit).

Classifies every remaining contested review packet in the *regenerated* Benchmark #1 queue (post
clustering fix) by why it entered review and whether candidates share semantic content or genuinely
diverge, and writes the result out as `benchmarks/calibration_corpus_1.jsonl` -- Calibration
Corpus #1 (docs/SEMANTIC_ALIGNMENT_AUDIT_2026-07-14.md). Read-only over archivetrust_data/;
regenerates Comparison/Confidence via `application.pipeline.run_comparison_and_assembly` against
persisted Evidence/Observations (no provider is re-invoked). Reproducible, like
`scripts/derive_benchmark.py`: rerunning this script regenerates the same corpus from the same
frozen telemetry (Comparison/Confidence/triage are all deterministic given fixed policies).
"""
from __future__ import annotations

import difflib
import io
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
CORPUS_OUT = REPO_ROOT / "benchmarks" / "calibration_corpus_1.jsonl"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split()).lower()


def _containment_ratio(short: str, long: str) -> float:
    """How much of `short` appears as one contiguous run inside `long` -- 1.0 if `short` is fully
    a substring (mod OCR noise via longest-common-substring rather than exact `in`), 0.0 if
    unrelated."""
    if not short:
        return 0.0
    matcher = difflib.SequenceMatcher(None, long, short, autojunk=False)
    match = matcher.find_longest_match(0, len(long), 0, len(short))
    return match.size / len(short)


def classify_packet(packet, canonical) -> dict:
    from archivetrust.domain.evidence.models import Precision
    from archivetrust.domain.comparison.geometry import intersection_over_union

    obs_type = packet.observation_type.value
    values = [c.value for c in packet.candidates if c.value]
    lengths = [len(v) for v in values]
    providers = sorted({c.provider_id for c in packet.candidates})
    any_missing_bbox = any(all(ev.bounding_box is None for ev in c.evidence) for c in packet.candidates)
    pages = {ev.page for c in packet.candidates for ev in c.evidence if ev.page is not None}

    scope_mismatched = bool(lengths) and min(lengths) > 0 and max(lengths) / min(lengths) > 5

    # Content-relatedness needs a *different* metric depending on length regime, verified by
    # manual inspection (docs/SEMANTIC_ALIGNMENT_AUDIT_2026-07-14.md): length-ratio-gated LCS
    # containment alone (the audit's first pass) mislabeled same-length-but-unrelated pairs (two
    # adjacent, semantically distinct form fields with overlapping boxes) as "legitimate OCR
    # disagreement" purely because their lengths happened to be close.
    #
    # - Length-mismatched pairs (one candidate much longer): LCS containment -- "is the short
    #   text's content one contiguous run inside the long text" -- is the right test for
    #   excerpt-in-blob (a small fragment vs a whole-page transcript).
    # - Similar-length pairs: LCS containment is unreliable (a single mid-string edit breaks the
    #   contiguous run and makes two 95%-identical strings score as "unrelated"). Use the
    #   Comparison Engine's own `comparison_confidence.magnitude` instead -- the same
    #   normalized-edit-distance-based agreement score `text_reconciliation.py` already computed
    #   for this exact pair, correctly calibrated for near-equal-length text.
    containment = None
    if len(values) >= 2:
        short = min(values, key=len)
        long_ = max(values, key=len)
        containment = round(_containment_ratio(_norm(short), _norm(long_)), 3)

    magnitude = packet.comparison_confidence.magnitude

    if containment is None:
        semantic_relation = "n/a"
    elif scope_mismatched:
        if containment >= 0.8:
            semantic_relation = "fully_contained"
        elif containment >= 0.3:
            semantic_relation = "partial_overlap"
        else:
            semantic_relation = "distinct_content"
    else:
        # Similar length: trust the engine's own edit-distance-based agreement magnitude.
        if magnitude is not None and magnitude >= 0.5:
            semantic_relation = "similar_content"
        elif magnitude is not None:
            semantic_relation = "distinct_content"
        else:
            semantic_relation = "n/a"

    # Real geometric IoU, when both candidates carry exactly one pixel-accurate box each -- lets
    # "these were clustered because of genuine spatial overlap" be distinguished from "these were
    # clustered some other way", and lets a near-threshold IoU (evidence of a marginal geometric
    # match) be reported rather than assumed.
    iou = None
    if len(packet.candidates) == 2:
        boxes = []
        for c in packet.candidates:
            if len(c.evidence) == 1 and c.evidence[0].bounding_box is not None:
                boxes.append(c.evidence[0].bounding_box)
        if len(boxes) == 2 and all(b.precision == Precision.PIXEL_ACCURATE for b in boxes):
            iou = round(intersection_over_union(boxes[0], boxes[1]), 4)

    # -- category assignment, in priority order, each grounded in a directly measured signal --
    if obs_type in ("table", "table_cell"):
        category = "table_reconstruction_disagreement"
    elif not pages:
        category = "missing_geometry"
    elif semantic_relation == "fully_contained":
        # A short candidate's text is (near-)entirely contained inside a long candidate's text --
        # both describe the same underlying content, just at different segmentation granularity.
        if any_missing_bbox:
            category = "provider_segmentation_philosophy"  # e.g. whole-page-transcript vs blocks
        else:
            category = "paragraph_vs_line_segmentation"
    elif semantic_relation == "similar_content":
        category = "legitimate_ocr_disagreement"
    elif semantic_relation == "distinct_content" and iou is not None:
        # Real pixel-accurate geometric overlap (this is what clustered them -- not the fixed
        # ordinal-fallback bug), but the recognized text shares almost nothing in common: either
        # two spatially adjacent but semantically distinct fields merged by a marginal IoU, or one
        # provider's OCR/layout genuinely captured only a small, unrelated-looking fragment of a
        # larger region the other provider read correctly. Both are the *same* observed mechanism
        # (see docs/SEMANTIC_ALIGNMENT_AUDIT_2026-07-14.md); reported together, IoU value retained
        # for the near-threshold-vs-not distinction.
        category = "low_iou_or_partial_capture_geometric_merge"
    elif semantic_relation == "distinct_content":
        category = "remaining_clustering_defect"  # distinct content, no 2-candidate pixel-accurate IoU to explain the merge (e.g. >2 candidates, or missing geometry)
    elif semantic_relation == "partial_overlap":
        category = "partial_content_overlap_needs_adjudication"
    else:
        category = "other"

    return {
        "semantic_slot_id": packet.semantic_slot_id,
        "document_ref": packet.document_ref,
        "archive_object_ref": packet.archive_object_ref,
        "canonical_observation_id": packet.canonical_observation_id,
        "observation_type": obs_type,
        "providers": providers,
        "num_candidates": len(packet.candidates),
        "candidate_lengths": lengths,
        "scope_mismatched": scope_mismatched,
        "containment_ratio": containment,
        "pairwise_iou": iou,
        "semantic_relation": semantic_relation,
        "any_missing_bbox": any_missing_bbox,
        "pages": sorted(pages),
        "comparison_classification": packet.comparison_confidence.classification.value,
        "comparison_magnitude": packet.comparison_confidence.magnitude,
        "canonical_confidence": packet.canonical_confidence.value if packet.canonical_confidence else None,
        "review_reason": packet.review_reason.value,
        "disclosure_tier": packet.disclosure_tier.value,
        "category": category,
        "clustering_basis": canonical.clustering_basis,
        "reconciliation_basis": canonical.reconciliation_basis,
        "evidence_ids": sorted({eid for c in packet.candidates for ev in c.evidence for eid in [ev.evidence_id]}),
        "observation_ids": sorted({c.observation_id for c in packet.candidates}),
    }


def main():
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.application.journal import Journal
    from archivetrust.review.triage import TriagePolicy
    from archivetrust.domain.comparison.policy import ReconciliationPolicy
    from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
    from archivetrust.domain.confidence.policy import ConfidencePolicy

    log("Loading telemetry (932MB)...")
    sink = FileTelemetrySink(WORKSPACE_DIR / "telemetry" / "events.jsonl")
    all_events = sink._events
    log(f"Loaded {len(all_events)} events")
    doc_refs = sorted({getattr(e, "document_ref", None) for e in all_events} - {None})

    journal = Journal()
    triage_policy = TriagePolicy()
    recon_policy = ReconciliationPolicy(policy_version=1)
    cap_matrix = production_capability_matrix()
    conf_policy = ConfidencePolicy(confidence_policy_version=1)

    from archivetrust.application.pipeline import run_comparison_and_assembly
    from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
    from archivetrust.domain.telemetry.events import (
        EvidenceCreated,
        EvidenceRejected,
        ObservationCreated,
        ProviderObservationAttempted,
    )
    from archivetrust.review.assembler import assemble_packet
    from archivetrust.review.triage import triage_review_queue
    from collections import defaultdict

    _PRE = (ProviderObservationAttempted, EvidenceRejected, EvidenceCreated, ObservationCreated)

    records = []
    for doc_ref in doc_refs:
        events = list(sink.events_for_document(doc_ref))
        if not events:
            continue

        pre = tuple(e for e in events if isinstance(e, _PRE))
        source_state = journal.replay(pre)
        obs_by_provider = defaultdict(list)
        for o in source_state.all_observations():
            obs_by_provider[(o.provider_id, o.provider_version)].append(o)
        if not obs_by_provider:
            continue
        graphs = tuple(
            ProviderObservationGraph(
                provider_id=p, provider_version=v, invocation_id=f"regenerated:{p}:{v}", observations=tuple(obs)
            )
            for (p, v), obs in sorted(obs_by_provider.items())
        )
        evidence_by_id = {e.evidence_id: e for e in source_state.all_evidence()}
        try:
            result = run_comparison_and_assembly(
                document_ref=doc_ref,
                archive_object_ref=doc_ref,
                provider_graphs=graphs,
                evidence_by_id=evidence_by_id,
                reconciliation_policy=recon_policy,
                capability_matrix=cap_matrix,
                confidence_policy=conf_policy,
            )
        except Exception as exc:
            log(f"  FAILED {doc_ref}: {exc}")
            continue

        canonical_by_id = {c.canonical_observation_id: c for c in result.reconciled_graph.canonical_observations}
        regenerated_state = journal.replay(pre + result.events)

        for item in triage_review_queue(regenerated_state, triage_policy):
            if item.reason.value != "sources_disagree":
                continue
            packet = assemble_packet(regenerated_state, item, document_ref=doc_ref, archive_object_ref=doc_ref)
            if len({c.provider_id for c in packet.candidates}) < 2:
                continue
            canonical = canonical_by_id.get(packet.canonical_observation_id)
            if canonical is None:
                continue
            records.append(classify_packet(packet, canonical))

    log(f"Total multi-provider contested packets classified: {len(records)}")

    category_counts = Counter(r["category"] for r in records)
    log("Category distribution:")
    for cat, n in category_counts.most_common():
        log(f"  {cat}: {n}")

    type_counts = Counter(r["observation_type"] for r in records)
    log("Observation-type distribution: " + json.dumps(dict(type_counts)))

    scope_records = [r for r in records if r["scope_mismatched"]]
    log(f"\nScope-mismatched packets: {len(scope_records)}")
    relation_counts = Counter(r["semantic_relation"] for r in scope_records)
    log("Semantic-relation distribution among scope-mismatched: " + json.dumps(dict(relation_counts)))

    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Wrote {len(records)} records to {CORPUS_OUT}")

    # A handful of concrete examples per category, for manual verification.
    log("\n--- sample records per category ---")
    seen_cats = set()
    for r in records:
        if r["category"] in seen_cats:
            continue
        seen_cats.add(r["category"])
        print(f"\n[{r['category']}] type={r['observation_type']} providers={r['providers']} lengths={r['candidate_lengths']} containment={r['containment_ratio']} relation={r['semantic_relation']}")
        print(f"  doc={r['document_ref']} slot={r['semantic_slot_id']}")
        print(f"  clustering_basis={r['clustering_basis']}")


if __name__ == "__main__":
    main()
