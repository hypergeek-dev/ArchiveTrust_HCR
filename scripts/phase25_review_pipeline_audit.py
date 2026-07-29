"""Phase 25 — Human Review Pipeline & Operational Audit.

Research-only script (not production code, not imported by the app): reconstructs every reviewed
ReviewPacket exactly as it was presented to the reviewer (a full Journal replay per reviewed
document, `assemble_packet` against the *pre-correction* canonical each `HumanCorrectionSubmitted`
targeted), then measures review-packet quality, source-selection asymmetry, ambiguity taxonomy,
table handling, and human-effort distribution.

Usage: `python scripts/phase25_review_pipeline_audit.py`. Writes `d:/tmp/phase25/packets.json`
(one record per reviewed packet, full detail) for further ad-hoc inspection, and prints summary
tables to stdout.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from archivetrust.application.journal import Journal
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.telemetry.events import parse_event
from archivetrust.review.assembler import assemble_packet
from archivetrust.review.packet import ReviewReason
from archivetrust.review.triage import TriageItem, TriagePolicy, _reason_for, _tier_for

WORK_DIR = Path("d:/tmp/phase25")
CORRECTIONS_PATH = WORK_DIR / "corrections.json"
DOC_EVENT_FILES = {
    "archive_object_c48423aa1f654b37a08ed970160b29c6": WORK_DIR
    / "archive_object_c48423aa1f654b37a08ed970160b29c6.jsonl",
    "archive_object_dbc6cc8190ea48de91fdec233e56bf99": WORK_DIR
    / "archive_object_dbc6cc8190ea48de91fdec233e56bf99.jsonl",
    "archive_object_2e3a7098428a44679e4e8ff3a17789d4": WORK_DIR
    / "archive_object_2e3a7098428a44679e4e8ff3a17789d4.jsonl",
}
ARCHIVE_REF = {doc: doc for doc in DOC_EVENT_FILES}  # archive_object_ref == document_ref here


def _load_journal_state(path: Path):
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(parse_event(json.loads(line)))
    return Journal().replay(events)


def _norm(text: str | None) -> str:
    return (text or "").strip()


def main() -> None:
    corrections = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    policy = TriagePolicy()

    states = {doc: _load_journal_state(path) for doc, path in DOC_EVENT_FILES.items()}

    records = []
    errors = []
    for corr in corrections:
        doc = corr["document_ref"]
        state = states[doc]
        original_id = corr["target_canonical_observation_id"]
        try:
            canonical = state.canonical_observation(original_id)
        except KeyError as exc:
            errors.append({"correction_id": corr["correction_id"], "error": repr(exc)})
            continue

        reason = _reason_for(canonical, policy)
        if reason is None:
            # Shouldn't happen (it was queued and reviewed), but record honestly if it does.
            errors.append(
                {"correction_id": corr["correction_id"], "error": "reason_for returned None"}
            )
            continue
        tier = _tier_for(canonical, reason)
        item = TriageItem(
            semantic_slot_id=corr["semantic_slot_id"],
            canonical_observation_id=original_id,
            observation_type=canonical.observation_type,
            reason=reason,
            disclosure_tier=tier,
            policy_version=policy.version,
        )
        try:
            packet = assemble_packet(
                state, item, document_ref=doc, archive_object_ref=ARCHIVE_REF[doc]
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"correction_id": corr["correction_id"], "error": repr(exc)})
            continue

        candidates = [
            {
                "observation_id": c.observation_id,
                "provider_id": c.provider_id,
                "value": c.value,
                "provider_confidence": c.provider_confidence,
                "has_bbox": any(e.bounding_box is not None for e in c.evidence),
                "has_evidence": len(c.evidence) > 0,
                "pages": sorted({e.page for e in c.evidence if e.page is not None}),
            }
            for c in packet.candidates
        ]
        values = [c["value"] for c in candidates if c["value"]]
        lengths = [len(v) for v in values]
        length_ratio = (max(lengths) / min(lengths)) if len(lengths) >= 2 and min(lengths) > 0 else None
        distinct_values = len({_norm(v) for v in values})

        action = corr["action"]
        raw_corrected = corr.get("raw_corrected_output")
        current_value = packet.current_value

        chosen_provider = None
        chosen_kind = None  # "confirmed_current" | "picked_other_candidate" | "typed_new_text" | "ambiguous_no_value_change"
        if action == "accept_provider":
            chosen_kind = "confirmed_current"
            agreeing = [c["provider_id"] for c in candidates if _norm(c["value"]) == _norm(current_value)]
            chosen_provider = agreeing
        elif action == "manual_edit":
            matched = [c["provider_id"] for c in candidates if _norm(c["value"]) == _norm(raw_corrected)]
            if matched:
                chosen_kind = "picked_other_candidate"
                chosen_provider = matched
            else:
                chosen_kind = "typed_new_text"
        elif action == "mark_ambiguous":
            chosen_kind = "ambiguous_no_value_change"

        records.append(
            {
                "correction_id": corr["correction_id"],
                "document_ref": doc,
                "semantic_slot_id": corr["semantic_slot_id"],
                "action": action,
                "observation_type": packet.observation_type.value,
                "review_reason": reason.value,
                "disclosure_tier": tier.value,
                "agreement_classification": packet.agreement.classification.value,
                "independent_source_count": packet.agreement.independent_source_count,
                "comparison_confidence_magnitude": packet.comparison_confidence.magnitude,
                "canonical_confidence": packet.canonical_confidence.value if packet.canonical_confidence else None,
                "candidate_count": len(candidates),
                "distinct_value_count": distinct_values,
                "length_ratio": length_ratio,
                "any_missing_geometry": any(not c["has_bbox"] for c in candidates),
                "all_missing_geometry": all(not c["has_bbox"] for c in candidates) if candidates else None,
                "candidates": candidates,
                "chosen_kind": chosen_kind,
                "chosen_provider": chosen_provider,
                "review_duration_seconds": None,  # attached below from HumanCorrectionSubmitted
            }
        )

    # attach review_duration_seconds from the original corrections list (kept separate above for clarity)
    dur_by_id = {}
    with open(WORK_DIR / "review_events.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["kind"] == "HumanCorrectionSubmitted":
                dur_by_id[d["correction_id"]] = d.get("review_duration_seconds")
    for r in records:
        r["review_duration_seconds"] = dur_by_id.get(r["correction_id"])

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "packets.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (WORK_DIR / "packet_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    print(f"Reconstructed {len(records)} / {len(corrections)} packets ({len(errors)} errors)")
    if errors:
        for e in errors[:10]:
            print("  ERROR:", e)

    # ---- Q2: outcome distribution (already known, restate) ----
    print("\n== Action distribution ==")
    print(Counter(r["action"] for r in records))

    # ---- Q3: source-selection asymmetry ----
    print("\n== Chosen-kind distribution ==")
    print(Counter(r["chosen_kind"] for r in records))

    provider_wins = Counter()
    provider_available = Counter()
    for r in records:
        providers_here = {c["provider_id"] for c in r["candidates"]}
        for p in providers_here:
            provider_available[p] += 1
        if r["chosen_provider"]:
            for p in set(r["chosen_provider"]):
                provider_wins[p] += 1
    print("\n== Provider availability (packets where provider contributed a candidate) ==")
    print(provider_available)
    print("\n== Provider 'chosen' counts (accept_provider confirmed-current OR manual_edit matched an existing candidate) ==")
    print(provider_wins)
    print("win rate (wins / available):")
    for p in provider_available:
        w = provider_wins.get(p, 0)
        a = provider_available[p]
        print(f"  {p}: {w}/{a} = {w/a:.1%}")

    # ---- Q4: ambiguity taxonomy ----
    ambiguous = [r for r in records if r["action"] == "mark_ambiguous"]
    print(f"\n== Ambiguous packets: {len(ambiguous)} ==")

    def _classify_ambiguous(r) -> str:
        if r["observation_type"] in ("table", "table_cell"):
            return "table_structure"
        if r["candidate_count"] < 2:
            return "missing_candidate"
        if r["all_missing_geometry"]:
            return "missing_geometry"
        if r["length_ratio"] is not None and r["length_ratio"] >= 5.0:
            return "scope_mismatch"
        if r["distinct_value_count"] <= 1:
            return "poor_packet_construction_no_real_disagreement"
        if r["any_missing_geometry"]:
            return "partial_missing_geometry"
        return "genuine_semantic_disagreement_or_other"

    taxonomy = Counter(_classify_ambiguous(r) for r in ambiguous)
    print(taxonomy)

    # ---- Q1: reviewability proxy across ALL packets (not just ambiguous) ----
    def _reviewability(r) -> str:
        if r["candidate_count"] < 2:
            return "unreviewable_missing_candidate"
        if r["all_missing_geometry"]:
            return "unreviewable_no_geometry"
        if r["length_ratio"] is not None and r["length_ratio"] >= 5.0:
            return "reviewable_with_effort_scope_mismatch"
        if r["distinct_value_count"] <= 1:
            return "reviewable_but_no_real_disagreement"
        return "immediately_reviewable"

    print("\n== Packet quality (all 162) ==")
    print(Counter(_reviewability(r) for r in records))

    # ---- Q5: table-specific ----
    tables = [r for r in records if r["observation_type"] in ("table", "table_cell")]
    table_ids = {r["correction_id"] for r in tables}
    print(f"\n== Table-type packets: {len(tables)} / {len(records)} ==")
    print("table action distribution:", Counter(r["action"] for r in tables))
    print(
        "non-table action distribution:",
        Counter(r["action"] for r in records if r["correction_id"] not in table_ids),
    )

    # ---- Q6: human effort ----
    print("\n== Review duration by action (seconds) ==")
    by_action_dur = defaultdict(list)
    for r in records:
        if r["review_duration_seconds"] is not None:
            by_action_dur[r["action"]].append(r["review_duration_seconds"])
    for action, durs in by_action_dur.items():
        print(
            f"  {action}: n={len(durs)} mean={statistics.mean(durs):.2f}s "
            f"median={statistics.median(durs):.2f}s min={min(durs):.2f}s max={max(durs):.2f}s "
            f"total={sum(durs):.1f}s"
        )
    total_seconds = sum(r["review_duration_seconds"] or 0 for r in records)
    print(f"  TOTAL reviewer time across all 162 packets: {total_seconds:.1f}s = {total_seconds/60:.1f} min")

    # ---- confidence / agreement distributions ----
    print("\n== Agreement classification distribution ==")
    print(Counter(r["agreement_classification"] for r in records))
    print("\n== Review reason distribution ==")
    print(Counter(r["review_reason"] for r in records))
    print("\n== Disclosure tier distribution ==")
    print(Counter(r["disclosure_tier"] for r in records))
    print("\n== Observation type distribution ==")
    print(Counter(r["observation_type"] for r in records))
    print("\n== Independent source count distribution ==")
    print(Counter(r["independent_source_count"] for r in records))

    ccm = [r["comparison_confidence_magnitude"] for r in records if r["comparison_confidence_magnitude"] is not None]
    if ccm:
        print(f"\ncomparison_confidence magnitude: n={len(ccm)} mean={statistics.mean(ccm):.3f} median={statistics.median(ccm):.3f}")
    canc = [r["canonical_confidence"] for r in records if r["canonical_confidence"] is not None]
    if canc:
        print(f"canonical_confidence: n={len(canc)} mean={statistics.mean(canc):.3f} median={statistics.median(canc):.3f}")


if __name__ == "__main__":
    main()
