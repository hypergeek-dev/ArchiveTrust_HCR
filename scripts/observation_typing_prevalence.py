"""Observation Typing Prevalence & Downstream Impact Measurement (2026-07-14 follow-up to
SEMANTIC_CONTRACT_AUDIT_2026-07-14.md).

That audit demonstrated the *mechanism* -- tesseract_layoutparser's native_label=="Text" maps
unconditionally to ObservationType.PARAGRAPH -- against 3 hand-picked observations. It explicitly
did not measure corpus-wide prevalence or downstream impact. This script does that, over the full
frozen benchmark corpus (archivetrust_data/workspaces/46e94f15d73347c2be79f1527e646c11), read-only,
deterministic, reproducible (same pattern as scripts/semantic_alignment_audit.py).

Produces:
  - Phase 1: prevalence of native_label=="Text" among tesseract_layoutparser Observations.
  - Phase 4: downstream fate of every such Observation (uncontested / corroborated / contested,
    and if contested, which Semantic Alignment Audit category), measured for the FULL population
    (not sampled -- no extrapolation needed for Phase 4).
  - Phase 2: a reproducible (seed=42) stratified random sample of native_label=="Text" observations
    across 7 strata, written to benchmarks/observation_typing_sample.jsonl for manual textual
    inspection (Phase 3 confusion matrix is computed from that inspection in a follow-up script).

Strata definition (mutually exclusive, applied to canonical observations that have at least one
contributing tesseract_layoutparser native_label=="Text" observation):
  - uncontested: canonical was never triaged into the review queue at all
  - single_source_or_low_confidence: triaged, but not for SOURCES_DISAGREE
  - corroborated: comparison_confidence.classification == CORROBORATED (implies not contested)
  - contested_other: CONTESTED and triaged SOURCES_DISAGREE, but packet category is not one of
    the 4 named Semantic Alignment Audit categories below
  - low_iou_or_partial_capture_geometric_merge / remaining_clustering_defect /
    provider_segmentation_philosophy / paragraph_vs_line_segmentation: the named categories from
    scripts/semantic_alignment_audit.py's classify_packet(), reused unmodified.

Note "corroborated" and "uncontested"/"single_source_or_low_confidence" can overlap conceptually
(a CORROBORATED canonical is by definition not triaged for SOURCES_DISAGREE, but may still be
triaged for LOW_CONFIDENCE) -- resolved by priority order: contested-category strata first, then
corroborated, then uncontested/single-source, so every observation lands in exactly one stratum.
"""
from __future__ import annotations

import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
PREVALENCE_OUT = REPO_ROOT / "benchmarks" / "observation_typing_prevalence.json"
SAMPLE_OUT = REPO_ROOT / "benchmarks" / "observation_typing_sample.jsonl"

SAMPLE_SEED = 42
SAMPLE_SIZE_PER_STRATUM = 20

NAMED_CATEGORIES = (
    "low_iou_or_partial_capture_geometric_merge",
    "remaining_clustering_defect",
    "provider_segmentation_philosophy",
    "paragraph_vs_line_segmentation",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def native_label_of(observation, evidence_by_id) -> str | None:
    if not observation.evidence_ids:
        return None
    ev = evidence_by_id.get(observation.evidence_ids[0])
    if ev is None:
        return None
    try:
        raw = json.loads(ev.raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    return raw.get("native_label")


def main():
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.application.journal import Journal
    from archivetrust.application.pipeline import run_comparison_and_assembly
    from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
    from archivetrust.domain.telemetry.events import (
        EvidenceCreated,
        EvidenceRejected,
        ObservationCreated,
        ProviderObservationAttempted,
    )
    from archivetrust.domain.comparison.policy import ReconciliationPolicy
    from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
    from archivetrust.domain.confidence.policy import ConfidencePolicy
    from archivetrust.domain.confidence.models import ComparisonClassification
    from archivetrust.review.triage import TriagePolicy, review_reason_for, triage_review_queue, ReviewReason
    from archivetrust.review.assembler import assemble_packet

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from semantic_alignment_audit import classify_packet  # reuse unmodified

    _PRE = (ProviderObservationAttempted, EvidenceRejected, EvidenceCreated, ObservationCreated)

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

    # -- Phase 1 accumulators --
    total_tlp_observations = 0
    text_labeled_count = 0
    text_labeled_by_mapped_type = Counter()
    text_labeled_docs = set()
    all_tlp_docs = set()
    per_doc_text_count = Counter()

    # -- Phase 4 / stratification accumulators --
    stratum_pool: dict[str, list[dict]] = defaultdict(list)  # stratum -> list of sample records
    stratum_counts = Counter()
    total_contested_packets = [0]  # mutable cell (closed-over-free-var-free), corpus-wide packet count
    contested_packets_with_mistyped_source = [0]

    failed_docs = 0
    for idx, doc_ref in enumerate(doc_refs):
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

        evidence_by_id = {e.evidence_id: e for e in source_state.all_evidence()}

        # -- Phase 1: prevalence, independent of comparison/clustering --
        tlp_observations = [
            o for (p, v), obs in obs_by_provider.items() if p == "tesseract_layoutparser" for o in obs
        ]
        if not tlp_observations:
            continue
        all_tlp_docs.add(doc_ref)
        doc_text_labeled_ids = set()
        for o in tlp_observations:
            total_tlp_observations += 1
            label = native_label_of(o, evidence_by_id)
            if label == "Text":
                text_labeled_count += 1
                text_labeled_by_mapped_type[o.observation_type.value] += 1
                doc_text_labeled_ids.add(o.observation_id)
        if doc_text_labeled_ids:
            text_labeled_docs.add(doc_ref)
            per_doc_text_count[doc_ref] = len(doc_text_labeled_ids)

        if not doc_text_labeled_ids:
            continue

        # -- Phase 4 / stratification: need comparison+triage to know each Text-labeled
        # observation's downstream fate. --
        graphs = tuple(
            ProviderObservationGraph(
                provider_id=p, provider_version=v, invocation_id=f"regenerated:{p}:{v}", observations=tuple(obs)
            )
            for (p, v), obs in sorted(obs_by_provider.items())
        )
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
            failed_docs += 1
            log(f"  FAILED {doc_ref}: {exc}")
            continue

        canonicals = result.reconciled_graph.canonical_observations
        # observation_id -> canonical (first found; shared-contribution is rare/edge-case)
        canonical_by_obs_id: dict[str, object] = {}
        canonical_by_id_full: dict[str, object] = {c.canonical_observation_id: c for c in canonicals}
        for c in canonicals:
            for ref in c.contributing_observations:
                canonical_by_obs_id.setdefault(ref.observation_id, c)

        regenerated_state = journal.replay(pre + result.events)
        triage_items = triage_review_queue(regenerated_state, triage_policy)
        triage_reason_by_canonical = {t.canonical_observation_id: t.reason for t in triage_items}

        # Build packet category only for contested (sources_disagree) canonicals -- reuse
        # classify_packet from the Semantic Alignment Audit, unmodified.
        category_by_canonical: dict[str, str] = {}
        for item in triage_items:
            if item.reason != ReviewReason.SOURCES_DISAGREE:
                continue
            packet = assemble_packet(regenerated_state, item, document_ref=doc_ref, archive_object_ref=doc_ref)
            if len({c.provider_id for c in packet.candidates}) < 2:
                continue
            canonical = next((c for c in canonicals if c.canonical_observation_id == item.canonical_observation_id), None)
            if canonical is None:
                continue
            category_by_canonical[item.canonical_observation_id] = classify_packet(packet, canonical)["category"]

        # -- Phase 7 Q2: packet-level (not observation-level) overlap. For every contested,
        # multi-provider, sources_disagree packet in this document (the same population
        # scripts/semantic_alignment_audit.py measured as 522 corpus-wide), does it contain at
        # least one contributing Observation that is a tesseract_layoutparser
        # native_label=='Text' observation? --
        for canonical_id in category_by_canonical:
            total_contested_packets[0] += 1
            canonical = canonical_by_id_full.get(canonical_id)
            if canonical is None:
                continue
            contributing_ids = {ref.observation_id for ref in canonical.contributing_observations}
            if contributing_ids & doc_text_labeled_ids:
                contested_packets_with_mistyped_source[0] += 1

        for o in tlp_observations:
            if o.observation_id not in doc_text_labeled_ids:
                continue
            canonical = canonical_by_obs_id.get(o.observation_id)
            record = {
                "document_ref": doc_ref,
                "observation_id": o.observation_id,
                "text": getattr(o.payload, "text", None),
                "mapped_observation_type": o.observation_type.value,
                "native_label": "Text",
            }
            if canonical is None:
                stratum = "no_canonical_link"  # should not happen; every Observation contributes somewhere
            else:
                classification = canonical.comparison_confidence.classification
                reason = triage_reason_by_canonical.get(canonical.canonical_observation_id)
                cat = category_by_canonical.get(canonical.canonical_observation_id)
                record["comparison_classification"] = classification.value
                record["canonical_observation_id"] = canonical.canonical_observation_id
                record["triage_reason"] = reason.value if reason else None
                record["alignment_category"] = cat
                if cat in NAMED_CATEGORIES:
                    stratum = cat
                elif classification == ComparisonClassification.CONTESTED and reason == ReviewReason.SOURCES_DISAGREE:
                    stratum = "contested_other"
                elif classification == ComparisonClassification.CORROBORATED:
                    stratum = "corroborated"
                elif reason is None:
                    stratum = "uncontested"
                else:
                    stratum = "single_source_or_low_confidence"
            stratum_counts[stratum] += 1
            stratum_pool[stratum].append(record)

        if (idx + 1) % 50 == 0:
            log(f"  ...{idx + 1}/{len(doc_refs)} docs processed")

    log(f"Failed docs: {failed_docs}")

    # -- Phase 1 report --
    pct_text = 100 * text_labeled_count / total_tlp_observations if total_tlp_observations else 0.0
    log("\n=== PHASE 1: PREVALENCE ===")
    log(f"Total tesseract_layoutparser observations: {total_tlp_observations}")
    log(f"native_label=='Text' observations: {text_labeled_count} ({pct_text:.2f}%)")
    log(f"By mapped observation_type: {dict(text_labeled_by_mapped_type)}")
    log(f"Documents with >=1 tesseract_layoutparser observation: {len(all_tlp_docs)}")
    log(f"Documents with >=1 native_label=='Text' observation: {len(text_labeled_docs)}")
    counts = list(per_doc_text_count.values())
    if counts:
        counts_sorted = sorted(counts)
        n = len(counts_sorted)
        log(
            f"Per-doc Text-labeled-observation count: min={counts_sorted[0]} "
            f"median={counts_sorted[n // 2]} max={counts_sorted[-1]} mean={sum(counts)/n:.2f}"
        )

    log("\n=== PHASE 4: DOWNSTREAM FATE (full population, not sampled) ===")
    total_strat = sum(stratum_counts.values())
    for stratum, stratum_n in stratum_counts.most_common():
        log(
            f"  {stratum}: {stratum_n} ({100*stratum_n/total_strat:.2f}%)"
            if total_strat
            else f"  {stratum}: {stratum_n}"
        )

    log("\n=== PHASE 7 Q2: PACKET-LEVEL OVERLAP (measured, not observation-level) ===")
    tot_pk = total_contested_packets[0]
    with_mistype = contested_packets_with_mistyped_source[0]
    pct_pk = 100 * with_mistype / tot_pk if tot_pk else 0.0
    log(f"Total contested (sources_disagree, multi-provider) packets, corpus-wide: {tot_pk}")
    log(f"Of those, containing >=1 tesseract_layoutparser native_label=='Text' contributing observation: {with_mistype} ({pct_pk:.2f}%)")

    prevalence_summary = {
        "total_tlp_observations": total_tlp_observations,
        "text_labeled_count": text_labeled_count,
        "text_labeled_pct": pct_text,
        "text_labeled_by_mapped_type": dict(text_labeled_by_mapped_type),
        "docs_with_tlp_observations": len(all_tlp_docs),
        "docs_with_text_labeled_observations": len(text_labeled_docs),
        "per_doc_text_count_distribution": {
            "min": counts_sorted[0] if counts else None,
            "median": counts_sorted[n // 2] if counts else None,
            "max": counts_sorted[-1] if counts else None,
            "mean": (sum(counts) / n) if counts else None,
        },
        "downstream_fate_full_population": dict(stratum_counts),
        "total_contested_packets_corpus_wide": tot_pk,
        "contested_packets_with_mistyped_source": with_mistype,
        "contested_packets_with_mistyped_source_pct": pct_pk,
        "failed_docs": failed_docs,
        "total_docs": len(doc_refs),
    }
    PREVALENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PREVALENCE_OUT.write_text(json.dumps(prevalence_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nWrote {PREVALENCE_OUT}")

    # -- Phase 2: stratified random sample --
    rng = random.Random(SAMPLE_SEED)
    sample_records = []
    for stratum, pool in stratum_pool.items():
        k = min(SAMPLE_SIZE_PER_STRATUM, len(pool))
        chosen = rng.sample(pool, k)
        for rec in chosen:
            rec = dict(rec)
            rec["stratum"] = stratum
            rec["stratum_population_size"] = len(pool)
            sample_records.append(rec)

    with SAMPLE_OUT.open("w", encoding="utf-8") as fh:
        for rec in sample_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"Wrote {len(sample_records)} sampled records to {SAMPLE_OUT}")
    log("\n=== SAMPLE SIZES DRAWN PER STRATUM ===")
    for stratum, pool in sorted(stratum_pool.items(), key=lambda kv: -len(kv[1])):
        log(f"  {stratum}: population={len(pool)}, sampled={min(SAMPLE_SIZE_PER_STRATUM, len(pool))}")


if __name__ == "__main__":
    main()
