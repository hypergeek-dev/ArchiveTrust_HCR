from __future__ import annotations

from archivetrust.domain.comparison.clustering import ClusteringBasisCode, cluster_observations
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload


def _evidence(provider: str, text: str, bbox: BoundingBox | None = None, page: int = 1) -> Evidence:
    return Evidence.create(
        provider=provider,
        provider_version="1.0",
        raw_output=text,
        processing_stage=ProcessingStage.OCR,
        page=page,
        bounding_box=bbox,
    )


def _obs(provider: str, payload, evidence: Evidence) -> Observation:
    return Observation.from_evidence(provider_id=provider, provider_version="1.0", payload=payload, evidence=(evidence,))


def _policy() -> ReconciliationPolicy:
    return ReconciliationPolicy(policy_version=1)


def test_overlapping_pixel_accurate_headings_from_two_providers_join_one_cluster():
    box_a = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    box_b = BoundingBox(x0=2, y0=1, x1=98, y1=19, precision=Precision.PIXEL_ACCURATE)
    ev_a = _evidence("docling", "Chapter 1", box_a)
    ev_b = _evidence("tesseract", "Chapter I", box_b)
    obs_a = _obs("docling", HeadingPayload(text="Chapter 1", level=1), ev_a)
    obs_b = _obs("tesseract", HeadingPayload(text="Chapter I", level=1), ev_b)

    evidence_by_id = {ev_a.evidence_id: ev_a, ev_b.evidence_id: ev_b}
    clusters = cluster_observations((obs_a, obs_b), evidence_by_id, _policy())

    assert len(clusters) == 1
    assert set(clusters[0].member_observation_ids) == {obs_a.observation_id, obs_b.observation_id}


def test_disjoint_pixel_accurate_headings_form_separate_clusters():
    box_a = BoundingBox(x0=0, y0=0, x1=50, y1=20, precision=Precision.PIXEL_ACCURATE)
    box_b = BoundingBox(x0=500, y0=500, x1=550, y1=520, precision=Precision.PIXEL_ACCURATE)
    ev_a = _evidence("docling", "Chapter 1", box_a)
    ev_b = _evidence("docling", "Chapter 2", box_b)
    obs_a = _obs("docling", HeadingPayload(text="Chapter 1", level=1), ev_a)
    obs_b = _obs("docling", HeadingPayload(text="Chapter 2", level=1), ev_b)

    evidence_by_id = {ev_a.evidence_id: ev_a, ev_b.evidence_id: ev_b}
    clusters = cluster_observations((obs_a, obs_b), evidence_by_id, _policy())

    assert len(clusters) == 2


def test_different_types_never_share_a_cluster_even_when_co_located():
    box = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    ev_heading = _evidence("docling", "Chapter 1", box)
    ev_paragraph = _evidence("docling", "Body text", box)
    heading = _obs("docling", HeadingPayload(text="Chapter 1", level=1), ev_heading)
    paragraph = _obs("docling", ParagraphPayload(text="Body text"), ev_paragraph)

    evidence_by_id = {ev_heading.evidence_id: ev_heading, ev_paragraph.evidence_id: ev_paragraph}
    clusters = cluster_observations((heading, paragraph), evidence_by_id, _policy())

    assert len(clusters) == 2
    assert {c.observation_type for c in clusters} == {heading.observation_type, paragraph.observation_type}


def test_single_provider_slot_is_a_singleton_cluster():
    box = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    ev = _evidence("docling", "Chapter 1", box)
    obs = _obs("docling", HeadingPayload(text="Chapter 1", level=1), ev)
    clusters = cluster_observations((obs,), {ev.evidence_id: ev}, _policy())
    assert len(clusters) == 1
    assert clusters[0].member_observation_ids == (obs.observation_id,)


def test_coarse_geometry_uses_ordinal_position_not_iou():
    # A Qwen heading (coarse/no geometry) at the same ordinal position as a Docling heading joins
    # via ordinal position (S2 step 2), never via IoU (there is no usable box to compute it from).
    box = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    ev_docling = _evidence("docling", "Chapter 1", box)
    ev_qwen = _evidence("qwen2.5-vl", "Chapter 1", bbox=None)
    obs_docling = _obs("docling", HeadingPayload(text="Chapter 1", level=1), ev_docling)
    obs_qwen = _obs("qwen2.5-vl", HeadingPayload(text="Chapter 1", level=1), ev_qwen)
    evidence_by_id = {ev_docling.evidence_id: ev_docling, ev_qwen.evidence_id: ev_qwen}
    clusters = cluster_observations((obs_docling, obs_qwen), evidence_by_id, _policy())
    assert len(clusters) == 1
    edge = clusters[0].affinity_edges[0]
    assert "ordinal" in edge.basis


def test_whole_page_transcript_does_not_fuse_with_one_of_many_line_fragments():
    """Regression test for the Review Packet Integrity Audit (2026-07-14).

    Real values from the 2026-07-13 benchmark corpus, workspace
    46e94f15d73347c2be79f1527e646c11, document archive_object_002964d6c0cf4da9a3002f829142de44,
    page 1: a source with no layout detection (real corpus example: paddleocr-vl, via
    `PlainTextDecoder`) emits one whole-page transcript per page with no bounding box, while a
    layout-aware source (real corpus example: tesseract_layoutparser) emits many small
    pixel-accurate paragraph/line fragments on the same page. Confirmed by direct replay against
    the frozen corpus: the pre-fix clustering (`sorted(observations, key=observation_id)` used as
    an "ordinal position" proxy for reading order) fused the 1649-character whole-page transcript
    with two arbitrary ~50-character fragments into one Canonical Observation and one contested
    Review Packet -- a reviewer facing "hundreds of words different" for what a tiny highlighted
    box suggested was a small OCR disagreement (the reported symptom this audit investigated).
    Text content is reproduced verbatim except for Swedish diacritics, mangled identically to the
    corpus's own OCR/encoding artifacts, which are irrelevant to this test and ASCII-folded here.
    """
    box_frag_1 = BoundingBox(x0=279.0, y0=273.0, x1=898.0, y1=301.0, precision=Precision.PIXEL_ACCURATE)
    box_frag_2 = BoundingBox(x0=279.0, y0=312.0, x1=800.0, y1=372.0, precision=Precision.PIXEL_ACCURATE)

    ev_page_transcript = _evidence(
        "paddleocr-vl",
        "Kommunstyrelsens arbetsutskott\nProtokollsutdrag\nSammantradesdatum 2020-09-14\n"
        "para 305 Remittering av forslag till reglemente for styrelser och namnder i Lunds "
        "kommun\n" + ("..." * 400),  # real candidate was 1649 chars; length is what matters here
        bbox=None,
    )
    ev_frag_1 = _evidence("tesseract_layoutparser", "para 305 Remittering av forslag till reglemente for", box_frag_1)
    ev_frag_2 = _evidence(
        "tesseract_layoutparser", "styrelser och namnder i Lunds kommun Dnr KS 2019/0643", box_frag_2
    )

    obs_page_transcript = _obs("paddleocr-vl", ParagraphPayload(text=ev_page_transcript.raw_output), ev_page_transcript)
    obs_frag_1 = _obs("tesseract_layoutparser", ParagraphPayload(text=ev_frag_1.raw_output), ev_frag_1)
    obs_frag_2 = _obs("tesseract_layoutparser", ParagraphPayload(text=ev_frag_2.raw_output), ev_frag_2)
    # The real page had 16 tesseract fragments and 9 docling paragraphs contributing to the
    # ambiguity; a third same-provider fragment is enough to reproduce "provider contributed more
    # than one same-type Observation to this page" without transcribing all 16.
    box_frag_3 = BoundingBox(x0=279.0, y0=400.0, x1=800.0, y1=460.0, precision=Precision.PIXEL_ACCURATE)
    ev_frag_3 = _evidence("tesseract_layoutparser", "Beslut", box_frag_3)
    obs_frag_3 = _obs("tesseract_layoutparser", ParagraphPayload(text=ev_frag_3.raw_output), ev_frag_3)

    evidence_by_id = {
        ev_page_transcript.evidence_id: ev_page_transcript,
        ev_frag_1.evidence_id: ev_frag_1,
        ev_frag_2.evidence_id: ev_frag_2,
        ev_frag_3.evidence_id: ev_frag_3,
    }
    clusters = cluster_observations(
        (obs_page_transcript, obs_frag_1, obs_frag_2, obs_frag_3), evidence_by_id, _policy()
    )

    # Every observation lands in its own singleton cluster: the whole-page transcript is never
    # guessed into correspondence with an arbitrary one of the fragments.
    assert len(clusters) == 4
    for cluster in clusters:
        assert len(cluster.member_observation_ids) == 1


def test_scope_mismatch_prevents_extended_transcript_from_joining_unit_paragraph():
    box = BoundingBox(x0=0, y0=0, x1=100, y1=80, precision=Precision.PIXEL_ACCURATE)
    unit_text = "A normal paragraph-sized observation."
    extended_text = " ".join(["whole page transcript"] * 80)
    ev_unit = _evidence("docling", unit_text, box)
    ev_extended = _evidence("paddleocr-vl", extended_text, box)
    obs_unit = _obs("docling", ParagraphPayload(text=unit_text), ev_unit)
    obs_extended = _obs("paddleocr-vl", ParagraphPayload(text=extended_text), ev_extended)
    evidence_by_id = {ev_unit.evidence_id: ev_unit, ev_extended.evidence_id: ev_extended}

    clusters = cluster_observations((obs_unit, obs_extended), evidence_by_id, _policy())

    assert len(clusters) == 2
    assert {clusters[0].member_observation_ids, clusters[1].member_observation_ids} == {
        (obs_unit.observation_id,),
        (obs_extended.observation_id,),
    }
    assert all(ClusteringBasisCode.SCOPE_MISMATCH not in {e.basis_code for e in c.affinity_edges} for c in clusters)


def test_scope_gate_can_be_disabled_for_counterfactual_replay():
    box = BoundingBox(x0=0, y0=0, x1=100, y1=80, precision=Precision.PIXEL_ACCURATE)
    unit_text = "A normal paragraph-sized observation."
    extended_text = " ".join(["whole page transcript"] * 80)
    ev_unit = _evidence("docling", unit_text, box)
    ev_extended = _evidence("paddleocr-vl", extended_text, box)
    obs_unit = _obs("docling", ParagraphPayload(text=unit_text), ev_unit)
    obs_extended = _obs("paddleocr-vl", ParagraphPayload(text=extended_text), ev_extended)
    evidence_by_id = {ev_unit.evidence_id: ev_unit, ev_extended.evidence_id: ev_extended}
    policy = ReconciliationPolicy(policy_version=1, enforce_scope_compatible_alignment=False)

    clusters = cluster_observations((obs_unit, obs_extended), evidence_by_id, policy)

    assert len(clusters) == 1
    assert set(clusters[0].member_observation_ids) == {
        obs_unit.observation_id,
        obs_extended.observation_id,
    }


def test_clustering_is_deterministic_regardless_of_input_order():
    box_a = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    box_b = BoundingBox(x0=2, y0=1, x1=98, y1=19, precision=Precision.PIXEL_ACCURATE)
    ev_a = _evidence("docling", "Chapter 1", box_a)
    ev_b = _evidence("tesseract", "Chapter I", box_b)
    obs_a = _obs("docling", HeadingPayload(text="Chapter 1", level=1), ev_a)
    obs_b = _obs("tesseract", HeadingPayload(text="Chapter I", level=1), ev_b)
    evidence_by_id = {ev_a.evidence_id: ev_a, ev_b.evidence_id: ev_b}

    first = cluster_observations((obs_a, obs_b), evidence_by_id, _policy())
    second = cluster_observations((obs_b, obs_a), evidence_by_id, _policy())
    assert [c.member_observation_ids for c in first] == [c.member_observation_ids for c in second]
