from __future__ import annotations

from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
from archivetrust.domain.ontology.payloads import ImagePayload, ParagraphPayload
from archivetrust.domain.ontology.scope import (
    ObservationScopeClassification,
    ObservationScopeMeasurement,
    ObservationScopePolicy,
    classify_observation_scope,
    measure_observation_scope,
)


def _evidence(*, text: str = "x", bounding_box: BoundingBox | None = None, supporting_metadata: dict | None = None) -> Evidence:
    return Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output=text,
        processing_stage=ProcessingStage.OCR,
        bounding_box=bounding_box,
        supporting_metadata=supporting_metadata,
    )


def _bbox(x0=0.0, y0=0.0, x1=100.0, y1=50.0) -> BoundingBox:
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, precision=Precision.PIXEL_ACCURATE)


# -- measure_observation_scope -----------------------------------------------------------------


def test_measure_counts_characters_words_sentences_for_text_payload():
    payload = ParagraphPayload(text="One sentence. Two sentences.")
    measurement = measure_observation_scope(payload, (_evidence(),))
    assert measurement.character_count == len("One sentence. Two sentences.")
    assert measurement.word_count == 4
    assert measurement.sentence_count == 2


def test_measure_non_text_payload_has_no_character_word_sentence_counts():
    payload = ImagePayload(image_type="photo")
    measurement = measure_observation_scope(payload, (_evidence(),))
    assert measurement.character_count is None
    assert measurement.word_count is None
    assert measurement.sentence_count is None


def test_measure_has_bounding_box_true_when_any_evidence_has_one():
    payload = ParagraphPayload(text="x")
    measurement = measure_observation_scope(payload, (_evidence(bounding_box=_bbox()),))
    assert measurement.has_bounding_box is True


def test_measure_has_bounding_box_false_when_no_evidence_has_one():
    payload = ParagraphPayload(text="x")
    measurement = measure_observation_scope(payload, (_evidence(bounding_box=None),))
    assert measurement.has_bounding_box is False


def test_measure_page_area_fraction_computed_when_page_dimensions_present():
    ev = _evidence(bounding_box=_bbox(x0=0, y0=0, x1=100, y1=50), supporting_metadata={"pixel_width": 1000, "pixel_height": 1000})
    measurement = measure_observation_scope(ParagraphPayload(text="x"), (ev,))
    assert measurement.page_area_fraction is not None
    assert abs(measurement.page_area_fraction - (100 * 50) / (1000 * 1000)) < 1e-9


def test_measure_page_area_fraction_none_when_page_dimensions_absent():
    ev = _evidence(bounding_box=_bbox(), supporting_metadata=None)
    measurement = measure_observation_scope(ParagraphPayload(text="x"), (ev,))
    assert measurement.page_area_fraction is None


def test_measure_is_deterministic():
    payload = ParagraphPayload(text="Repeat me exactly.")
    ev = (_evidence(bounding_box=_bbox(), supporting_metadata={"pixel_width": 800, "pixel_height": 600}),)
    first = measure_observation_scope(payload, ev)
    second = measure_observation_scope(payload, ev)
    assert first == second


# -- classify_observation_scope --------------------------------------------------------------


def _measurement(character_count: int | None) -> ObservationScopeMeasurement:
    return ObservationScopeMeasurement(
        character_count=character_count,
        word_count=None,
        sentence_count=None,
        has_bounding_box=False,
        page_area_fraction=None,
    )


def test_classify_fragment_at_default_threshold():
    assert classify_observation_scope(_measurement(40)) == ObservationScopeClassification.FRAGMENT
    assert classify_observation_scope(_measurement(41)) != ObservationScopeClassification.FRAGMENT


def test_classify_unit_at_default_thresholds():
    assert classify_observation_scope(_measurement(500)) == ObservationScopeClassification.UNIT
    assert classify_observation_scope(_measurement(41)) == ObservationScopeClassification.UNIT


def test_classify_extended_above_default_threshold():
    assert classify_observation_scope(_measurement(501)) == ObservationScopeClassification.EXTENDED
    assert classify_observation_scope(_measurement(1727)) == ObservationScopeClassification.EXTENDED


def test_classify_non_text_measurement_defaults_to_unit():
    assert classify_observation_scope(_measurement(None)) == ObservationScopeClassification.UNIT


def test_classify_with_custom_policy_changes_classification_of_same_measurement():
    """The concrete proof that facts stay stable while policy interpretation can vary: the same
    measurement classifies differently under two different policy versions."""
    measurement = _measurement(100)
    default_result = classify_observation_scope(measurement)
    custom_policy = ObservationScopePolicy(version=2, fragment_max_characters=200, unit_max_characters=1000)
    custom_result = classify_observation_scope(measurement, custom_policy)
    assert default_result == ObservationScopeClassification.UNIT
    assert custom_result == ObservationScopeClassification.FRAGMENT
    assert measurement == _measurement(100)  # unchanged
