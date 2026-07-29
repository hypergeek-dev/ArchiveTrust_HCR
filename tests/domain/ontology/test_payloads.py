from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.ontology.payloads import (
    PAYLOAD_TYPE_BY_OBSERVATION_TYPE,
    PAYLOAD_TYPES,
)
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.payloads.paragraph import IllegibleSpan, ParagraphPayload
from archivetrust.domain.ontology.types import ObservationType


def test_sixteen_payload_types_registered():
    # 13 ratified in MILESTONE1_DOMAIN_MODEL.md S5.17, plus TableCell (S5.13/S6.1 open item) and
    # HandwrittenNote (ROADMAP.md S15 Open Question 8) = 15, plus docs/htr-migration-plan.md
    # Stage 2's 5 additive HTR types (TextLine, Region, RawTranscription, ParsedTranscription,
    # NormalizedTranscription) = 20, minus docs/htr-migration-plan.md Stage 5's deletion (EXECUTED)
    # of TableCell/NamedEntity/Relationship/HandwrittenNote = 16.
    assert len(PAYLOAD_TYPES) == 16
    assert set(PAYLOAD_TYPE_BY_OBSERVATION_TYPE) == set(ObservationType)


def test_reading_order_and_document_hierarchy_are_not_payload_types():
    names = {t.value for t in ObservationType}
    assert "reading_order" not in names
    assert "document_hierarchy" not in names


def test_bounding_box_is_not_a_payload_type():
    names = {t.value for t in ObservationType}
    assert "bounding_box" not in names


@pytest.mark.parametrize("payload_cls", PAYLOAD_TYPES)
def test_every_payload_type_is_frozen(payload_cls: type[ObservationPayload]):
    assert payload_cls.model_config.get("frozen") is True


def test_graph_reference_rule_rejects_observation_id_fields():
    with pytest.raises(TypeError):

        class BadPayload(ObservationPayload):
            observation_type = ObservationType.CAPTION
            captioned_observation_id: str


def test_caption_payload_has_no_graph_reference_field():
    assert "captioned_observation_id" not in [
        f for f in PAYLOAD_TYPE_BY_OBSERVATION_TYPE[ObservationType.CAPTION].model_fields
    ]


def test_footnote_payload_has_no_graph_reference_field():
    assert "referenced_from_observation_id" not in [
        f for f in PAYLOAD_TYPE_BY_OBSERVATION_TYPE[ObservationType.FOOTNOTE].model_fields
    ]


def test_section_payload_has_no_independently_authored_label():
    assert "label" not in PAYLOAD_TYPE_BY_OBSERVATION_TYPE[ObservationType.SECTION].model_fields


def test_illegible_span_rejects_inverted_range():
    with pytest.raises(ValidationError):
        IllegibleSpan(start=10, end=2)


def test_paragraph_payload_round_trips_with_illegible_spans():
    payload = ParagraphPayload(text="ab[??]cd", illegible_spans=(IllegibleSpan(start=2, end=5),))
    restored = ParagraphPayload.model_validate(payload.model_dump())
    assert restored == payload


# `NamedEntityPayload`/`RelationshipPayload`/`TableCellPayload`/`HandwrittenNotePayload` tests
# (span-has-no-observation-reference, the Graph-Reference-Rule documented exception, table-cell-
# is-distinct-from-table, illegible-content support) were deleted in
# docs/htr-migration-plan.md Stage 5 (EXECUTED) along with those payload types themselves. The
# underlying principles -- a payload legitimately carrying an Observation reference as content
# (Graph-Reference-Rule's documented exception), and line-level illegibility support -- transfer to
# whichever future payload type needs them (`TEXT_LINE`/`RAW_TRANSCRIPTION`'s own tests, a later
# phase's job), not reimplemented here against a type that no longer exists.
