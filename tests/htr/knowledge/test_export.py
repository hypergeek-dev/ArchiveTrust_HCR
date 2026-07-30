"""Knowledge export: format correctness, round-trip, and the status-visibility guarantee.

The centrepiece is `test_status_word_is_adjacent_to_every_statement_in_every_format` and its three
siblings. The follow-up's constraint is *"do not export candidate or disputed findings as established
conclusions without preserving their status"*, and the failure mode it guards against is not a missing
field -- it is a status that is present but separable from the claim. So the assertion is positional,
per finding, per format: the status word must appear within `MAX_STATUS_DISTANCE` characters of **every**
occurrence of that finding's statement, in all four rendered formats. A single status banner at the top
of a file passes no test here.

The redaction tests (`test_redaction_*`) exercise `htr/knowledge/redaction.py` against a real
`Disputed` finding carrying a real reviewer name in three different places, and assert both halves of
the requirement: the identity is gone from the export, and the status/statement/limitations/history
survived it untouched.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from archivetrust.application.htr_journal import HtrJournal
from archivetrust.htr.knowledge.export import (
    EVIDENCE_BUNDLE_SCHEMA,
    FINDING_CSV_COLUMNS,
    KNOWLEDGE_EXPORT_SCHEMA,
    MAX_STATUS_DISTANCE,
    OBSERVATION_CSV_COLUMNS,
    REQUIRED_FINDING_EXPORT_FIELDS,
    KnowledgeExport,
    UnsupportedExportFormatError,
    collect_knowledge,
    findings_to_csv,
    knowledge_from_json,
    knowledge_to_json,
    knowledge_to_markdown,
    observations_to_csv,
    status_qualified_statement,
    write_evidence_bundle,
    write_knowledge,
)
from archivetrust.htr.knowledge.lifecycle import transition_finding_status
from archivetrust.htr.knowledge.models import (
    ContradictionSourceKind,
    ContradictoryEvidence,
    EvidenceReference,
    EvidenceReferenceKind,
    FindingConfidence,
    FindingStatus,
    ObservationConfidence,
    ObservationType,
    ResearchFinding,
    ResearchObservation,
    ResearchScope,
)
from archivetrust.htr.knowledge.redaction import (
    DEMONSTRATION_SALT,
    IdentityLedger,
    identity_values,
    probable_component_identity,
    probable_human_identities,
    pseudonymise,
    pseudonymise_mapping,
    redact_review_record_identities,
    strip_reviewer_identities,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"

AT = "2026-07-30T04:00:00+00:00"
REVIEWER = "erika.lindqvist@example.org"
"""A deliberately email-shaped identity, so the redaction tests are testing against the worst case the
brief names ("a real username/email vs. an opaque reviewer id") rather than against a handle that might
be mistaken for an opaque id."""


# -- Fixtures: one Candidate finding and one Disputed finding, both real-shaped -------------------


def _scope(*, run_id: str = "experiment_run_fixture_a") -> ResearchScope:
    return ResearchScope(
        experiment_id="experiment_fixture",
        experiment_version_id="experiment_version_fixture",
        experiment_run_ids=(run_id,),
        unit_of_analysis="line_crop",
        covered_unit_ids=("input_crop_fixture_1",),
        dataset_id="dataset_fixture",
        dataset_version_id="dataset_version_fixture",
        method_ids=("satrn", "florence2_htr"),
        model_version_ids=("a40c7093", "994f47e8"),
        normalization_profile="evaluation.metrics.normalize_text v1",
        ground_truth_ref="ground_truth_fixture_1",
    )


def _observation(title: str, *, reference_id: str) -> ResearchObservation:
    return ResearchObservation.create(
        observation_type=ObservationType.UNEXPECTED_METHOD_DISAGREEMENT,
        title=title,
        description=f"Recorded fact backing {title}.",
        scope=_scope(),
        supporting_evidence=(
            EvidenceReference(
                kind=EvidenceReferenceKind.METRIC_RESULT,
                reference_id=reference_id,
                stream="htr_research_events.jsonl",
                note="the CER this observation quotes",
            ),
        ),
        source_experiment_id="experiment_fixture",
        source_experiment_run_id="experiment_run_fixture_a",
        author_or_source_component="tests.htr.knowledge.test_export",
        creation_timestamp=AT,
        observation_confidence=ObservationConfidence.HIGH,
        affected_method="satrn",
        affected_model_version="a40c7093",
    )


CANDIDATE_STATEMENT = (
    "On the single fixture line, method A produced a lower character error rate than method B."
)
DISPUTED_STATEMENT = (
    "The peak GPU memory measurement for method B on this fixture is reproducible across sessions."
)


def _candidate_finding(observation: ResearchObservation) -> ResearchFinding:
    return ResearchFinding.create(
        statement=CANDIDATE_STATEMENT,
        scope=_scope(),
        supporting_observations=(observation.observation_id,),
        limitations=(
            "N=1: one line crop, one run, one checkpoint per method.",
            "No general ranking of methods can be inferred from this scope.",
        ),
        confidence_level=FindingConfidence.LOW,
        author="tests.htr.knowledge.test_export",
        creation_date=AT,
        supporting_experiments=("experiment_fixture",),
        supporting_metrics=("metric_result_fixture_cer",),
        affected_methods=("satrn", "florence2_htr"),
        affected_model_versions=("a40c7093", "994f47e8"),
        affected_datasets=("dataset_fixture",),
        affected_dataset_versions=("dataset_version_fixture",),
    )


def _disputed_finding(observation: ResearchObservation) -> ResearchFinding:
    """Reaches `Disputed` the only way the domain permits: `Candidate -> Under review -> Disputed`,
    each step through `transition_finding_status` with a named reviewer and a recorded contradiction.
    Constructing one directly is refused by the model, which is the point."""
    finding = ResearchFinding.create(
        statement=DISPUTED_STATEMENT,
        scope=_scope(),
        supporting_observations=(observation.observation_id,),
        limitations=("The two measurements come from two sessions with different recorded state.",),
        confidence_level=FindingConfidence.LOW,
        author="tests.htr.knowledge.test_export",
        creation_date=AT,
        affected_methods=("florence2_htr",),
        affected_model_versions=("994f47e8",),
    )
    finding = transition_finding_status(
        finding,
        FindingStatus.UNDER_REVIEW,
        reviewer=REVIEWER,
        reasoning="Two committed records disagree; examining rather than assuming.",
        at=AT,
    )
    return transition_finding_status(
        finding,
        FindingStatus.DISPUTED,
        reviewer=REVIEWER,
        reasoning="The two measurements differ by a factor of ~3.3.",
        at=AT,
        contradiction=ContradictoryEvidence.create(
            source_kind=ContradictionSourceKind.EXTERNAL_DOCUMENT,
            source_id="providers/florence2_htr/README.md",
            description="The adapter README documents a different peak-GPU-memory figure.",
            recorded_by=REVIEWER,
            recorded_at=AT,
        ),
    )


@pytest.fixture
def export() -> KnowledgeExport:
    observation_a = _observation("relative error rates", reference_id="metric_result_fixture_cer")
    observation_b = _observation("gpu memory variability", reference_id="metric_result_fixture_gpu")
    return KnowledgeExport(
        observations=(observation_a, observation_b),
        findings=(_candidate_finding(observation_a), _disputed_finding(observation_b)),
        generated_at=AT,
        source_description="synthetic fixture for tests/htr/knowledge/test_export.py",
    )


# -- The status-visibility guarantee --------------------------------------------------------------


def _assert_status_adjacent(text: str, *, statement: str, status: str, fmt: str) -> None:
    """Every occurrence of `statement` in `text` must have `status` within `MAX_STATUS_DISTANCE`
    characters on one side or the other.

    Checking *every* occurrence, not the first, is deliberate: a format that quoted the statement once
    with its status and again without it (a summary table, a footer) would still let a reader lift the
    unqualified copy.
    """
    assert statement in text, f"{fmt}: statement absent entirely"
    start = 0
    occurrences = 0
    while (found := text.find(statement, start)) != -1:
        occurrences += 1
        window = text[
            max(0, found - MAX_STATUS_DISTANCE) : found + len(statement) + MAX_STATUS_DISTANCE
        ]
        assert status in window, (
            f"{fmt}: occurrence {occurrences} of the statement at offset {found} has no "
            f"{status!r} within {MAX_STATUS_DISTANCE} characters. Window:\n{window!r}"
        )
        start = found + 1
    assert occurrences >= 1


def test_status_word_is_adjacent_to_every_statement_in_every_format(export, tmp_path) -> None:
    """The follow-up's explicit requirement, asserted per finding per format.

    One `Candidate` finding and one `Disputed` finding, four formats, every occurrence of every
    statement. The bundle is checked as the concatenation of its manifest, its evidence index and its
    per-finding files -- i.e. a consumer reading any one of them cannot obtain a statement without a
    status.
    """
    bundle_dir = tmp_path / "bundle"
    write_evidence_bundle(export, bundle_dir)
    bundle_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(bundle_dir.rglob("*.json"))
    )

    formats = {
        "json": knowledge_to_json(export),
        "csv": findings_to_csv(export),
        "markdown": knowledge_to_markdown(export),
        "bundle": bundle_text,
    }

    for fmt, text in formats.items():
        _assert_status_adjacent(text, statement=CANDIDATE_STATEMENT, status="Candidate", fmt=fmt)
        _assert_status_adjacent(text, statement=DISPUTED_STATEMENT, status="Disputed", fmt=fmt)


def test_status_appears_more_than_once_per_finding_and_not_only_in_a_header(export) -> None:
    """"Not just once at the top of the file." Each status word must occur at least twice in the
    Markdown export *below* the status-summary table, which is where a single top-of-file banner would
    live."""
    markdown = knowledge_to_markdown(export)
    body = markdown.split("## Findings", 1)[1]
    assert body.count("Candidate") >= 2
    assert body.count("Disputed") >= 2


def test_a_disputed_finding_is_never_rendered_with_a_supported_family_word(export) -> None:
    """The specific misreading the constraint exists to prevent: nothing in any format may describe
    these two findings with a supported-family status they do not hold."""
    texts = (knowledge_to_json(export), findings_to_csv(export), knowledge_to_markdown(export))
    for text in texts:
        assert "Provisionally supported" not in text
        # "Supported" must not appear at all: neither finding has ever held it, so any occurrence
        # would be this exporter inventing a status.
        assert "Supported" not in text


def test_markdown_puts_the_status_in_the_heading_and_a_badge_not_a_footnote(export) -> None:
    markdown = knowledge_to_markdown(export)
    assert "### 1. Candidate — `" in markdown
    assert "### 2. Disputed — `" in markdown
    assert "**Status: Candidate**" in markdown
    assert "**Status: Disputed**" in markdown
    # The blockquote carrying the claim carries the status inside it too.
    assert f"> [Candidate] {CANDIDATE_STATEMENT}" in markdown
    assert f"> [Disputed] {DISPUTED_STATEMENT}" in markdown


def test_csv_status_columns_precede_the_statement_column() -> None:
    """Positional, not just present: a spreadsheet reader must see the status before the claim, and
    hiding `review_status` must not strip the status from the row."""
    assert FINDING_CSV_COLUMNS.index("review_status") < FINDING_CSV_COLUMNS.index("statement")
    assert FINDING_CSV_COLUMNS.index("status_qualified_statement") < FINDING_CSV_COLUMNS.index(
        "statement"
    )


# -- Required content ------------------------------------------------------------------------------


def test_every_exported_finding_carries_every_required_field(export) -> None:
    payload = json.loads(knowledge_to_json(export))
    assert len(payload["findings"]) == 2
    for finding in payload["findings"]:
        missing = [f for f in REQUIRED_FINDING_EXPORT_FIELDS if f not in finding]
        assert not missing, f"missing required export fields: {missing}"


def test_required_fields_are_all_csv_columns_too(export) -> None:
    """The CSV is lossy but not *selectively* lossy: every required field has a column, under its own
    name or an explicitly-mapped one."""
    csv_text = findings_to_csv(export)
    header = csv_text.splitlines()[0]
    mapped = {
        "scope": "scope_description",
        "evidence_references": "evidence_references",
        "contradictory_evidence": "contradictions",
    }
    for field in REQUIRED_FINDING_EXPORT_FIELDS:
        column = mapped.get(field, field)
        assert column in header, f"{field!r} has no CSV column (looked for {column!r})"


def test_finding_evidence_references_are_resolved_through_its_observations(export) -> None:
    payload = json.loads(knowledge_to_json(export))
    candidate = next(f for f in payload["findings"] if f["review_status"] == "Candidate")
    assert candidate["evidence_references"], "a finding with a resolvable observation has references"
    reference = candidate["evidence_references"][0]
    assert reference["reference_id"] == "metric_result_fixture_cer"
    assert reference["via_observation_id"] == candidate["supporting_observations"][0]
    assert candidate["unresolved_supporting_observations"] == []


def test_an_unresolvable_supporting_observation_is_reported_not_dropped() -> None:
    """An export assembled from a partial replay must not silently show less support than the finding
    claims."""
    observation = _observation("present", reference_id="metric_result_present")
    finding = _candidate_finding(observation).model_copy(
        update={"supporting_observations": (observation.observation_id, "research_observation_absent")}
    )
    partial = KnowledgeExport(
        observations=(observation,), findings=(finding,), generated_at=AT
    )
    payload = json.loads(knowledge_to_json(partial))
    assert payload["findings"][0]["unresolved_supporting_observations"] == [
        "research_observation_absent"
    ]


def test_observation_csv_keeps_the_unverified_hypothesis_in_its_own_column() -> None:
    observation = _observation("gpu", reference_id="metric_result_gpu").model_copy(
        update={"unverified_hypothesis": "Possibly the allocator's caching state. Not measured."}
    )
    csv_text = observations_to_csv(
        KnowledgeExport(
            observations=(observation,),
            findings=(),
            generated_at=AT,
        )
    )
    assert "unverified_hypothesis" in OBSERVATION_CSV_COLUMNS
    assert "Not measured." in csv_text
    # And it is not merged into the description cell.
    assert "Recorded fact backing gpu. Possibly" not in csv_text


# -- Round trip and dispatch -----------------------------------------------------------------------


def test_json_round_trips_the_findings_and_observations(export) -> None:
    restored = knowledge_from_json(knowledge_to_json(export))
    assert restored.findings == export.findings
    assert restored.observations == export.observations
    assert restored.generated_at == export.generated_at


def test_json_refuses_a_foreign_schema(export) -> None:
    payload = json.loads(knowledge_to_json(export))
    payload["export_schema"] = "archivetrust.research_knowledge.v99"
    with pytest.raises(ValueError, match="Unsupported research knowledge export schema"):
        knowledge_from_json(json.dumps(payload))


@pytest.mark.parametrize("unsupported", ["html", "pdf", "PDF", "xlsx"])
def test_unsupported_formats_refuse_with_a_reason(export, tmp_path, unsupported) -> None:
    with pytest.raises(UnsupportedExportFormatError):
        write_knowledge(export, tmp_path / "out", export_format=unsupported)


def test_write_knowledge_csv_also_writes_the_observation_table(export, tmp_path) -> None:
    path = write_knowledge(export, tmp_path / "knowledge.csv", export_format="csv")
    assert path.exists()
    sibling = tmp_path / "knowledge_observations.csv"
    assert sibling.exists()
    assert sibling.read_text(encoding="utf-8").splitlines()[0].startswith("observation_id,")


def test_csv_uses_lf_line_endings_on_every_platform(export) -> None:
    assert "\r\n" not in findings_to_csv(export)
    assert "\r\n" not in observations_to_csv(export)


# -- The evidence bundle ---------------------------------------------------------------------------


def test_bundle_manifest_digests_verify_against_the_files_it_lists(export, tmp_path) -> None:
    import hashlib

    manifest_path = write_evidence_bundle(export, tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_schema"] == EVIDENCE_BUNDLE_SCHEMA
    assert manifest["export_schema"] == KNOWLEDGE_EXPORT_SCHEMA
    assert manifest["files"], "a bundle manifest with no files is not a bundle"
    for entry in manifest["files"]:
        target = (tmp_path / "bundle" / entry["path"]).resolve()
        assert target.exists(), entry["path"]
        payload = target.read_bytes()
        assert entry["size_bytes"] == len(payload)
        assert entry["digest"] == hashlib.sha256(payload).hexdigest()


def test_bundle_status_index_carries_a_status_for_every_finding(export, tmp_path) -> None:
    manifest_path = write_evidence_bundle(export, tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = {row["finding_id"]: row for row in manifest["status_index"]}
    assert len(index) == len(export.findings)
    for finding in export.findings:
        row = index[finding.finding_id]
        assert row["review_status"] == finding.review_status.value
        assert row["status_qualified_statement"] == status_qualified_statement(finding)


def test_bundle_evidence_index_is_navigable_in_both_directions(export, tmp_path) -> None:
    write_evidence_bundle(export, tmp_path / "bundle")
    index = json.loads((tmp_path / "bundle" / "evidence_index.json").read_text(encoding="utf-8"))
    by_reference = {row["reference_id"]: row for row in index["by_reference"]}
    assert "metric_result_fixture_cer" in by_reference
    assert by_reference["metric_result_fixture_cer"]["cited_by_observations"]
    assert by_reference["metric_result_fixture_cer"]["cited_by_findings"]
    for row in index["by_finding"]:
        assert row["review_status"] in {"Candidate", "Disputed"}


def test_bundle_is_byte_reproducible_for_the_same_export(export, tmp_path) -> None:
    """No clock read anywhere in the bundle writer -- `generated_at` is the caller's. This is what
    lets a committed bundle be diffed meaningfully."""
    first = write_evidence_bundle(export, tmp_path / "a")
    second = write_evidence_bundle(export, tmp_path / "b")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


# -- Reviewer-identity redaction (Part B) ----------------------------------------------------------


def test_component_identities_are_distinguished_from_human_ones() -> None:
    assert probable_component_identity("htr.knowledge.baseline_knowledge")
    assert probable_component_identity("tests.htr.knowledge.test_export")
    assert not probable_component_identity("hypergeek-dev")
    assert not probable_component_identity(REVIEWER)


def test_identity_values_finds_every_identity_bearing_field(export) -> None:
    values = identity_values(export)
    assert REVIEWER in values  # reviewer, revision actor, contradiction recorded_by
    assert "tests.htr.knowledge.test_export" in values
    assert probable_human_identities(export) == frozenset({REVIEWER})


def test_redaction_removes_the_reviewer_identity_from_every_field_and_every_format(
    export, tmp_path
) -> None:
    redacted, ledger = strip_reviewer_identities(
        export, human_identities={REVIEWER}, salt=DEMONSTRATION_SALT
    )
    pseudonym = pseudonymise(REVIEWER, salt=DEMONSTRATION_SALT)

    bundle_dir = tmp_path / "bundle"
    write_evidence_bundle(redacted, bundle_dir)
    texts = {
        "json": knowledge_to_json(redacted),
        "csv": findings_to_csv(redacted),
        "markdown": knowledge_to_markdown(redacted),
        "bundle": "\n".join(p.read_text(encoding="utf-8") for p in sorted(bundle_dir.rglob("*"))
                            if p.is_file()),
    }
    for name, text in texts.items():
        assert REVIEWER not in text, f"{name}: the real reviewer identity survived redaction"
        assert pseudonym in text, f"{name}: the pseudonym is absent, so attribution was lost entirely"

    # The ledger -- the internal accountability id -- is the only place the mapping exists, and it is
    # not in any exported text.
    assert ledger.real_identity(pseudonym) == REVIEWER
    for text in texts.values():
        assert ledger.salt not in text


def test_redaction_preserves_status_statement_limitations_and_history(export) -> None:
    """The two requirements are not in tension, and this is where that is demonstrated."""
    redacted, _ledger = strip_reviewer_identities(
        export, human_identities={REVIEWER}, salt=DEMONSTRATION_SALT
    )
    for before, after in zip(export.findings, redacted.findings, strict=True):
        assert after.review_status is before.review_status
        assert after.statement == before.statement
        assert after.limitations == before.limitations
        assert after.scope == before.scope
        assert after.supporting_observations == before.supporting_observations
        assert len(after.revision_history) == len(before.revision_history)
        for old, new in zip(before.revision_history, after.revision_history, strict=True):
            assert new.from_status is old.from_status
            assert new.to_status is old.to_status
            assert new.reasoning == old.reasoning
        assert len(after.contradictory_evidence) == len(before.contradictory_evidence)


def test_redaction_leaves_component_authors_intact(export) -> None:
    redacted, _ledger = strip_reviewer_identities(
        export, human_identities={REVIEWER}, salt=DEMONSTRATION_SALT
    )
    assert redacted.observations[0].author_or_source_component == "tests.htr.knowledge.test_export"
    assert redacted.findings[0].author == "tests.htr.knowledge.test_export"


def test_redacted_export_says_of_itself_that_it_is_redacted(export) -> None:
    redacted, _ledger = strip_reviewer_identities(
        export, human_identities={REVIEWER}, salt=DEMONSTRATION_SALT
    )
    assert "pseudonyms" in (redacted.source_description or "")
    assert "pseudonyms" in knowledge_to_markdown(redacted)


def test_pseudonyms_are_stable_within_a_salt_and_differ_across_salts() -> None:
    assert pseudonymise(REVIEWER, salt="a") == pseudonymise(REVIEWER, salt="a")
    assert pseudonymise(REVIEWER, salt="a") != pseudonymise(REVIEWER, salt="b")
    assert pseudonymise(REVIEWER, salt="a") != pseudonymise("someone.else@example.org", salt="a")


def test_blind_review_records_redact_against_the_same_ledger() -> None:
    """The latent half of the PII finding: `ReviewSubmission.reviewer_ref` is a bare `str`, so the
    redaction has to work on real blind-review records too -- and produce the *same* pseudonym as the
    knowledge export, or cross-referencing the two is impossible for an external reader."""
    from archivetrust.review.blind_review.assignment import create_blind_review_pair
    from archivetrust.review.htr_models import Adjudication, ReviewSubmission

    ledger = pseudonymise_mapping({REVIEWER, "second.reviewer@example.org"}, salt=DEMONSTRATION_SALT)
    assignment_a, assignment_b = create_blind_review_pair(
        target_ref="text_line_fixture_1",
        reviewer_a_ref=REVIEWER,
        reviewer_b_ref="second.reviewer@example.org",
        convention_id="convention_fixture",
        convention_version=1,
        assigned_at=AT,
    )
    submission = ReviewSubmission.create(
        assignment_id=assignment_a.assignment_id,
        reviewer_ref=REVIEWER,
        submitted_value="bekiendt.",
        submitted_at=AT,
    )
    adjudication = Adjudication.create(
        agreement_result_id="agreement_result_fixture",
        adjudicator_ref=REVIEWER,
        resolved_value="bekiendt.",
        rationale="Reviewer A's reading matches the ink.",
        adjudicated_at=AT,
    )

    redacted = redact_review_record_identities(
        (assignment_a, assignment_b, submission, adjudication), ledger=ledger
    )
    serialized = json.dumps([r.model_dump(mode="json") for r in redacted])
    assert REVIEWER not in serialized
    assert "second.reviewer@example.org" not in serialized
    expected = pseudonymise(REVIEWER, salt=DEMONSTRATION_SALT)
    assert expected in serialized
    # Same person, same pseudonym, across the knowledge layer and the review layer.
    assert redacted[2].reviewer_ref == expected
    assert redacted[3].adjudicator_ref == expected
    # The submitted transcription itself is untouched -- redaction is about who, never about what.
    assert redacted[2].submitted_value == "bekiendt."


def test_ledger_is_not_reversible_without_it() -> None:
    """States the honest scope of the mechanism: the pseudonym alone carries no identity, and the
    ledger is the only mapping. (Pseudonymisation, not anonymisation -- see the module docstring.)"""
    ledger = IdentityLedger(salt="secret", entries=(("reviewer_deadbeefdeadbeef", REVIEWER),))
    assert ledger.real_identity("reviewer_deadbeefdeadbeef") == REVIEWER
    assert ledger.real_identity("reviewer_0000000000000000") is None


# -- Against the real committed baseline knowledge -------------------------------------------------


def test_export_of_the_real_committed_knowledge_log_carries_every_real_status(tmp_path) -> None:
    """Not a synthetic fixture: replays the two committed logs and exports what they actually contain.

    Copies both to `tmp_path` first, for the reason
    `tests/htr/knowledge/test_baseline_knowledge.py` states -- `FileTelemetrySink` is append-capable
    and a test must not mutate a committed research artifact.
    """
    work = tmp_path / "logs"
    work.mkdir()
    for name in (
        "htr_research_events.jsonl",
        "htr_knowledge_events.jsonl",
        "htr_knowledge_feedback_events.jsonl",
    ):
        shutil.copy2(BASELINE_DIR / name, work / name)

    events = []
    for name in (
        "htr_research_events.jsonl",
        "htr_knowledge_events.jsonl",
        "htr_knowledge_feedback_events.jsonl",
    ):
        events.extend(FileTelemetrySink(work / name).all_events())

    store = HtrJournal().replay(tuple(events))
    export = collect_knowledge(
        store, generated_at=AT, source_description="the committed baseline logs"
    )

    assert len(export.observations) == 5
    assert len(export.findings) == 5
    statuses = {f.review_status.value for f in export.findings}
    assert statuses == {"Candidate", "Provisionally supported", "Disputed"}

    markdown = knowledge_to_markdown(export)
    for finding in export.findings:
        _assert_status_adjacent(
            markdown,
            statement=finding.statement,
            status=finding.review_status.value,
            fmt="markdown/real",
        )


def test_the_committed_knowledge_export_matches_the_committed_logs(tmp_path) -> None:
    """The committed artifact under `docs/experiments/baseline-comparison/knowledge-export/` is
    regenerable, and this asserts it has not drifted from the logs it was rendered from.

    Renders in memory and compares against the committed bytes rather than rewriting them: a test must
    not mutate a committed research artifact. Regenerate deliberately with
    `PYTHONPATH=src .venv/Scripts/python.exe scripts/export_baseline_knowledge.py`.

    This is the Part A counterpart of the capability matrix's drift test -- an exported research
    artifact that silently stops matching its source log is exactly as misleading as a stale capability
    table, and the `generated_at` being a fixed constant rather than a clock read is what makes the
    comparison possible at all.
    """
    from archivetrust.htr.knowledge.baseline_knowledge import EXTRACTED_AT
    from archivetrust.htr.knowledge.redaction import (
        DEMONSTRATION_SALT,
        probable_human_identities,
    )

    export_dir = BASELINE_DIR / "knowledge-export"
    if not export_dir.exists():  # pragma: no cover - the artifact is committed
        pytest.skip(f"{export_dir} is not present in this checkout")

    work = tmp_path / "logs"
    work.mkdir()
    log_names = (
        "htr_research_events.jsonl",
        "htr_knowledge_events.jsonl",
        "htr_knowledge_feedback_events.jsonl",
    )
    for name in log_names:
        shutil.copy2(BASELINE_DIR / name, work / name)
    events = []
    for name in log_names:
        events.extend(FileTelemetrySink(work / name).all_events())

    store = HtrJournal().replay(tuple(events))
    source_description = (
        "Replayed from the committed append-only logs "
        "docs/experiments/baseline-comparison/{htr_research_events, htr_knowledge_events, "
        "htr_knowledge_feedback_events}.jsonl (2026-07-30 baseline run)"
    )
    rendered = collect_knowledge(
        store, generated_at=EXTRACTED_AT, source_description=source_description
    )

    assert (export_dir / "research-knowledge.json").read_text(
        encoding="utf-8"
    ) == knowledge_to_json(rendered)
    assert (export_dir / "research-knowledge-findings.csv").read_text(
        encoding="utf-8"
    ) == findings_to_csv(rendered)
    assert (export_dir / "research-knowledge-findings_observations.csv").read_text(
        encoding="utf-8"
    ) == observations_to_csv(rendered)
    assert (export_dir / "research-knowledge.md").read_text(
        encoding="utf-8"
    ) == knowledge_to_markdown(rendered)

    redacted, _ledger = strip_reviewer_identities(
        rendered,
        human_identities=probable_human_identities(rendered),
        salt=DEMONSTRATION_SALT,
    )
    assert (export_dir / "research-knowledge-external.json").read_text(
        encoding="utf-8"
    ) == knowledge_to_json(redacted)


def test_the_committed_external_export_contains_no_real_reviewer_identity() -> None:
    """The concrete privacy assertion on the committed artifact itself, not on a fixture.

    `hypergeek-dev` is the real personal GitHub handle recorded 23 times in the committed knowledge log
    (`docs/telemetry-retention.md` §7). It must appear zero times in the export destined for external
    use, and the pseudonym must appear in its place -- attribution replaced, not deleted.
    """
    export_dir = BASELINE_DIR / "knowledge-export"
    if not export_dir.exists():  # pragma: no cover - the artifact is committed
        pytest.skip(f"{export_dir} is not present in this checkout")

    external = (export_dir / "research-knowledge-external.json").read_text(encoding="utf-8")
    internal = (export_dir / "research-knowledge.json").read_text(encoding="utf-8")

    assert "hypergeek-dev" in internal, (
        "the internal export is expected to keep real attribution -- if this fails, the redaction was "
        "applied to the wrong file"
    )
    assert "hypergeek-dev" not in external
    assert pseudonymise("hypergeek-dev", salt=DEMONSTRATION_SALT) in external
    # Statuses survived the redaction in the committed artifact too.
    for status in ("Candidate", "Provisionally supported", "Disputed"):
        assert status in external
