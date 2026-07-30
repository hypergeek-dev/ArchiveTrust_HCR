"""Research-knowledge export: JSON, CSV, structured Markdown, and a machine-readable evidence bundle.

Written to `src/archivetrust/research/reports/export.py`'s conventions rather than a second style --
that module is this codebase's existing export idiom and this one deliberately reuses it:

* a versioned `*_EXPORT_SCHEMA` constant stamped into every payload, so an exported file states which
  schema produced it (`archivetrust.canonical_document.v2`, `archivetrust.research_report.v1`, and now
  `archivetrust.research_knowledge.v1`);
* `UNSUPPORTED_FORMATS` naming HTML/PDF explicitly, so a caller asking for one gets a refusal with a
  reason rather than a silently missing option (the dependency set still has no template engine and no
  PDF writer);
* JSON round-trips and is tested to; CSV is documented as a **lossy flat view**, never called a
  serialization format;
* `write_*` writes UTF-8 without a BOM and `csv.writer(lineterminator="\\n")` so output is
  byte-identical across platforms.

`UnsupportedExportFormatError` is **redefined here rather than imported** from
`research/reports/export.py`. `research/reports/models.py` imports from `htr/`, so importing back
would make the two packages mutually dependent for the sake of one exception class. The two errors are
siblings by intent; the docstring on each says so.

**The one requirement that shapes every format in this module.** The knowledge follow-up is explicit:
*do not export candidate or disputed findings as established conclusions without preserving their
status*. A `review_status` field buried among thirty others satisfies the letter of that and fails it
in practice -- a reader quoting a blockquote, a spreadsheet user hiding a column, or a script reading
`statement` alone all lose it. So this module does not rely on the field's presence. Every format
carries `status_qualified_statement(finding)` -- the status and the statement in **one string**:

    [Candidate] On the single controlled baseline line, Florence-2 produced lower CER and WER ...

The verbatim `statement` is preserved untouched alongside it for machine consumers, and
`review_status` is still its own typed field. The qualified form is the *third* place the status
appears, and the only one that cannot be separated from the claim it qualifies by any downstream
projection, column hiding, or copy-paste. `tests/htr/knowledge/test_export.py` asserts the status word
lands within `MAX_STATUS_DISTANCE` characters of every statement in **all four** formats, per finding,
not once per file.

Nothing here computes, re-derives, or normalises a measured value: the export reads entities out of a
store and renders them. A finding's `evidence_references` are *resolved* (looked up through its
`supporting_observations`) and each resolved reference records the observation it came via, so the
resolution is auditable rather than asserted -- see `ResolvedEvidenceReference`.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.knowledge.models import (
    EvidenceReference,
    FindingStatus,
    ResearchFinding,
    ResearchObservation,
    ResearchQuestion,
)

KNOWLEDGE_EXPORT_SCHEMA = "archivetrust.research_knowledge.v1"
"""Versioned exactly like `REPORT_EXPORT_SCHEMA` and `archivetrust.canonical_document.v2`."""

EVIDENCE_BUNDLE_SCHEMA = "archivetrust.research_knowledge_bundle.v1"
"""The bundle manifest's own schema -- a separate id because a bundle is a *directory* of files with
an integrity manifest, not a single serialized export, and a reader must be able to tell which it
holds without inspecting the shape."""

SUPPORTED_FORMATS = ("json", "csv", "markdown", "bundle")

UNSUPPORTED_FORMATS = ("html", "pdf")
"""Named rather than silently absent, for the reason `research/reports/export.py` records: this
project has no template engine and no PDF writer, and adding a rendering dependency as a side effect
of an export feature was judged out of scope there and is judged out of scope here."""


class UnsupportedExportFormatError(ValueError):
    """Raised for an export format this project deliberately does not implement.

    Sibling of `research/reports/export.py::UnsupportedExportFormatError`, deliberately not the same
    class -- see this module's docstring on why `htr/` does not import `research/`.
    """


MAX_STATUS_DISTANCE = 400
"""How close a finding's status word must be to its statement, in characters, in every rendered
format. Not a formatting preference: it is the machine-checkable form of "status is impossible to
lose or misread", and `tests/htr/knowledge/test_export.py` enforces it per finding per format.

400 rather than 0 because the qualified statement is `"[Candidate] <statement>"` -- the status
precedes the statement by a dozen characters in Markdown and CSV, and in JSON the verbatim
`statement` line and the `status_qualified_statement` line that follows it are separated by the
statement's own length in the sorted key order. A window, not an exact offset, is what the property
actually is.
"""


def status_qualified_statement(finding: ResearchFinding) -> str:
    """`"[<status>] <statement>"` -- the status and the claim in one inseparable string.

    The single mechanism behind this module's status-visibility guarantee. Used by all four writers,
    so a new format cannot accidentally omit it and a change to the convention changes every format at
    once.
    """
    return f"[{finding.review_status.value}] {finding.statement}"


def status_badge(status: FindingStatus) -> str:
    """The Markdown badge form. `**Status: Candidate**` -- bold, on its own line, directly under the
    heading that also names the status. Deliberately not a footnote marker: a footnote is exactly the
    "buried" rendering the brief forbids."""
    return f"**Status: {status.value}**"


# -- Resolved evidence ----------------------------------------------------------------------------


class ResolvedEvidenceReference(BaseModel):
    """One `EvidenceReference` reached from a finding, with the path by which it was reached.

    A `ResearchFinding` carries no `supporting_evidence` field of its own -- by design
    (`htr/knowledge/models.py`: a finding rests on observations and never re-derives their evidence).
    So a finding's evidence references are *derived*, and this type records the derivation rather than
    flattening it away: `via_observation_id` names the `ResearchObservation` whose
    `supporting_evidence` this entry came from, so a reader can check the hop instead of trusting it.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    reference_id: str
    stream: str | None = None
    note: str | None = None
    via_observation_id: str
    """The observation this reference was read from. Never synthesised."""

    @classmethod
    def of(cls, reference: EvidenceReference, *, observation_id: str) -> "ResolvedEvidenceReference":
        return cls(
            kind=reference.kind.value,
            reference_id=reference.reference_id,
            stream=reference.stream,
            note=reference.note,
            via_observation_id=observation_id,
        )


def resolve_finding_evidence(
    finding: ResearchFinding, observations: dict[str, ResearchObservation]
) -> tuple[tuple[ResolvedEvidenceReference, ...], tuple[str, ...]]:
    """`(resolved references, unresolvable observation ids)` for one finding.

    Returns the unresolvable ids rather than dropping them or raising: an export assembled from a
    partial replay legitimately may not hold every supporting observation, and an export that silently
    showed fewer evidence references than the finding actually rests on would understate its support
    with nothing saying so. Both halves end up in the exported payload
    (`unresolved_supporting_observations`).
    """
    resolved: list[ResolvedEvidenceReference] = []
    unresolved: list[str] = []
    for observation_id in finding.supporting_observations:
        observation = observations.get(observation_id)
        if observation is None:
            unresolved.append(observation_id)
            continue
        for reference in observation.supporting_evidence:
            resolved.append(ResolvedEvidenceReference.of(reference, observation_id=observation_id))
    return tuple(resolved), tuple(unresolved)


# -- The export payload ---------------------------------------------------------------------------

REQUIRED_FINDING_EXPORT_FIELDS = (
    "statement",
    "review_status",
    "status_qualified_statement",
    "scope",
    "evidence_references",
    "supporting_experiments",
    "supporting_metrics",
    "affected_methods",
    "affected_model_versions",
    "affected_datasets",
    "affected_dataset_versions",
    "transcription_convention",
    "limitations",
    "contradictory_evidence",
    "revision_history",
)
"""The follow-up's required per-finding content, as the exact keys every exported finding object
carries. A module constant rather than a list in a test, so the test reads the requirement off the
code that implements it instead of restating it and drifting.

`evidence_references` is the derived, resolved list (see `resolve_finding_evidence`); the rest are
`ResearchFinding` fields. `methods`/`model versions` map to `affected_methods`/
`affected_model_versions`, which is what the entity calls them.
"""


def finding_payload(
    finding: ResearchFinding, observations: dict[str, ResearchObservation]
) -> dict[str, Any]:
    """One finding as an export-ready dict: every `ResearchFinding` field, plus the derived
    `status_qualified_statement` and resolved `evidence_references`.

    `model_dump(mode="json")` supplies the entity's own fields verbatim -- `statement`,
    `revision_history`, `contradictory_evidence` and the rest are the stored values, never
    reformatted. Only the two derived keys are added.
    """
    references, unresolved = resolve_finding_evidence(finding, observations)
    payload = finding.model_dump(mode="json")
    payload["status_qualified_statement"] = status_qualified_statement(finding)
    payload["evidence_references"] = [r.model_dump(mode="json") for r in references]
    payload["unresolved_supporting_observations"] = list(unresolved)
    return payload


def observation_payload(observation: ResearchObservation) -> dict[str, Any]:
    """One observation as an export-ready dict.

    `unverified_hypothesis` stays in its own key exactly as stored -- it is the one field on a
    `ResearchObservation` that is explicitly not a fact, and folding it into `description` on the way
    out would erase the distinction the entity exists to keep.
    """
    return observation.model_dump(mode="json")


def question_payload(question: ResearchQuestion) -> dict[str, Any]:
    return question.model_dump(mode="json")


class KnowledgeExport(BaseModel):
    """Everything one export renders: the observations, findings and questions of a knowledge store.

    Assembled by `collect_knowledge`, never constructed field-by-field by a writer -- so the four
    formats are guaranteed to render the same set of entities and a difference between two exports is
    a rendering difference, not a collection difference.
    """

    model_config = ConfigDict(frozen=True)

    observations: tuple[ResearchObservation, ...]
    findings: tuple[ResearchFinding, ...]
    questions: tuple[ResearchQuestion, ...] = ()
    generated_at: str
    """Supplied by the caller, never read from a clock here: a committed research artifact must be
    reproducible byte-for-byte, which is the same reason `scripts/register_baseline_knowledge.py`
    passes a fixed `EXTRACTED_AT` instead of `datetime.now()`."""
    source_description: str | None = None
    """Where these entities were read from, in one line -- e.g. the committed log's path. Free text
    because it is provenance annotation for a human, and every load-bearing id is already on the
    entities themselves."""

    @property
    def observations_by_id(self) -> dict[str, ResearchObservation]:
        return {o.observation_id: o for o in self.observations}

    def status_counts(self) -> dict[str, int]:
        """`status -> count`, for the summary every format opens with. Derived, so a summary can never
        disagree with the findings it summarises."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.review_status.value] = counts.get(finding.review_status.value, 0) + 1
        return dict(sorted(counts.items()))


def collect_knowledge(
    store: Any, *, generated_at: str, source_description: str | None = None
) -> KnowledgeExport:
    """Reads a knowledge store's observations, findings and questions into a `KnowledgeExport`.

    `store` is a `DurableHtrResearchStore` (or the plain `HtrResearchStore` projection behind it),
    typed `Any` for the reason `htr/knowledge/registration.py` records: keeping this module's imports
    free of the persistence layer. Only the three query accessors are used
    (`research_observations()`, `findings()`, `research_questions()`), all of which return their rows
    already deterministically sorted, so an export's ordering is the store's ordering rather than
    something this function invents.
    """
    return KnowledgeExport(
        observations=tuple(store.research_observations()),
        findings=tuple(store.findings()),
        questions=tuple(store.research_questions()),
        generated_at=generated_at,
        source_description=source_description,
    )


# -- JSON -----------------------------------------------------------------------------------------


def knowledge_to_json(export: KnowledgeExport, *, indent: int | None = 2) -> str:
    """Serializes one export. Round-trips through `knowledge_from_json` (tested).

    `sort_keys=True` matches `report_to_json`, and has a second effect worth naming here: it puts
    `status_qualified_statement` on the line immediately after `statement` in every finding object, so
    the status travels with the claim even in the raw text a reader greps.
    """
    payload = {
        "export_schema": KNOWLEDGE_EXPORT_SCHEMA,
        "generated_at": export.generated_at,
        "source_description": export.source_description,
        "status_counts": export.status_counts(),
        "observations": [observation_payload(o) for o in export.observations],
        "findings": [finding_payload(f, export.observations_by_id) for f in export.findings],
        "research_questions": [question_payload(q) for q in export.questions],
    }
    return json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)


def knowledge_from_json(text: str) -> KnowledgeExport:
    """Reconstructs a `KnowledgeExport` from `knowledge_to_json` output.

    Rejects a payload written by a different schema version rather than best-effort-parsing it into
    something subtly wrong -- `report_from_json`'s rule, and it matters more here: a knowledge export
    whose `review_status` semantics changed between schema versions is exactly the payload that must
    not be silently reinterpreted.

    The two derived keys (`status_qualified_statement`, `evidence_references`) are ignored on the way
    back in: they are projections of stored fields, and re-validating them would let a hand-edited
    export assert a status its own finding does not carry.
    """
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("A research knowledge export must be a JSON object")
    schema = payload.get("export_schema")
    if schema != KNOWLEDGE_EXPORT_SCHEMA:
        raise ValueError(
            f"Unsupported research knowledge export schema {schema!r} "
            f"(this build reads {KNOWLEDGE_EXPORT_SCHEMA!r})"
        )
    return KnowledgeExport(
        observations=tuple(
            ResearchObservation.model_validate(row) for row in payload.get("observations", ())
        ),
        findings=tuple(
            ResearchFinding.model_validate(
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in (
                        "status_qualified_statement",
                        "evidence_references",
                        "unresolved_supporting_observations",
                    )
                }
            )
            for row in payload.get("findings", ())
        ),
        questions=tuple(
            ResearchQuestion.model_validate(row) for row in payload.get("research_questions", ())
        ),
        generated_at=payload.get("generated_at", ""),
        source_description=payload.get("source_description"),
    )


# -- CSV ------------------------------------------------------------------------------------------

FINDING_CSV_COLUMNS = (
    "finding_id",
    "review_status",
    "status_qualified_statement",
    "statement",
    "confidence_level",
    "hypothesis_relationship",
    "research_question",
    "scope_description",
    "scope_sample_size",
    "scope_experiment_id",
    "scope_experiment_version_id",
    "scope_experiment_run_ids",
    "scope_normalization_profile",
    "affected_methods",
    "affected_model_versions",
    "affected_datasets",
    "affected_dataset_versions",
    "transcription_convention",
    "supporting_observations",
    "supporting_experiments",
    "supporting_metrics",
    "evidence_reference_count",
    "evidence_references",
    "limitations",
    "contradiction_count",
    "contradictions",
    "revision_count",
    "revision_history",
    "author",
    "reviewer",
    "creation_date",
    "superseded_by",
)
"""One row per finding. **`review_status` is column 2 and `status_qualified_statement` is column 3**,
immediately before `statement` -- so the status is visible in the first screen of any spreadsheet, and
sorting or filtering by it needs no scrolling. A CSV column can be hidden, which is exactly why the
qualified statement is also a column: hiding `review_status` does not strip the status from the row.

Lossy by design, like `report_to_csv`: `scope` is a nested object rendered as a one-sentence
`describe()` plus its load-bearing ids, and the four list-valued columns are space- or `" | "`-joined.
Use `knowledge_to_json` when the records must survive a round trip.
"""

OBSERVATION_CSV_COLUMNS = (
    "observation_id",
    "observation_type",
    "review_status",
    "title",
    "description",
    "observation_confidence",
    "scope_description",
    "scope_sample_size",
    "source_experiment_id",
    "source_experiment_run_id",
    "affected_method",
    "affected_model_version",
    "affected_dataset_id",
    "affected_dataset_version_id",
    "evidence_reference_count",
    "supporting_evidence",
    "unverified_hypothesis",
    "author_or_source_component",
    "creation_timestamp",
    "tags",
)
"""One row per observation. `unverified_hypothesis` gets its own column and is never merged into
`description` -- a guess and a measurement in one cell would be indistinguishable, which is the whole
reason the entity separates them."""

_LIST_SEPARATOR = " | "
"""Used for list-valued CSV cells whose members can contain spaces (limitations, statements). Plain
space-joining is kept for id lists, matching `report_to_csv`'s `" ".join(...)`."""


def _join_ids(values: Any) -> str:
    return " ".join(str(v) for v in values)


def _join_text(values: Any) -> str:
    return _LIST_SEPARATOR.join(str(v) for v in values)


def _reference_cell(references: tuple[ResolvedEvidenceReference, ...]) -> str:
    return _join_text(f"{r.kind}:{r.reference_id}(via {r.via_observation_id})" for r in references)


def findings_to_csv(export: KnowledgeExport) -> str:
    """A flat, one-row-per-finding view. See `FINDING_CSV_COLUMNS` on what is lost and why.

    Line terminator is `\\n` explicitly, so output is byte-identical across platforms rather than
    picking up `\\r\\n` from `csv`'s default on Windows -- exported research artifacts differing only
    by platform would defeat the hash-comparison discipline used elsewhere in this codebase.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(FINDING_CSV_COLUMNS)
    observations = export.observations_by_id
    for finding in export.findings:
        references, _unresolved = resolve_finding_evidence(finding, observations)
        scope = finding.scope
        writer.writerow(
            [
                finding.finding_id,
                finding.review_status.value,
                status_qualified_statement(finding),
                finding.statement,
                finding.confidence_level.value,
                finding.hypothesis_relationship.value if finding.hypothesis_relationship else "",
                finding.research_question or "",
                scope.describe(),
                scope.sample_size,
                scope.experiment_id,
                scope.experiment_version_id,
                _join_ids(scope.experiment_run_ids),
                scope.normalization_profile or "",
                _join_ids(finding.affected_methods),
                _join_ids(finding.affected_model_versions),
                _join_ids(finding.affected_datasets),
                _join_ids(finding.affected_dataset_versions),
                finding.transcription_convention or "",
                _join_ids(finding.supporting_observations),
                _join_ids(finding.supporting_experiments),
                _join_ids(finding.supporting_metrics),
                len(references),
                _reference_cell(references),
                _join_text(finding.limitations),
                len(finding.contradictory_evidence),
                _join_text(
                    f"{c.source_kind.value}:{c.source_id} -- {c.description}"
                    for c in finding.contradictory_evidence
                ),
                len(finding.revision_history),
                _join_text(
                    f"{r.from_status.value} -> {r.to_status.value} by {r.actor}"
                    for r in finding.revision_history
                ),
                finding.author,
                finding.reviewer or "",
                finding.creation_date,
                finding.superseded_by or "",
            ]
        )
    return buffer.getvalue()


def observations_to_csv(export: KnowledgeExport) -> str:
    """A flat, one-row-per-observation view. Lossy by design, same as `findings_to_csv`."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(OBSERVATION_CSV_COLUMNS)
    for observation in export.observations:
        scope = observation.scope
        writer.writerow(
            [
                observation.observation_id,
                observation.observation_type.value,
                observation.review_status.value,
                observation.title,
                observation.description,
                observation.observation_confidence.value,
                scope.describe(),
                scope.sample_size,
                observation.source_experiment_id,
                observation.source_experiment_run_id,
                observation.affected_method or "",
                observation.affected_model_version or "",
                observation.affected_dataset_id or "",
                observation.affected_dataset_version_id or "",
                len(observation.supporting_evidence),
                _join_text(
                    f"{r.kind.value}:{r.reference_id}" for r in observation.supporting_evidence
                ),
                observation.unverified_hypothesis or "",
                observation.author_or_source_component,
                observation.creation_timestamp,
                _join_ids(observation.tags),
            ]
        )
    return buffer.getvalue()


def knowledge_to_csv(export: KnowledgeExport) -> str:
    """`findings_to_csv` -- the default CSV view, because a finding is the record a reader is most
    likely to quote and therefore the one whose status most needs to survive the flattening.

    Observations get their own table (`observations_to_csv`) rather than being interleaved here:
    one CSV holding two entity types with disjoint columns is a spreadsheet nobody can filter.
    `write_knowledge` writes both when asked for `"csv"` against a directory.
    """
    return findings_to_csv(export)


# -- Markdown -------------------------------------------------------------------------------------


def knowledge_to_markdown(export: KnowledgeExport) -> str:
    """Structured Markdown: a status summary, then one section per finding, then the observations,
    then the research-question chain.

    **Every finding's status appears three times before its statement is readable**: in the section
    heading, in a bold badge line directly under it, and inside the blockquote as the qualified
    statement. There is no rendering of this document in which a reader reaches the claim without
    having passed the status, and no footnote carrying it.
    """
    lines: list[str] = []
    lines.append("# Research Knowledge Export")
    lines.append("")
    lines.append(f"* Export schema: `{KNOWLEDGE_EXPORT_SCHEMA}`")
    lines.append(f"* Generated at: `{export.generated_at}`")
    if export.source_description:
        lines.append(f"* Source: {export.source_description}")
    lines.append(
        f"* Contents: {len(export.findings)} finding(s), {len(export.observations)} observation(s), "
        f"{len(export.questions)} research question(s)"
    )
    lines.append("")
    lines.append(
        "> **Read the status on every finding below.** A `Candidate` or `Disputed` finding is not an "
        "established conclusion. Statuses are carried in each finding's heading, in a badge under it, "
        "and inside the quoted statement itself -- none of the three is a footnote, and none may be "
        "dropped when quoting."
    )
    lines.append("")

    lines.append("## Status summary")
    lines.append("")
    lines.append("| Status | Findings |")
    lines.append("|---|---|")
    for status, count in export.status_counts().items():
        lines.append(f"| **{status}** | {count} |")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    observations = export.observations_by_id
    for index, finding in enumerate(export.findings, start=1):
        lines.extend(_finding_markdown(index, finding, observations))

    lines.append("## Observations")
    lines.append("")
    lines.append(
        "An observation claims only *what the records for its scope show*, never how a method "
        "behaves in general. `unverified_hypothesis`, where present, is a candidate explanation that "
        "has **not** been measured and is held separately from every factual field."
    )
    lines.append("")
    for index, observation in enumerate(export.observations, start=1):
        lines.extend(_observation_markdown(index, observation))

    if export.questions:
        lines.append("## Research questions")
        lines.append("")
        for index, question in enumerate(export.questions, start=1):
            lines.extend(_question_markdown(index, question))

    return "\n".join(lines).rstrip() + "\n"


def _finding_markdown(
    index: int, finding: ResearchFinding, observations: dict[str, ResearchObservation]
) -> list[str]:
    references, unresolved = resolve_finding_evidence(finding, observations)
    scope = finding.scope
    lines: list[str] = []
    lines.append(f"### {index}. {finding.review_status.value} — `{finding.finding_id}`")
    lines.append("")
    lines.append(status_badge(finding.review_status))
    lines.append("")
    lines.append(f"> {status_qualified_statement(finding)}")
    lines.append("")
    lines.append(f"* **Status**: `{finding.review_status.value}`")
    lines.append(f"* Confidence in the claim: `{finding.confidence_level.value}`")
    if finding.hypothesis_relationship is not None:
        lines.append(f"* Hypothesis relationship: `{finding.hypothesis_relationship.value}`")
    if finding.research_question:
        lines.append(f"* Research question: {finding.research_question}")
    lines.append(f"* Scope: {scope.describe()}")
    lines.append(f"* Sample size (derived, `len(covered_unit_ids)`): **N={scope.sample_size}**")
    lines.append(
        "* Methods @ model versions: "
        + (
            ", ".join(
                f"`{method}` @ `{revision}`"
                for method, revision in zip(
                    finding.affected_methods, finding.affected_model_versions, strict=False
                )
            )
            or "*(none named)*"
        )
    )
    lines.append(
        "* Datasets @ dataset versions: "
        + (
            ", ".join(
                f"`{dataset}` @ `{version}`"
                for dataset, version in zip(
                    finding.affected_datasets, finding.affected_dataset_versions, strict=False
                )
            )
            or "*(none named)*"
        )
    )
    lines.append(
        "* Transcription convention: "
        + (
            f"`{finding.transcription_convention}`"
            if finding.transcription_convention
            else "**none recorded** — no versioned `TranscriptionConvention` exists for this "
            "ground truth"
        )
    )
    lines.append(
        "* Supporting experiments: "
        + (", ".join(f"`{e}`" for e in finding.supporting_experiments) or "*(none)*")
    )
    lines.append(
        "* Supporting metrics: "
        + (", ".join(f"`{m}`" for m in finding.supporting_metrics) or "*(none — see limitations)*")
    )
    lines.append(
        "* Supporting observations: "
        + ", ".join(f"`{o}`" for o in finding.supporting_observations)
    )
    lines.append("")

    lines.append(f"**Evidence references** ({len(references)}, resolved via supporting observations)")
    lines.append("")
    if references:
        lines.append("| Kind | Reference | Stream | Via observation |")
        lines.append("|---|---|---|---|")
        for reference in references:
            lines.append(
                f"| `{reference.kind}` | `{reference.reference_id}` | "
                f"`{reference.stream or '(this stream)'}` | `{reference.via_observation_id}` |"
            )
    else:
        lines.append("*No evidence reference resolved.*")
    if unresolved:
        lines.append("")
        lines.append(
            "**Unresolved supporting observations** (named by the finding, absent from this export): "
            + ", ".join(f"`{o}`" for o in unresolved)
        )
    lines.append("")

    lines.append("**Limitations**")
    lines.append("")
    for limitation in finding.limitations:
        lines.append(f"* {limitation}")
    lines.append("")

    lines.append(f"**Contradictions** ({len(finding.contradictory_evidence)})")
    lines.append("")
    if finding.contradictory_evidence:
        for contradiction in finding.contradictory_evidence:
            lines.append(
                f"* `{contradiction.contradiction_id}` — {contradiction.source_kind.value} "
                f"`{contradiction.source_id}`, recorded by {contradiction.recorded_by} at "
                f"{contradiction.recorded_at}"
            )
            lines.append(f"  * {contradiction.description}")
    else:
        lines.append("*None recorded.*")
    lines.append("")

    lines.append(f"**Revision history** ({len(finding.revision_history)}, append-only)")
    lines.append("")
    if finding.revision_history:
        lines.append("| Revised at | From | To | Actor | Reasoning |")
        lines.append("|---|---|---|---|---|")
        for revision in finding.revision_history:
            reasoning = revision.reasoning.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {revision.revised_at} | `{revision.from_status.value}` | "
                f"`{revision.to_status.value}` | {revision.actor} | {reasoning} |"
            )
    else:
        lines.append(
            "*None — this finding is still at the status it was created at, which is why it has no "
            "reviewer.*"
        )
    lines.append("")
    lines.append(
        f"*End of finding `{finding.finding_id}` — status `{finding.review_status.value}`.*"
    )
    lines.append("")
    return lines


def _observation_markdown(index: int, observation: ResearchObservation) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"### O{index}. {observation.title} — `{observation.observation_type.value}` "
        f"(review: {observation.review_status.value})"
    )
    lines.append("")
    lines.append(f"`{observation.observation_id}`")
    lines.append("")
    lines.append(f"> {observation.description}")
    lines.append("")
    lines.append(f"* Observer confidence in the observation: `{observation.observation_confidence.value}`")
    lines.append(f"* Scope: {observation.scope.describe()}")
    lines.append(
        f"* Source: experiment `{observation.source_experiment_id}`, run "
        f"`{observation.source_experiment_run_id}`"
    )
    if observation.affected_method:
        lines.append(
            f"* Method @ model version: `{observation.affected_method}` @ "
            f"`{observation.affected_model_version}`"
        )
    if observation.tags:
        lines.append("* Tags: " + ", ".join(f"`{t}`" for t in observation.tags))
    lines.append(f"* Recorded by: {observation.author_or_source_component} at {observation.creation_timestamp}")
    lines.append("")
    if observation.unverified_hypothesis:
        lines.append(
            "**UNVERIFIED HYPOTHESIS — not a measurement, not part of this observation's factual "
            "content:**"
        )
        lines.append("")
        lines.append(f"> {observation.unverified_hypothesis}")
        lines.append("")
    lines.append(f"**Supporting evidence** ({len(observation.supporting_evidence)} typed references)")
    lines.append("")
    lines.append("| Kind | Reference | Stream | Note |")
    lines.append("|---|---|---|---|")
    for reference in observation.supporting_evidence:
        note = (reference.note or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{reference.kind.value}` | `{reference.reference_id}` | "
            f"`{reference.stream or '(this stream)'}` | {note} |"
        )
    lines.append("")
    return lines


def _question_markdown(index: int, question: ResearchQuestion) -> list[str]:
    lines: list[str] = []
    lines.append(f"### Q{index}. {question.status.value} — `{question.question_id}`")
    lines.append("")
    lines.append(f"> {question.statement}")
    lines.append("")
    lines.append(f"* **Status**: `{question.status.value}`")
    lines.append(f"* Motivation: {question.motivation}")
    if question.originating_observation_id:
        lines.append(f"* Originating observation: `{question.originating_observation_id}`")
    if question.originating_finding_id:
        lines.append(f"* Originating finding: `{question.originating_finding_id}`")
    if question.originating_contradiction_id:
        lines.append(f"* Originating contradiction: `{question.originating_contradiction_id}`")
    if question.created_experiment_id:
        lines.append(
            f"* Drafted experiment: `{question.created_experiment_id}` version "
            f"`{question.created_experiment_version_id}` — **drafted, not executed**"
        )
    if question.answered_by_finding_id:
        lines.append(f"* Answered by finding: `{question.answered_by_finding_id}`")
    lines.append(f"* Raised by: {question.created_by} at {question.created_at}")
    lines.append("")
    for hypothesis in question.hypotheses:
        lines.append(f"**Hypothesis** `{hypothesis.hypothesis_id}`")
        lines.append("")
        lines.append(f"> {hypothesis.statement}")
        lines.append("")
        lines.append(f"* Falsification criterion: {hypothesis.falsification_criterion}")
        lines.append(f"* Author: {hypothesis.author} at {hypothesis.created_at}")
        lines.append("")
    if not question.hypotheses:
        lines.append("*No hypothesis attached.*")
        lines.append("")
    return lines


# -- Machine-readable evidence bundle -------------------------------------------------------------

BUNDLE_FILES_DIRECTORY_NAMES = ("findings", "observations", "research_questions")


def write_evidence_bundle(export: KnowledgeExport, directory: Path) -> Path:
    """Writes a machine-readable evidence bundle: one JSON file per entity plus a digest manifest.

    Layout:

        <directory>/
          manifest.json                      -- schema, generated_at, counts, status index, digests
          evidence_index.json                -- every evidence reference, flattened, both directions
          findings/<finding_id>.json         -- one finding, with its resolved evidence references
          observations/<observation_id>.json
          research_questions/<question_id>.json

    **Why a directory rather than one big JSON.** A bundle's job is to be *consumed* -- a downstream
    tool resolving one finding's evidence should read one small file and verify one digest, not parse
    the whole knowledge base. That is also what makes the digests useful: a per-file digest localises
    a mismatch to one entity, where a single whole-export digest only says "something changed".

    Integrity uses `infrastructure/storage/integrity.py::build_file_digest_manifest` -- the same
    helper `evaluation/ground_truth.py::export_with_provenance` reuses for exactly this purpose,
    rather than a fourth hashing idiom. Only the digest and size are kept per file; the manifest's one
    `generated_at` is `export.generated_at` (caller-supplied), so a bundle is reproducible
    byte-for-byte instead of embedding a clock read per file.

    **Status in the bundle.** `manifest.json` carries a `status_index` mapping every `finding_id` to
    its status *and* its `status_qualified_statement`, so a consumer that reads only the manifest
    still cannot obtain a statement without its status. Each per-finding file carries all three forms
    as well.
    """
    from archivetrust.infrastructure.storage.integrity import build_file_digest_manifest

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES_DIRECTORY_NAMES:
        (directory / name).mkdir(parents=True, exist_ok=True)

    observations = export.observations_by_id
    written: list[tuple[str, Path]] = []

    for finding in export.findings:
        path = directory / "findings" / f"{finding.finding_id}.json"
        _write_json(path, finding_payload(finding, observations))
        written.append((f"findings/{finding.finding_id}.json", path))

    for observation in export.observations:
        path = directory / "observations" / f"{observation.observation_id}.json"
        _write_json(path, observation_payload(observation))
        written.append((f"observations/{observation.observation_id}.json", path))

    for question in export.questions:
        path = directory / "research_questions" / f"{question.question_id}.json"
        _write_json(path, question_payload(question))
        written.append((f"research_questions/{question.question_id}.json", path))

    index_path = directory / "evidence_index.json"
    _write_json(index_path, _evidence_index(export))
    written.append(("evidence_index.json", index_path))

    manifest = {
        "bundle_schema": EVIDENCE_BUNDLE_SCHEMA,
        "export_schema": KNOWLEDGE_EXPORT_SCHEMA,
        "generated_at": export.generated_at,
        "source_description": export.source_description,
        "counts": {
            "findings": len(export.findings),
            "observations": len(export.observations),
            "research_questions": len(export.questions),
        },
        "status_counts": export.status_counts(),
        "status_index": [
            {
                "finding_id": finding.finding_id,
                "review_status": finding.review_status.value,
                "status_qualified_statement": status_qualified_statement(finding),
            }
            for finding in export.findings
        ],
        "files": [
            {
                "path": relative,
                "algorithm": digest_manifest.algorithm,
                "size_bytes": digest_manifest.size_bytes,
                "digest": digest_manifest.digest,
            }
            for relative, digest_manifest in (
                (relative, build_file_digest_manifest(path)) for relative, path in written
            )
        ],
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _evidence_index(export: KnowledgeExport) -> dict[str, Any]:
    """Both directions of the evidence relation, neither derived from the other by deletion.

    `by_reference` answers "what knowledge cites this durable record?" -- the direction a reader
    looking at a `MetricCalculated` event needs, and the one no entity carries, because an
    `EvidenceReference` points outward only.
    """
    observations = export.observations_by_id
    by_record: dict[str, dict[str, Any]] = {}

    for observation in export.observations:
        for reference in observation.supporting_evidence:
            entry = by_record.setdefault(
                reference.reference_id,
                {
                    "reference_id": reference.reference_id,
                    "kind": reference.kind.value,
                    "stream": reference.stream,
                    "cited_by_observations": [],
                    "cited_by_findings": [],
                },
            )
            entry["cited_by_observations"].append(observation.observation_id)

    for finding in export.findings:
        references, _unresolved = resolve_finding_evidence(finding, observations)
        for reference in references:
            entry = by_record.setdefault(
                reference.reference_id,
                {
                    "reference_id": reference.reference_id,
                    "kind": reference.kind,
                    "stream": reference.stream,
                    "cited_by_observations": [],
                    "cited_by_findings": [],
                },
            )
            entry["cited_by_findings"].append(
                {
                    "finding_id": finding.finding_id,
                    "review_status": finding.review_status.value,
                    "via_observation_id": reference.via_observation_id,
                }
            )

    return {
        "bundle_schema": EVIDENCE_BUNDLE_SCHEMA,
        "generated_at": export.generated_at,
        "by_reference": [by_record[key] for key in sorted(by_record)],
        "by_finding": [
            {
                "finding_id": finding.finding_id,
                "review_status": finding.review_status.value,
                "status_qualified_statement": status_qualified_statement(finding),
                "supporting_observations": list(finding.supporting_observations),
                "supporting_metrics": list(finding.supporting_metrics),
                "evidence_reference_ids": [
                    reference.reference_id
                    for reference in resolve_finding_evidence(finding, observations)[0]
                ],
            }
            for finding in export.findings
        ],
    }


# -- Dispatch -------------------------------------------------------------------------------------


def write_knowledge(export: KnowledgeExport, path: Path, *, export_format: str) -> Path:
    """Writes one export to `path` in `export_format`.

    | format | `path` means | produces |
    |---|---|---|
    | `"json"` | a file | round-trippable JSON |
    | `"csv"` | a file | `findings_to_csv`; `observations_to_csv` goes to `<stem>_observations.csv` beside it |
    | `"markdown"` | a file | structured Markdown |
    | `"bundle"` | a **directory** | the evidence bundle; returns its `manifest.json` |

    Raises `UnsupportedExportFormatError` for `"html"`/`"pdf"` with the reason, and for anything else.
    UTF-8 without a BOM, matching every other writer in this codebase.
    """
    normalized = export_format.strip().lower()
    if normalized in UNSUPPORTED_FORMATS:
        raise UnsupportedExportFormatError(
            f"{normalized!r} export is not implemented: this project has no template engine or "
            "PDF writer, and adding a rendering dependency for an export format was judged out of "
            "scope (the same decision research/reports/export.py records). Export json, csv, "
            "markdown or bundle instead."
        )
    if normalized == "bundle":
        return write_evidence_bundle(export, path)
    if normalized == "json":
        text = knowledge_to_json(export)
    elif normalized == "markdown":
        text = knowledge_to_markdown(export)
    elif normalized == "csv":
        text = findings_to_csv(export)
    else:
        raise UnsupportedExportFormatError(
            f"Unknown research knowledge export format {export_format!r}; supported: "
            f"{', '.join(SUPPORTED_FORMATS)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if normalized == "csv":
        sibling = path.with_name(f"{path.stem}_observations{path.suffix}")
        sibling.write_text(observations_to_csv(export), encoding="utf-8")
    return path
