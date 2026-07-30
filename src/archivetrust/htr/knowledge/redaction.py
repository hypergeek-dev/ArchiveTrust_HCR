"""Reviewer-identity stripping for research-knowledge exports destined outside this repository.

**This module exists because real personal identity is genuinely stored, not because a policy said it
might be.** Checked before writing anything (`docs/telemetry-retention.md` §7 records the check and
its command):

* `docs/experiments/baseline-comparison/htr_knowledge_events.jsonl`, a committed artifact, contains the
  literal string `hypergeek-dev` -- a real personal GitHub handle, the repository maintainer -- 23
  times, across `ResearchFinding.reviewer` (4), `FindingRevision.actor` (6),
  `ContradictoryEvidence.recorded_by` (1) and the event envelope's `actor_id` (8). It is not an opaque
  id and was never intended to be: `scripts/register_baseline_knowledge.py::DEFAULT_REVIEWER`
  attributes it deliberately and honestly, because a finding cannot advance past `Candidate` without an
  attributable human.
* `review/blind_review/store.py` stores **no** real identity today -- it is in-memory and no real
  review has ever been run through it -- but the field it would store one in,
  `review/htr_models.py::ReviewAssignment.reviewer_ref` / `ReviewSubmission.reviewer_ref` /
  `Adjudication.adjudicator_ref`, is an undecorated `str` with no docstring requiring opacity and
  nothing in `src/` that mints an opaque value for it. So the *shape* permits a username or an email
  and the first real caller would decide by accident.

The honest reading of both facts together: the durable HTR store already holds real personal identity,
so a redaction mechanism is warranted and is built here; and the blind-review field is a latent version
of the same problem, so this module's primitives are deliberately generic enough to serve it
(`pseudonymise`, `IdentityLedger`) without `htr/knowledge/` importing `review/`.

**What is kept and what is stripped.** Accountability is not weakened: attribution stays complete in
the durable log, which is never rewritten (Constitution Article 15 -- supersede, never erase). What
changes is what leaves. `strip_reviewer_identities` produces a *derived export* in which each human
identity is replaced by a stable pseudonym, and returns the `IdentityLedger` mapping pseudonyms back to
real identities **separately**, so the accountability id exists and is simply not in the file that
goes out. Writing the ledger next to the redacted export would defeat the whole exercise; nothing in
this module writes it, and `IdentityLedger.model_dump` is the caller's deliberate act.

**On reversibility, stated plainly.** A pseudonym is `sha256(salt + identity)` truncated. With a
secret, high-entropy `salt` that is a one-way mapping in practice. With a guessable salt and a small
known population of reviewers it is trivially reversible by enumeration -- pseudonymisation is not
anonymisation, and this module does not claim otherwise. `DEMONSTRATION_SALT` is a fixed, published
constant used only so the committed example artifact is reproducible byte-for-byte; it is explicitly
not a privacy control, and `strip_reviewer_identities` requires `salt` to be passed rather than
defaulting to it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.knowledge.export import KnowledgeExport
from archivetrust.htr.knowledge.models import (
    ResearchFinding,
    ResearchObservation,
    ResearchQuestion,
)

IDENTITY_BEARING_FIELDS = {
    "ResearchObservation": ("author_or_source_component",),
    "ResearchFinding": ("author", "reviewer"),
    "FindingRevision": ("actor",),
    "ContradictoryEvidence": ("recorded_by",),
    "ResearchQuestion": ("created_by",),
    "Hypothesis": ("author",),
}
"""Every field on a knowledge entity that can carry a human identity, by owning type.

Enumerated rather than discovered by scanning for `str` fields: `statement`, `reasoning` and
`description` are also strings and also frequently *mention* people, and a mechanism that redacted
those would silently mangle the research content. This module only ever touches the fields listed
here. That a reviewer's name written into a free-text `reasoning` field would survive redaction is a
real limitation, disclosed in `docs/telemetry-retention.md` §7 rather than papered over.
"""

REVIEW_RECORD_IDENTITY_FIELDS = ("reviewer_ref", "adjudicator_ref")
"""The blind-review identity fields (`review/htr_models.py`). Named here, and consumed by
`pseudonymise_mapping`, so a blind-review caller gets the same treatment without this module importing
`review/` -- which would invert the dependency direction `htr/knowledge/models.py`'s docstring
protects."""

COMPONENT_IDENTITY_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
"""A dotted, all-lowercase path -- `htr.knowledge.baseline_knowledge`, `scripts.register_baseline_knowledge`.

Every automated author in this repository's committed knowledge log matches this and every human one
(`hypergeek-dev`) does not, which is what makes `probable_component_identity` usable. It is a
**heuristic on a naming convention**, not a fact about the data, and `strip_reviewer_identities`
therefore takes an explicit `human_identities` set rather than applying it silently.
"""

PSEUDONYM_PREFIX = "reviewer_"
PSEUDONYM_HEX_LENGTH = 16

DEMONSTRATION_SALT = "archivetrust.knowledge-export.demonstration-salt.v1"
"""Fixed and published, so `scripts/export_baseline_knowledge.py`'s committed artifact is reproducible.

**Not a privacy control.** A published salt makes the pseudonyms in that one artifact enumerable by
anyone who can guess the reviewer population, which for this repository is one person named in its own
git history. It is used because the committed example's value is demonstrating the *mechanism* on real
data reproducibly; a real external release would pass a secret salt.
"""


def probable_component_identity(identity: str) -> bool:
    """`True` when `identity` looks like an automated component rather than a person.

    See `COMPONENT_IDENTITY_PATTERN`. A heuristic, and named one -- callers that must be certain pass
    `human_identities` explicitly.
    """
    return bool(COMPONENT_IDENTITY_PATTERN.match(identity.strip()))


def pseudonymise(identity: str, *, salt: str) -> str:
    """A stable, salted pseudonym for one identity: `reviewer_<16 hex chars>`.

    Stable across calls and across exports for the same `(identity, salt)` pair -- which is the point:
    two redacted exports must agree on which pseudonym is which reviewer, or an external reader cannot
    tell that the same person reviewed two findings, and inter-reviewer agreement becomes unanalysable
    outside this repository. Correlatability within one salt is a deliberate property, not a leak.
    """
    if not identity:
        raise ValueError("pseudonymise() requires a non-empty identity")
    digest = hashlib.sha256(f"{salt}\x00{identity}".encode()).hexdigest()
    return f"{PSEUDONYM_PREFIX}{digest[:PSEUDONYM_HEX_LENGTH]}"


class IdentityLedger(BaseModel):
    """The internal accountability record: `pseudonym -> real identity`.

    **The half that must not leave.** `strip_reviewer_identities` returns this beside the redacted
    export precisely so the caller has to make a separate, visible decision to persist it. Nothing in
    this module writes it to disk, and no redacted export contains it or its salt.

    `salt` is stored on the ledger because a ledger without its salt cannot be extended consistently
    later -- a second export of the same population would otherwise produce different pseudonyms for
    the same people.
    """

    model_config = ConfigDict(frozen=True)

    salt: str
    entries: tuple[tuple[str, str], ...]
    """`(pseudonym, real_identity)` pairs, sorted by pseudonym. A tuple of pairs rather than a dict so
    the model stays frozen and hashable like every other entity in this package."""

    def real_identity(self, pseudonym: str) -> str | None:
        for candidate, identity in self.entries:
            if candidate == pseudonym:
                return identity
        return None

    def pseudonym_for(self, identity: str) -> str | None:
        for pseudonym, candidate in self.entries:
            if candidate == identity:
                return pseudonym
        return None


def identity_values(export: KnowledgeExport) -> tuple[str, ...]:
    """Every distinct value appearing in any `IDENTITY_BEARING_FIELDS` field of `export`, sorted.

    The input a caller inspects before choosing `human_identities`. Deliberately does not classify:
    that is `probable_component_identity`'s job and the caller's decision.
    """
    found: set[str] = set()
    for observation in export.observations:
        found.add(observation.author_or_source_component)
    for finding in export.findings:
        found.add(finding.author)
        if finding.reviewer:
            found.add(finding.reviewer)
        for revision in finding.revision_history:
            found.add(revision.actor)
        for contradiction in finding.contradictory_evidence:
            found.add(contradiction.recorded_by)
    for question in export.questions:
        found.add(question.created_by)
        for hypothesis in question.hypotheses:
            found.add(hypothesis.author)
    return tuple(sorted(found))


def probable_human_identities(export: KnowledgeExport) -> frozenset[str]:
    """`identity_values` minus everything `probable_component_identity` accepts.

    A convenience for the common case, built on a named heuristic. A caller who cannot afford a
    heuristic passes its own set to `strip_reviewer_identities`; a caller who uses this one is opting
    into `COMPONENT_IDENTITY_PATTERN`'s naming convention and this docstring says so.
    """
    return frozenset(v for v in identity_values(export) if not probable_component_identity(v))


def pseudonymise_mapping(
    identities: frozenset[str] | set[str] | tuple[str, ...], *, salt: str
) -> IdentityLedger:
    """Builds the ledger for a known identity population.

    Exposed separately from `strip_reviewer_identities` so a blind-review caller can pseudonymise
    `reviewer_ref`/`adjudicator_ref` (`REVIEW_RECORD_IDENTITY_FIELDS`) against the *same* salt and get
    pseudonyms that line up with the knowledge export's -- without this module importing `review/`.
    """
    return IdentityLedger(
        salt=salt,
        entries=tuple(
            sorted((pseudonymise(identity, salt=salt), identity) for identity in identities)
        ),
    )


def strip_reviewer_identities(
    export: KnowledgeExport,
    *,
    human_identities: frozenset[str] | set[str] | tuple[str, ...],
    salt: str,
) -> tuple[KnowledgeExport, IdentityLedger]:
    """`(redacted export, ledger)`.

    Replaces every occurrence of every identity in `human_identities`, in every field listed in
    `IDENTITY_BEARING_FIELDS`, with its pseudonym. Identities not in the set are left verbatim -- which
    is how `htr.knowledge.baseline_knowledge` (a module path, not a person) survives redaction intact.

    **Nothing else changes.** Statements, statuses, scopes, limitations, evidence references,
    contradictions and revision histories are carried through unmodified, including
    `ResearchFinding.review_status` and every `FindingRevision`'s `from_status`/`to_status` and
    `reasoning`. A redacted export is still a full-fidelity research record: the follow-up's
    status-preservation requirement and its reviewer-privacy requirement are not in tension, and this
    function is where that is demonstrated rather than asserted.

    Re-validation is real, not `model_copy`: every rebuilt entity goes back through
    `model_validate`, so a redaction that produced e.g. a `Supported` finding with a now-empty reviewer
    would fail here instead of producing an invalid record. That is the same reason
    `ResearchQuestion._revalidated` exists.

    `human_identities` has no default. Guessing which of a repository's authors are people is exactly
    the decision that must not be made silently -- `probable_human_identities` is available for a
    caller that wants the heuristic, and using it is then that caller's visible choice.
    """
    population = frozenset(human_identities)
    ledger = pseudonymise_mapping(population, salt=salt)

    def redact(value: str | None) -> str | None:
        if value is None:
            return None
        return ledger.pseudonym_for(value) or value

    observations = tuple(
        ResearchObservation.model_validate(
            {
                **observation.model_dump(),
                "author_or_source_component": redact(observation.author_or_source_component),
            }
        )
        for observation in export.observations
    )

    findings = tuple(_redact_finding(finding, redact) for finding in export.findings)

    questions = tuple(
        ResearchQuestion.model_validate(
            {
                **question.model_dump(),
                "created_by": redact(question.created_by),
                "hypotheses": [
                    {**hypothesis.model_dump(), "author": redact(hypothesis.author)}
                    for hypothesis in question.hypotheses
                ],
            }
        )
        for question in export.questions
    )

    redacted = KnowledgeExport(
        observations=observations,
        findings=findings,
        questions=questions,
        generated_at=export.generated_at,
        source_description=_redacted_source_description(export.source_description),
    )
    return redacted, ledger


def _redact_finding(finding: ResearchFinding, redact: Any) -> ResearchFinding:
    payload = finding.model_dump()
    payload["author"] = redact(finding.author)
    payload["reviewer"] = redact(finding.reviewer)
    payload["revision_history"] = [
        {**revision.model_dump(), "actor": redact(revision.actor)}
        for revision in finding.revision_history
    ]
    payload["contradictory_evidence"] = [
        {**contradiction.model_dump(), "recorded_by": redact(contradiction.recorded_by)}
        for contradiction in finding.contradictory_evidence
    ]
    return ResearchFinding.model_validate(payload)


REDACTION_NOTICE = (
    "Reviewer identities in this export are pseudonyms "
    "(archivetrust.htr.knowledge.redaction.strip_reviewer_identities). The pseudonym-to-identity "
    "ledger is held internally and is not part of this export."
)
"""Stamped into the redacted export's `source_description` so the file says of itself that it is
redacted. An external reader must not have to infer from the shape of a string that `reviewer_a1b2...`
is not somebody's account name."""


def _redacted_source_description(original: str | None) -> str:
    if not original:
        return REDACTION_NOTICE
    return f"{original} — {REDACTION_NOTICE}"


def redact_review_record_identities(
    records: tuple[Any, ...], *, ledger: IdentityLedger
) -> tuple[Any, ...]:
    """Pseudonymises `reviewer_ref`/`adjudicator_ref` on blind-review records against an existing
    ledger.

    Structurally typed (`Any`) on purpose: `ReviewAssignment`, `ReviewSubmission` and `Adjudication`
    live in `review/htr_models.py`, and importing them here would make `htr/knowledge/` depend on
    `review/`. The three are frozen Pydantic models, so `model_validate(model_dump() | patch)` is the
    only way to derive one anyway, and that needs no static type.

    Takes a ledger rather than a salt so the pseudonyms match the knowledge export's for the same
    people -- a redacted review record and a redacted finding attributed to the same reviewer must
    agree, or agreement analysis across the two is impossible for an external reader.
    """
    redacted: list[Any] = []
    for record in records:
        payload = record.model_dump()
        changed = False
        for field in REVIEW_RECORD_IDENTITY_FIELDS:
            current = payload.get(field)
            if isinstance(current, str) and current:
                pseudonym = ledger.pseudonym_for(current)
                if pseudonym is not None:
                    payload[field] = pseudonym
                    changed = True
        redacted.append(type(record).model_validate(payload) if changed else record)
    return tuple(redacted)
