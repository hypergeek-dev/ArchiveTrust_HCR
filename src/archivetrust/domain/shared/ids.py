"""Identity helpers shared across the domain layer.

Two distinct id strategies are used deliberately:

- Evidence ids are *content-addressed* (Constitution Article 5, ROADMAP.md S5.2): identical raw
  provider output must always hash to the same id, so it is never stored twice. Content addressing
  is the mechanism, not a convenience -- it is what makes "the same Evidence record" a well-defined
  question across separate provider invocations.
- Every other domain identity (Observation, Canonical Observation instance, semantic-slot identity,
  graph invocation, etc.) is a random, non-content-derived identifier, because those objects are not
  defined by their content alone -- two Observations with identical payloads from the same provider
  invocation are still two distinct claims.
"""

from __future__ import annotations

import hashlib
import uuid


def new_id(prefix: str) -> str:
    """A random, non-content-derived identifier for a domain object."""
    return f"{prefix}_{uuid.uuid4().hex}"


def content_address(*parts: str, prefix: str = "evidence") -> str:
    """A deterministic, content-derived identifier.

    Given the same ordered parts, always produces the same id. Used for Evidence (default
    `prefix="evidence"`, unchanged for every existing caller), where identical raw provider output
    from the same provider/version/stage must resolve to one stored record -- and, since
    Constitution Article 32/`ARCHITECTURE_TELEMETRY_STANDARD.md` S8.4, for
    `ProvenanceContextEstablished`'s `context_id` (`prefix="context"`), the second content-addressed
    identity in the domain model. `prefix` distinguishes the two id spaces; it is never itself part
    of the hashed content.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return f"{prefix}_{digest.hexdigest()}"
