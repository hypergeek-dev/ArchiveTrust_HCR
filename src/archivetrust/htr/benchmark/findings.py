"""Findings -- the one vocabulary every benchmark stage uses to say what is wrong with an input.

Severity decides what happens next, never a guess by the harness:

- `blocker`       the dataset cannot be frozen until this is fixed or the affected lines are excluded.
- `needs_review`  a human must decide (the line goes to the review queue); no automatic fix is applied.
- `warning`       recorded and reported, does not block freezing (e.g. an auto-normalization that ran).
- `info`          descriptive facts (formats detected, counts).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["blocker", "needs_review", "warning", "info"]
SEVERITY_ORDER: tuple[Severity, ...] = ("blocker", "needs_review", "warning", "info")


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    severity: Severity
    code: str
    """Stable machine-readable identifier, e.g. `gt.embedded_newline` or `image.unreadable`."""
    message: str
    path: str | None = None
    """Relative path (forward slashes) of the file the finding is about, when there is one."""
    line_id: str | None = None
    detail: dict = Field(default_factory=dict)


def summarize(findings: Iterable[Finding]) -> dict:
    findings = list(findings)
    by_severity = Counter(f.severity for f in findings)
    by_code = Counter(f.code for f in findings)
    return {
        "counts_by_severity": {s: by_severity.get(s, 0) for s in SEVERITY_ORDER},
        "counts_by_code": dict(sorted(by_code.items())),
        "blocking": by_severity.get("blocker", 0) > 0,
    }


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    rank = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    return sorted(findings, key=lambda f: (rank[f.severity], f.code, f.path or "", f.line_id or ""))


def findings_to_jsonl(findings: Iterable[Finding]) -> str:
    return "".join(
        json.dumps(f.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for f in sort_findings(findings)
    )


def findings_to_markdown(findings: Iterable[Finding], *, max_per_code: int = 20) -> str:
    """Grouped by severity then code; long groups are truncated with a count (the JSONL is complete)."""
    ordered = sort_findings(findings)
    if not ordered:
        return "No findings.\n"
    out: list[str] = []
    for severity in SEVERITY_ORDER:
        group = [f for f in ordered if f.severity == severity]
        if not group:
            continue
        out.append(f"### {severity} ({len(group)})\n")
        codes = Counter(f.code for f in group)
        for code in sorted(codes):
            items = [f for f in group if f.code == code]
            out.append(f"- **{code}** ({len(items)})")
            for f in items[:max_per_code]:
                where = f.line_id or f.path or ""
                out.append(f"  - {'`' + where + '`: ' if where else ''}{f.message}")
            if len(items) > max_per_code:
                out.append(f"  - ... {len(items) - max_per_code} more (see findings.jsonl)")
        out.append("")
    return "\n".join(out) + "\n"
