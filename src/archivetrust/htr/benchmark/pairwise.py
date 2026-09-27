"""Pairwise error analysis between two models on the same lines.

`align` reproduces the exact DP + backtrace of `htr.evaluation.recognition._classify_edits`, but
returns the operations instead of counting them, so every error tag here adds up to exactly the edit
counts in the scores (tested).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from archivetrust.htr.benchmark.scoring import LineScore

Op = tuple[Literal["match", "sub", "ins", "del"], str, str, int]
"""(operation, reference char or "", hypothesis char or "", reference index the op sits at)."""

CATASTROPHIC_LINE_CER = 0.5
SWEDISH_LETTERS = frozenset("åäöÅÄÖéÉüÜ")
_ABBREVIATION_WORD = re.compile(r"\S*\w[:.]\S*")


def align(reference: str, hypothesis: str) -> list[Op]:
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", reference[i - 1], hypothesis[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", reference[i - 1], hypothesis[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", reference[i - 1], "", i - 1))
            i -= 1
        else:
            ops.append(("ins", "", hypothesis[j - 1], i))
            j -= 1
    ops.reverse()
    return ops


def _base(char: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", char) if not unicodedata.combining(c))


def _kind(char: str) -> str:
    if not char:
        return ""
    if char.isspace():
        return "space"
    category = unicodedata.category(char)
    if category.startswith("P"):
        return "punctuation"
    if category.startswith("N"):
        return "digit"
    if char in SWEDISH_LETTERS:
        return "swedish_letter"
    if category.startswith("L") and char.isascii():
        return "ascii_letter"
    return "special"


def tag_op(op: Op, reference: str) -> str:
    """One tag per error, most specific first: spacing, case, diacritic, punctuation, digit,
    special/historical character, abbreviation context, other letter."""
    kind, r, h, index = op
    kinds = {_kind(r), _kind(h)} - {""}
    if "space" in kinds:
        return "spacing"
    if kind == "sub" and r.lower() == h.lower():
        return "case"
    if kind == "sub" and _base(r) == _base(h):
        return "diacritic"
    if "punctuation" in kinds:
        return "punctuation"
    if "digit" in kinds:
        return "digit"
    if "special" in kinds:
        return "special_character"
    for match in _ABBREVIATION_WORD.finditer(reference):
        if match.start() <= index < match.end():
            return "abbreviation_word"
    return "letter"


def error_profile(scores: Sequence[LineScore]) -> dict:
    tags: Counter[str] = Counter()
    confusions: Counter[str] = Counter()
    for s in scores:
        for op in align(s.reference, s.prediction):
            if op[0] == "match":
                continue
            tags[tag_op(op, s.reference)] += 1
            confusions[f"{op[1] or '∅'}→{op[2] or '∅'}"] += 1
    return {"tags": dict(tags.most_common()), "top_confusions": confusions.most_common(40)}


def categorize(a: LineScore, b: LineScore) -> str:
    a_ok, b_ok = a.char_edits == 0, b.char_edits == 0
    if a_ok and b_ok:
        return "both_correct"
    if a_ok:
        return "only_a_correct"
    if b_ok:
        return "only_b_correct"
    if a.prediction == b.prediction:
        return "both_wrong_identically"
    if a.char_edits < b.char_edits:
        return "both_wrong_a_better"
    if b.char_edits < a.char_edits:
        return "both_wrong_b_better"
    return "both_wrong_equal_edits"


def catastrophic(s: LineScore) -> bool:
    return s.status in ("failed", "missing") or (s.ref_chars > 0 and s.cer >= CATASTROPHIC_LINE_CER)


def compare(a_name: str, a_scores: Sequence[LineScore], b_name: str, b_scores: Sequence[LineScore], *, sample_size: int = 30) -> dict:
    b_by_id = {s.line_id: s for s in b_scores}
    pairs = [(a, b_by_id[a.line_id]) for a in a_scores]
    categories = Counter(categorize(a, b) for a, b in pairs)
    better = [(a, b) for a, b in pairs if a.char_edits != b.char_edits]
    ranked = sorted(better, key=lambda p: (-abs(p[0].char_edits - p[1].char_edits), p[0].line_id))

    def row(a: LineScore, b: LineScore) -> dict:
        return {"line_id": a.line_id, "reference": a.reference, a_name: a.prediction, b_name: b.prediction,
                f"{a_name}_edits": a.char_edits, f"{b_name}_edits": b.char_edits}

    return {
        "a": a_name, "b": b_name, "lines": len(pairs),
        "categories": {k.replace("_a_", f"_{a_name}_").replace("_b_", f"_{b_name}_").replace("only_a", f"only_{a_name}")
                       .replace("only_b", f"only_{b_name}"): v for k, v in sorted(categories.items())},
        "lines_a_fewer_edits": sum(1 for a, b in pairs if a.char_edits < b.char_edits),
        "lines_b_fewer_edits": sum(1 for a, b in pairs if b.char_edits < a.char_edits),
        "catastrophic": {a_name: sum(1 for a, _ in pairs if catastrophic(a)), b_name: sum(1 for _, b in pairs if catastrophic(b)),
                         "both": sum(1 for a, b in pairs if catastrophic(a) and catastrophic(b))},
        "error_profile": {a_name: error_profile(a_scores), b_name: error_profile(b_scores)},
        "disagreement_sample": {
            f"{a_name}_much_better": [row(a, b) for a, b in ranked if a.char_edits < b.char_edits][:sample_size],
            f"{b_name}_much_better": [row(a, b) for a, b in ranked if b.char_edits < a.char_edits][:sample_size],
        },
    }
