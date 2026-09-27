"""Ground-truth normalization protocol -- the *only* transformations the benchmark may apply to a
supplied transcription, each named, versioned, and recorded per line.

Source GT (exactly as supplied) is always preserved next to canonical GT. Nothing lexical is ever
done here: no spelling modernization, no abbreviation expansion, no punctuation or case changes, no
removal of diacritics or historical letterforms, and never the removal of a character because a model
cannot emit it. A character a model cannot produce is a real error for that model.

Predictions get the same Unicode canonicalization (`canonicalize_prediction`), so a model is not
charged for emitting a decomposed `a` + combining ring where the reference holds a precomposed `å`,
plus removal of leading/trailing whitespace -- a decoder artifact (BPE detokenizers can emit a leading
space), applied identically to both models and counted in the report. Nothing else.
"""

from __future__ import annotations

import unicodedata

PROTOCOL_ID = "gt-normalization/1"

RULE_NFC = "unicode_nfc"
"""Unicode canonical composition. Changes code points, never the letter a reader sees."""

RULE_STRIP_FILE_LINE_TERMINATOR = "strip_file_line_terminator"
"""Remove exactly one trailing `\\n` or `\\r\\n` that ends a per-line transcription *file*. That
terminator belongs to the file format, not to the transcription. Only applied by adapters that read
one transcription per text file; never to text taken from XML or a table cell."""

RULE_HUMAN_CORRECTION = "human_correction"
"""The reviewer replaced the transcription through a recorded decision (`decisions.jsonl`), made
while building the candidate -- i.e. before the freeze and before any model output exists. The
delivered text stays in `gt_source`."""

RULES: dict[str, str] = {
    RULE_NFC: "Unicode NFC composition (code-point spelling only; no visible change).",
    RULE_STRIP_FILE_LINE_TERMINATOR: (
        "One trailing newline that terminates a per-line .txt file is removed (file format, not text)."
    ),
    RULE_HUMAN_CORRECTION: "Transcription replaced by a recorded pre-freeze reviewer decision (never informed by model output).",
}

NOT_APPLIED: tuple[str, ...] = (
    "whitespace collapsing or outer-whitespace stripping (flagged for review instead)",
    "case folding",
    "punctuation removal or substitution",
    "diacritic removal",
    "historical-character replacement (e.g. long s, abbreviation marks)",
    "abbreviation expansion or spelling modernization",
    "removal of characters outside any model's vocabulary",
)


def strip_file_line_terminator(text: str) -> tuple[str, bool]:
    if text.endswith("\r\n"):
        return text[:-2], True
    if text.endswith("\n"):
        return text[:-1], True
    return text, False


def canonicalize_gt(source_text: str, *, from_line_file: bool = False) -> tuple[str, tuple[str, ...]]:
    """Returns `(canonical_text, rules_that_changed_it)`. A rule that ran but changed nothing is not
    listed, so a non-empty tuple always means the canonical text differs from the source."""
    applied: list[str] = []
    text = source_text
    if from_line_file:
        text, changed = strip_file_line_terminator(text)
        if changed:
            applied.append(RULE_STRIP_FILE_LINE_TERMINATOR)
    composed = unicodedata.normalize("NFC", text)
    if composed != text:
        applied.append(RULE_NFC)
    return composed, tuple(applied)


PREDICTION_RULES = ("unicode_nfc", "strip_outer_whitespace")


def canonicalize_prediction(prediction: str) -> str:
    """NFC + outer-whitespace strip, identical for every model. Inner whitespace is untouched."""
    return unicodedata.normalize("NFC", prediction).strip()


def protocol_record() -> dict:
    return {"protocol_id": PROTOCOL_ID, "rules": RULES, "explicitly_not_applied": list(NOT_APPLIED),
            "prediction_rules": list(PREDICTION_RULES)}
