"""Character/vocabulary compatibility between the pilot's selected Swedish transcriptions and the
generic Loghi-HTR checkpoint's own charlist (docs/methods/loghi-swedish-finetuning.md, Work Package 6).

**Never trust a terminal for this.** Verifying character content by `print()`-ing it during this
integration silently corrupted `ö`/`Æ`/etc. into `�` in the terminal's own rendering, while the
underlying Python string (and the parquet round-trip) were correct the whole time -- confirmed by
dumping raw codepoints to a UTF-8 file and reading them back. Every function here operates on real
`str` codepoints and writes findings to files; nothing in this module's *correctness* depends on how a
terminal happens to render the output.

Reuses `swedish_dataset_inventory.py`'s already-computed per-row `unique_characters`/
`unsupported_characters` fields (computed against the same charlist at inventory time) rather than
re-deriving them -- this module aggregates across the pilot's selected lines, it does not recompute
per-character membership from scratch a second time.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict


class CharacterFrequency(BaseModel):
    model_config = ConfigDict(frozen=True)

    character: str
    codepoint: str
    """`U+00F6` style, never the raw character alone in any report meant to be read reliably."""
    unicode_name: str | None
    unicode_category: str
    frequency: int
    supported_by_generic_checkpoint: bool


class CharacterCompatibilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_lines_examined: int
    total_distinct_characters: int
    unsupported_character_count: int
    lines_with_any_unsupported_character: int
    frequencies: tuple[CharacterFrequency, ...]
    unsupported_characters: tuple[CharacterFrequency, ...]
    charlist_path: str
    charlist_size: int
    compatible: bool
    """`True` only when every character in the examined lines is in the charlist -- the brief's hard
    gate: "Do not start the real training session if the character mapping is incomplete or
    ambiguous." A caller must check this before proceeding to training-data preparation."""


def _codepoint_str(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def build_character_compatibility_report(
    *,
    inventory_path: str | Path,
    manifest_paths: list[str | Path],
    charlist_path: str | Path,
) -> CharacterCompatibilityReport:
    """Aggregates the real character inventory across every line named in `manifest_paths` (the pilot
    train + val manifests), against the real downloaded charlist -- never a sample, never estimated.
    """
    charlist_text = Path(charlist_path).read_text(encoding="utf-8")
    charlist = set(charlist_text)

    line_ids: set[str] = set()
    for manifest_path in manifest_paths:
        table = pq.read_table(manifest_path, columns=["line_id"])
        line_ids.update(table.column("line_id").to_pylist())

    inventory_table = pq.read_table(
        inventory_path, columns=["line_id", "transcription"]
    )
    all_line_ids = inventory_table.column("line_id").to_pylist()
    all_transcriptions = inventory_table.column("transcription").to_pylist()

    counter: Counter[str] = Counter()
    lines_examined = 0
    lines_with_unsupported = 0
    for line_id, transcription in zip(all_line_ids, all_transcriptions):
        if line_id not in line_ids or transcription is None:
            continue
        lines_examined += 1
        counter.update(transcription)
        if any(ch not in charlist for ch in transcription):
            lines_with_unsupported += 1

    frequencies: list[CharacterFrequency] = []
    unsupported: list[CharacterFrequency] = []
    for ch, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = None
        entry = CharacterFrequency(
            character=ch,
            codepoint=_codepoint_str(ch),
            unicode_name=name,
            unicode_category=unicodedata.category(ch),
            frequency=freq,
            supported_by_generic_checkpoint=ch in charlist,
        )
        frequencies.append(entry)
        if not entry.supported_by_generic_checkpoint:
            unsupported.append(entry)

    return CharacterCompatibilityReport(
        total_lines_examined=lines_examined,
        total_distinct_characters=len(counter),
        unsupported_character_count=len(unsupported),
        lines_with_any_unsupported_character=lines_with_unsupported,
        frequencies=tuple(frequencies),
        unsupported_characters=tuple(unsupported),
        charlist_path=str(charlist_path),
        charlist_size=len(charlist),
        compatible=not unsupported,
    )


def write_character_compatibility_report(
    report: CharacterCompatibilityReport, output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def render_human_readable(report: CharacterCompatibilityReport) -> str:
    """A plain-text rendering safe to write to a file and read back (never printed to a terminal
    during this integration, per the module docstring's own lesson)."""
    lines = [
        f"Lines examined: {report.total_lines_examined}",
        f"Distinct characters: {report.total_distinct_characters}",
        f"Charlist: {report.charlist_path} ({report.charlist_size} characters)",
        f"Compatible: {report.compatible}",
        f"Unsupported characters: {report.unsupported_character_count}",
        f"Lines containing >=1 unsupported character: {report.lines_with_any_unsupported_character}",
        "",
        "Unsupported character detail:",
    ]
    for entry in report.unsupported_characters:
        lines.append(
            f"  {entry.codepoint} ({entry.unicode_name or 'UNNAMED'}, category={entry.unicode_category}) "
            f"-- frequency {entry.frequency}"
        )
    lines.append("")
    lines.append("Top 40 most frequent characters overall:")
    for entry in report.frequencies[:40]:
        lines.append(
            f"  {entry.codepoint} ({entry.unicode_name or 'UNNAMED'}) freq={entry.frequency} "
            f"supported={entry.supported_by_generic_checkpoint}"
        )
    return "\n".join(lines)
