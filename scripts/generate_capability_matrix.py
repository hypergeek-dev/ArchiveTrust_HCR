#!/usr/bin/env python
"""Regenerates (or checks) `docs/CAPABILITY_MATRIX_HTR.md`'s adapter-reported tables.

The rendering itself lives in `src/archivetrust/providers/htr_capability_matrix.py` so
`tests/providers/test_htr_capability_matrix.py` can drive the identical code path -- this file is the
CLI over it, nothing more.

    # rewrite the generated region from the live adapters
    PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_capability_matrix.py

    # fail if the committed document has drifted (what CI/the test suite asserts)
    PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_capability_matrix.py --check

Only the region between `BEGIN GENERATED FROM ADAPTERS` and `END GENERATED FROM ADAPTERS` is touched.
Everything else in that document is prose about what the flags mean and where they mislead, which no
generator can write and a whole-file regeneration would delete.

`docs/CAPABILITY_MATRIX.md` (the OCR-era document) is deliberately **not** touched: it carries a
supersession notice pointing at the new file and its historical rating table is an accurate record of a
decision that was made, not a claim about the current codebase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.providers.htr_capability_matrix import (  # noqa: E402
    build_htr_adapters,
    regenerate_document,
    split_document,
)

MATRIX_PATH = REPO_ROOT / "docs" / "CAPABILITY_MATRIX_HTR.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the committed document differs from what the adapters report",
    )
    args = parser.parse_args()

    if not MATRIX_PATH.exists():
        raise SystemExit(f"Missing {MATRIX_PATH.relative_to(REPO_ROOT)}")

    adapters = build_htr_adapters()
    current = MATRIX_PATH.read_text(encoding="utf-8")
    regenerated = regenerate_document(current, adapters)

    if args.check:
        if current == regenerated:
            print(
                f"OK: {MATRIX_PATH.relative_to(REPO_ROOT)} matches what "
                f"{len(adapters)} adapters report."
            )
            return 0
        _, expected_region, _ = split_document(regenerated)
        _, actual_region, _ = split_document(current)
        print(
            f"DRIFT: {MATRIX_PATH.relative_to(REPO_ROOT)}'s generated region does not match the "
            "adapters' own get_capabilities()/get_metadata().\n"
        )
        print("--- committed ---")
        print(actual_region)
        print("\n--- adapters report ---")
        print(expected_region)
        print(
            "\nFix by re-running this script without --check (if the adapters are right) or by "
            "correcting the adapter (if the document was right)."
        )
        return 1

    if current == regenerated:
        print(f"{MATRIX_PATH.relative_to(REPO_ROOT)} already up to date.")
        return 0
    MATRIX_PATH.write_text(regenerated, encoding="utf-8")
    print(f"Rewrote the generated region of {MATRIX_PATH.relative_to(REPO_ROOT)}.")
    for adapter in adapters:
        metadata = adapter.get_metadata()
        print(f"  {metadata.method_id:28s} @ {metadata.model_revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
