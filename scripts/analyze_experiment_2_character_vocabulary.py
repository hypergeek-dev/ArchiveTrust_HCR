"""Phase 3 character-vocabulary verification for Experiment 2 (scratch training).

Loghi's own DataManager builds the from-scratch tokenizer automatically by scanning every line of
--train_list and collecting every character it encounters (data/manager.py::_process_raw_data,
_is_valid_ground_truth) -- there is no --charlist flag to pass for a scratch run. This script
independently re-derives that same vocabulary directly from the real 562,123-line train list Loghi
will actually read, verifies it against the real 1,000-line validation list, and persists the result
so Experiment 2 does not silently inherit anything and so this is auditable before launch.

Read-only w.r.t. the sealed reserved test manifest -- never opened.

Usage:
    python scripts/analyze_experiment_2_character_vocabulary.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = REPO_ROOT / "training" / "_prepared_data" / "fc21a708_e12796a0"
TRAIN_LIST = POOL_DIR / "train_list.txt"
VAL_LIST = POOL_DIR / "val_list.txt"
OUT_PATH = REPO_ROOT / "training" / "experiment_2_character_inventory.json"

COMBINING_MARK_CATEGORIES = {"Mn", "Mc", "Me"}
PUNCTUATION_CATEGORIES = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
WHITESPACE_CATEGORIES = {"Zs", "Zl", "Zp"}


def load_transcriptions(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            texts.append(fields[-1])
    return texts


def char_report(c: str) -> dict:
    cp = ord(c)
    try:
        name = unicodedata.name(c)
    except ValueError:
        name = "UNNAMED"
    category = unicodedata.category(c)
    return {
        "char": c,
        "codepoint": f"U+{cp:04X}",
        "unicode_name": name,
        "category": category,
        "is_combining_mark": category in COMBINING_MARK_CATEGORIES,
        "is_punctuation": category in PUNCTUATION_CATEGORIES,
        "is_whitespace": category in WHITESPACE_CATEGORIES or c.isspace(),
    }


def main() -> int:
    train_texts = load_transcriptions(TRAIN_LIST)
    val_texts = load_transcriptions(VAL_LIST)

    train_chars: collections.Counter = collections.Counter()
    for t in train_texts:
        train_chars.update(t)
    val_chars: collections.Counter = collections.Counter()
    for t in val_texts:
        val_chars.update(t)

    train_set = set(train_chars)
    val_set = set(val_chars)
    val_only = val_set - train_set
    train_only = train_set - val_set

    # The exact vocabulary Loghi's DataManager will build for a from-scratch run:
    # Tokenizer(sorted(characters)) where `characters` is every character seen scanning --train_list.
    # Python's sorted() on str is codepoint order -- the same order Tokenizer.__init__ receives it in.
    scratch_vocabulary = sorted(train_set)

    all_chars_seen = train_set | val_set
    combining_marks = sorted(c for c in all_chars_seen if unicodedata.category(c) in COMBINING_MARK_CATEGORIES)
    punctuation = sorted(c for c in all_chars_seen if unicodedata.category(c) in PUNCTUATION_CATEGORIES)
    whitespace = sorted(c for c in all_chars_seen if unicodedata.category(c) in WHITESPACE_CATEGORIES or c.isspace())
    rare_train_chars = sorted([(c, n) for c, n in train_chars.items() if n <= 5], key=lambda x: x[1])

    # Unicode normalization form check: is the corpus consistently NFC, NFD, both, or neither?
    def normalization_form_of_text(texts: list[str]) -> dict:
        nfc_count = sum(1 for t in texts if unicodedata.is_normalized("NFC", t))
        nfd_count = sum(1 for t in texts if unicodedata.is_normalized("NFD", t))
        return {"nfc_normalized_lines": nfc_count, "nfd_normalized_lines": nfd_count, "total_lines": len(texts)}

    train_norm = normalization_form_of_text(train_texts)
    val_norm = normalization_form_of_text(val_texts)

    # Abbreviation marks / historical symbols: flag characters outside the common Latin-1/Latin
    # Extended-A ranges plus common punctuation, as candidates worth a human look (not an exhaustive
    # taxonomy -- Unicode has no single "historical symbol" or "abbreviation mark" category).
    notable_candidates = sorted(
        c for c in all_chars_seen
        if unicodedata.category(c) not in ({"Ll", "Lu"} | PUNCTUATION_CATEGORIES | WHITESPACE_CATEGORIES | {"Nd"})
        and c not in combining_marks
    )

    charlist_text = "".join(scratch_vocabulary)
    charlist_sha256 = hashlib.sha256(charlist_text.encode("utf-8")).hexdigest()

    report = {
        "generated_from": {
            "train_list_path": str(TRAIN_LIST),
            "val_list_path": str(VAL_LIST),
            "train_line_count": len(train_texts),
            "val_line_count": len(val_texts),
        },
        "unique_training_characters": len(train_set),
        "unique_validation_characters": len(val_set),
        "validation_only_characters": sorted(val_only),
        "validation_only_character_count": len(val_only),
        "train_only_characters_count": len(train_only),
        "character_frequencies_train": {c: n for c, n in train_chars.most_common()},
        "character_frequencies_val": {c: n for c, n in val_chars.most_common()},
        "rare_train_characters_count_le_5": [{"char": c, "count": n, **char_report(c)} for c, n in rare_train_chars],
        "combining_marks": [char_report(c) for c in combining_marks],
        "punctuation": [char_report(c) for c in punctuation],
        "whitespace_chars": [char_report(c) for c in whitespace],
        "notable_non_letter_non_digit_non_punct_candidates": [char_report(c) for c in notable_candidates],
        "unicode_normalization_check": {"train": train_norm, "validation": val_norm},
        "scratch_model_vocabulary": {
            "note": "The exact vocabulary Loghi's DataManager will build automatically for a "
                    "from-scratch run (data/manager.py::_process_raw_data) -- sorted(set of every "
                    "character seen scanning --train_list). No explicit --charlist is passed or needed.",
            "size": len(scratch_vocabulary),
            "characters": scratch_vocabulary,
            "sha256": charlist_sha256,
        },
        "evaluation_partition_risk_check": {
            "description": "Loghi's 'evaluation' partition (aliased to --validation_list, used as "
                            "Keras's actual validation_data during training) silently drops any line "
                            "whose ground truth contains a character outside the train-derived "
                            "vocabulary (data/manager.py::_is_valid_ground_truth). Validation-only "
                            "characters would mean those lines vanish from the in-training CER metric "
                            "with no warning.",
            "validation_only_character_count": len(val_only),
            "result": "PASS -- zero validation-only characters; no validation lines will be silently "
                      "dropped from the in-training evaluation partition." if len(val_only) == 0
                      else "FAIL -- validation contains characters absent from training; see "
                           "validation_only_characters above.",
        },
        "launch_gate": {
            "passed": len(val_only) == 0,
            "reason": "no unsupported validation characters" if len(val_only) == 0
                      else f"{len(val_only)} validation-only characters would be silently dropped "
                           "from in-training validation",
        },
    }

    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT_PATH}")
    print(f"unique training characters: {len(train_set)}")
    print(f"unique validation characters: {len(val_set)}")
    print(f"validation-only characters: {len(val_only)}")
    print(f"scratch vocabulary size: {len(scratch_vocabulary)}")
    print(f"scratch vocabulary sha256: {charlist_sha256}")
    print(f"launch gate: {'PASS' if report['launch_gate']['passed'] else 'FAIL'}")
    return 0 if report["launch_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
