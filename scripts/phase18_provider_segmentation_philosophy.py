"""Phase 18 -- Provider Segmentation Philosophy Model (2026-07-15).

Measurement-only, per this phase's own charter: no threshold, comparison rule, confidence logic,
or Review Center behavior is touched. Unlike Phases 14-17 (which examined *contested* packets
only), this phase profiles **every** `ObservationCreated`/`EvidenceCreated` event pair in the
corpus, corpus-wide, per provider -- the full population each provider produced, not just the
subset that ended up disagreeing with another provider.

Single streaming pass over `events.jsonl` (2.37M lines, ~195k Observation/Evidence pairs) --
no replay, no OCR, no provider invocation.

Run: `python scripts/phase18_provider_segmentation_philosophy.py`
Writes: `benchmarks/phase18_provider_segmentation_philosophy.json`
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_PATH = (
    REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
    / "telemetry" / "events.jsonl"
)
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "phase18_provider_segmentation_philosophy.json"

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split()).lower()


def _extract_text(payload: dict) -> str | None:
    """Text-bearing ontology payloads (paragraph, heading, caption, footnote, table_cell, ...)
    carry a `text` field; structural/non-text payloads (image, table, layout_region, ...) do
    not -- `None` is a real, expected outcome for those, not a parsing failure."""
    text = payload.get("text")
    return text if isinstance(text, str) and text else None


def _bbox_area(bbox: dict) -> float:
    return max(0.0, bbox["x1"] - bbox["x0"]) * max(0.0, bbox["y1"] - bbox["y0"])


def build_profiles() -> dict:
    # observation_id -> (provider_id, observation_type, text_len, word_count, sentence_count, evidence_ids)
    observations: dict[str, dict] = {}
    # evidence_id -> (bbox_area, width, height, page, document_ref)
    evidence: dict[str, dict] = {}
    doc_pages_by_provider: dict[str, set[tuple[str, int]]] = defaultdict(set)  # (doc, page) seen per provider
    obs_count_by_doc_page_provider: Counter = Counter()

    with TELEMETRY_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if '"kind":"ObservationCreated"' in line:
                obj = json.loads(line)
                obs = obj["observation"]
                payload = obs.get("payload") or {}
                text = _extract_text(payload)
                observations[obs["observation_id"]] = {
                    "provider_id": obs["provider_id"],
                    "observation_type": obs["observation_type"],
                    "text": text,
                    "evidence_ids": obs.get("evidence_ids") or [],
                    "document_ref": obj["document_ref"],
                }
            elif '"kind":"EvidenceCreated"' in line:
                obj = json.loads(line)
                ev = obj["evidence"]
                bbox = ev.get("bounding_box")
                evidence[ev["evidence_id"]] = {
                    "bbox": bbox,
                    "page": ev.get("page"),
                }

    # Join and aggregate per provider.
    per_provider_text_lengths: dict[str, list[int]] = defaultdict(list)
    per_provider_word_counts: dict[str, list[int]] = defaultdict(list)
    per_provider_sentence_counts: dict[str, list[int]] = defaultdict(list)
    per_provider_bbox_areas: dict[str, list[float]] = defaultdict(list)
    per_provider_bbox_widths: dict[str, list[float]] = defaultdict(list)
    per_provider_bbox_heights: dict[str, list[float]] = defaultdict(list)
    per_provider_type_counts: dict[str, Counter] = defaultdict(Counter)
    per_provider_doc_page_obs: dict[str, Counter] = defaultdict(Counter)

    for obs_id, obs in observations.items():
        provider = obs["provider_id"]
        per_provider_type_counts[provider][obs["observation_type"]] += 1

        if obs["text"]:
            t = obs["text"]
            per_provider_text_lengths[provider].append(len(t))
            per_provider_word_counts[provider].append(len(_WORD_RE.findall(_norm(t))))
            sentences = [p for p in _SENTENCE_SPLIT_RE.split(t) if p.strip()]
            per_provider_sentence_counts[provider].append(max(len(sentences), 1))

        for eid in obs["evidence_ids"]:
            ev = evidence.get(eid)
            if ev is None:
                continue
            if ev["bbox"]:
                per_provider_bbox_areas[provider].append(_bbox_area(ev["bbox"]))
                per_provider_bbox_widths[provider].append(ev["bbox"]["x1"] - ev["bbox"]["x0"])
                per_provider_bbox_heights[provider].append(ev["bbox"]["y1"] - ev["bbox"]["y0"])
            if ev["page"] is not None:
                per_provider_doc_page_obs[provider][(obs["document_ref"], ev["page"])] += 1

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        arr = np.array(values, dtype=float)
        return {
            "n": len(arr),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(arr.std()), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
        }

    providers = sorted(per_provider_type_counts.keys())
    profiles = {}
    for p in providers:
        density_counts = list(per_provider_doc_page_obs[p].values())
        profiles[p] = {
            "total_observations": sum(per_provider_type_counts[p].values()),
            "observation_type_mix": dict(per_provider_type_counts[p].most_common()),
            "text_length_chars": _stats(per_provider_text_lengths[p]),
            "word_count": _stats(per_provider_word_counts[p]),
            "sentence_count": _stats(per_provider_sentence_counts[p]),
            "bbox_area": _stats(per_provider_bbox_areas[p]),
            "bbox_width": _stats(per_provider_bbox_widths[p]),
            "bbox_height": _stats(per_provider_bbox_heights[p]),
            "observations_per_document_page": _stats(density_counts),
            "distinct_document_pages_touched": len(per_provider_doc_page_obs[p]),
        }

    return {
        "total_observations_processed": len(observations),
        "total_evidence_processed": len(evidence),
        "providers": providers,
        "profiles": profiles,
    }


def main() -> None:
    result = build_profiles()
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps({k: v for k, v in result.items() if k != "profiles"}, indent=2))
    for p, prof in result["profiles"].items():
        print(f"\n=== {p} ===")
        print(json.dumps(prof, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
