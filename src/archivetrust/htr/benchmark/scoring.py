"""Scoring: corpus CER/WER with totals, breakdowns and cluster-bootstrap confidence intervals.

Definitions (fixed by `SCORING_VERSION`):
- Primary ("raw") score: reference = `gt_canonical`, hypothesis = `prediction` (NFC + outer strip).
  Character edits are case-, punctuation- and whitespace-sensitive.
- Secondary ("whitespace-normalized") score: `evaluation.metrics.normalize_text` on both sides
  (NFC, whitespace runs collapsed). Reported next to the primary, never instead of it.
- Corpus CER = sum of character edits / sum of reference characters over *all* benchmark lines;
  failed, missing and empty predictions are included as empty hypotheses (all deletions).
  Corpus WER likewise over whitespace-split words. Mean per-line CER is reported as a secondary
  macro average only.
- Confidence intervals resample whole clusters (documents, else pages, else lines -- the unit is
  reported), because lines from one page share hand, ink and scan; resampling lines would give
  intervals that are too narrow. The model difference uses a *paired* bootstrap: the same resampled
  clusters for both models.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from archivetrust.evaluation.metrics import normalize_text
from archivetrust.htr.benchmark.contract import BenchmarkLine
from archivetrust.htr.benchmark.inference import PredictionRecord
from archivetrust.htr.evaluation.recognition import classify_char_edits, classify_word_edits

SCORING_VERSION = "1"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260927
MIN_CLUSTERS = 10


class ScoringError(ValueError):
    pass


@dataclass(frozen=True)
class LineScore:
    line_id: str
    document_id: str
    page_id: str
    collection: str | None
    status: str
    reference: str
    prediction: str
    ref_chars: int
    char_sub: int
    char_ins: int
    char_del: int
    ref_words: int
    word_sub: int
    word_ins: int
    word_del: int
    norm_ref_chars: int
    norm_char_edits: int
    norm_ref_words: int
    norm_word_edits: int

    @property
    def char_edits(self) -> int:
        return self.char_sub + self.char_ins + self.char_del

    @property
    def word_edits(self) -> int:
        return self.word_sub + self.word_ins + self.word_del

    @property
    def cer(self) -> float:
        return self.char_edits / self.ref_chars if self.ref_chars else float(self.char_edits > 0)


def score_line(line: BenchmarkLine, prediction: PredictionRecord) -> LineScore:
    ref, hyp = line.gt_canonical, prediction.prediction
    chars = classify_char_edits(ref, hyp)
    words = classify_word_edits(ref.split(), hyp.split())
    nref, nhyp = normalize_text(ref), normalize_text(hyp)
    nchars = classify_char_edits(nref, nhyp)
    nwords = classify_word_edits(nref.split(), nhyp.split())
    return LineScore(
        line_id=line.line_id, document_id=line.document_id, page_id=line.page_id, collection=line.collection,
        status=prediction.status, reference=ref, prediction=hyp,
        ref_chars=len(ref), char_sub=chars.substitutions, char_ins=chars.insertions, char_del=chars.deletions,
        ref_words=len(ref.split()), word_sub=words.substitutions, word_ins=words.insertions, word_del=words.deletions,
        norm_ref_chars=len(nref), norm_char_edits=nchars.substitutions + nchars.insertions + nchars.deletions,
        norm_ref_words=len(nref.split()), norm_word_edits=nwords.substitutions + nwords.insertions + nwords.deletions,
    )


def check_alignment(lines: Sequence[BenchmarkLine], predictions: Sequence[PredictionRecord], *, manifest_sha256: str) -> None:
    """A prediction file is scorable only if it was produced from exactly this frozen manifest and
    covers every line exactly once with the same image bytes."""
    seen: dict[str, PredictionRecord] = {}
    for p in predictions:
        if p.manifest_sha256 != manifest_sha256:
            raise ScoringError(f"{p.model_id}: prediction for {p.line_id} was made on manifest {p.manifest_sha256[:12]}, not {manifest_sha256[:12]}")
        if p.line_id in seen:
            raise ScoringError(f"{p.model_id}: duplicate prediction for {p.line_id}")
        seen[p.line_id] = p
    expected = {line.line_id: line for line in lines}
    missing, extra = expected.keys() - seen.keys(), seen.keys() - expected.keys()
    if missing or extra:
        raise ScoringError(f"prediction set does not match the manifest: {len(missing)} missing, {len(extra)} extra "
                           "(a partial run cannot be scored as the benchmark)")
    for line_id, p in seen.items():
        if p.image_sha256 != expected[line_id].image_sha256:
            raise ScoringError(f"{line_id}: prediction image hash differs from the manifest")
    if len({(p.model_id, p.decoding_profile) for p in predictions}) > 1:
        raise ScoringError("one prediction file mixes models or decoding profiles")


def aggregate(scores: Sequence[LineScore]) -> dict:
    ref_chars = sum(s.ref_chars for s in scores)
    ref_words = sum(s.ref_words for s in scores)
    char_edits = sum(s.char_edits for s in scores)
    word_edits = sum(s.word_edits for s in scores)
    norm_chars = sum(s.norm_ref_chars for s in scores)
    norm_words = sum(s.norm_ref_words for s in scores)
    status = defaultdict(int)
    for s in scores:
        status[s.status] += 1
    return {
        "lines": len(scores),
        "reference_characters": ref_chars,
        "reference_words": ref_words,
        "cer": char_edits / ref_chars if ref_chars else None,
        "wer": word_edits / ref_words if ref_words else None,
        "char_edits": char_edits,
        "char_substitutions": sum(s.char_sub for s in scores),
        "char_insertions": sum(s.char_ins for s in scores),
        "char_deletions": sum(s.char_del for s in scores),
        "word_edits": word_edits,
        "word_substitutions": sum(s.word_sub for s in scores),
        "word_insertions": sum(s.word_ins for s in scores),
        "word_deletions": sum(s.word_del for s in scores),
        "cer_whitespace_normalized": sum(s.norm_char_edits for s in scores) / norm_chars if norm_chars else None,
        "wer_whitespace_normalized": sum(s.norm_word_edits for s in scores) / norm_words if norm_words else None,
        "exact_line_accuracy": sum(1 for s in scores if s.char_edits == 0) / len(scores) if scores else None,
        "mean_line_cer_macro": sum(s.cer for s in scores) / len(scores) if scores else None,
        "status_counts": dict(status),
        "failed_or_missing_lines": status["failed"] + status["missing"],
        "empty_predictions": status["empty"] + status["failed"] + status["missing"],
    }


def cluster_unit(lines: Sequence[BenchmarkLine]) -> tuple[str, dict[str, str]]:
    documents = {line.document_id for line in lines}
    pages = {(line.document_id, line.page_id) for line in lines}
    if len(documents) >= MIN_CLUSTERS:
        return "document", {line.line_id: line.document_id for line in lines}
    if len(pages) >= MIN_CLUSTERS:
        return "page", {line.line_id: f"{line.document_id}/{line.page_id}" for line in lines}
    return "line", {line.line_id: line.line_id for line in lines}


def _cluster_sums(scores: Sequence[LineScore], cluster_of: dict[str, str], keys: list[str]) -> dict[str, list[tuple[int, int]]]:
    sums: dict[str, list[int]] = {k: [0, 0, 0, 0] for k in keys}
    for s in scores:
        acc = sums[cluster_of[s.line_id]]
        acc[0] += s.char_edits
        acc[1] += s.ref_chars
        acc[2] += s.word_edits
        acc[3] += s.ref_words
    return {k: [(v[0], v[1]), (v[2], v[3])] for k, v in sums.items()}


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(per_model: dict[str, Sequence[LineScore]], lines: Sequence[BenchmarkLine], *, samples: int = BOOTSTRAP_SAMPLES,
              seed: int = BOOTSTRAP_SEED) -> dict:
    """Cluster bootstrap CIs for each model's corpus CER/WER and paired CIs for each difference."""
    unit, cluster_of = cluster_unit(lines)
    keys = sorted(set(cluster_of.values()))
    sums = {model: _cluster_sums(scores, cluster_of, keys) for model, scores in per_model.items()}
    rng = random.Random(seed)
    draws: dict[str, dict[str, list[float]]] = {m: {"cer": [], "wer": []} for m in per_model}
    for _ in range(samples):
        picked = rng.choices(keys, k=len(keys))
        for model in per_model:
            ce = cc = we = wc = 0
            for key in picked:
                (e1, c1), (e2, c2) = sums[model][key]
                ce, cc, we, wc = ce + e1, cc + c1, we + e2, wc + c2
            draws[model]["cer"].append(ce / cc if cc else 0.0)
            draws[model]["wer"].append(we / wc if wc else 0.0)
    out = {"cluster_unit": unit, "clusters": len(keys), "samples": samples, "seed": seed, "interval": "95% percentile",
           "per_model": {}, "paired_differences": {}}
    if unit == "line":
        out["warning"] = (f"fewer than {MIN_CLUSTERS} documents and pages: resampling individual lines, which ignores "
                          "within-page correlation -- intervals are too narrow")
    for model, d in draws.items():
        out["per_model"][model] = {metric: [_percentile(v, 0.025), _percentile(v, 0.975)] for metric, v in d.items()}
    models = sorted(per_model)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            entry = {}
            for metric in ("cer", "wer"):
                diffs = [x - y for x, y in zip(draws[a][metric], draws[b][metric], strict=True)]
                entry[metric] = {"difference": f"{a} minus {b}", "ci95": [_percentile(diffs, 0.025), _percentile(diffs, 0.975)],
                                 "share_of_resamples_a_lower": sum(1 for x in diffs if x < 0) / len(diffs)}
            out["paired_differences"][f"{a}__vs__{b}"] = entry
    return out


def breakdown(scores: Sequence[LineScore], key: str) -> dict[str, dict]:
    groups: dict[str, list[LineScore]] = defaultdict(list)
    for s in scores:
        groups[str(getattr(s, key) or "(none)")].append(s)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def line_score_rows(scores: Sequence[LineScore]) -> list[dict]:
    return [asdict(s) | {"char_edits": s.char_edits, "word_edits": s.word_edits, "cer": s.cer} for s in scores]
