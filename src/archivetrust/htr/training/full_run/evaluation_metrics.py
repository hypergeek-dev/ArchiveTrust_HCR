"""CER/WER on true Levenshtein edit distance, plus confidence-calibration bucketing.

**Why this module exists.** A manual inspection during the first shard review used
`difflib.SequenceMatcher` to approximate CER and reported 0.82 where the real figure was 0.25 --
a 3x overstatement that briefly looked like a contradiction of the container's own metric.
`difflib` finds *matching blocks*, not a minimum edit script: it never considers substitutions and
cannot cross a mismatch to find a better alignment, so it systematically overestimates distance.
CER is defined on Levenshtein distance, so Levenshtein is what this computes.

**Corpus-level, not mean-of-per-line.** `sum(edits) / sum(reference_chars)` weights every character
equally, which is what the container's own `CERMetric` reports and therefore what a comparison
against it must use. Averaging per-line CER instead over-weights short lines -- and short lines are
exactly where this corpus is weakest (`alvsborgs_losen`'s terse tax entries), so the two differ
materially rather than academically. Both are returned, labelled, so a reader can see the gap.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def levenshtein(a: str, b: str) -> int:
    """Minimum single-character insertions, deletions and substitutions turning `a` into `b`."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over word tokens rather than characters -- the same distance, different alphabet."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # Map each distinct word to a single sentinel character so the character routine can be reused.
    vocab = {w: chr(i + 256) for i, w in enumerate(dict.fromkeys(ref + hyp))}
    return levenshtein("".join(vocab[w] for w in ref), "".join(vocab[w] for w in hyp)) / len(ref)


class CorpusMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_count: int
    corpus_cer: float
    """`total edits / total reference characters` -- comparable with the container's CERMetric."""
    corpus_wer: float
    mean_per_line_cer: float
    """Reported alongside `corpus_cer` because they diverge when line lengths vary, and this corpus's
    weakest subgroup is also its shortest-line subgroup."""
    total_reference_chars: int
    total_edits: int


def corpus_metrics(pairs: list[tuple[str, str]]) -> CorpusMetrics:
    """`pairs` is `[(reference, hypothesis), ...]`."""
    if not pairs:
        return CorpusMetrics(sample_count=0, corpus_cer=0.0, corpus_wer=0.0,
                             mean_per_line_cer=0.0, total_reference_chars=0, total_edits=0)
    edits = sum(levenshtein(r, h) for r, h in pairs)
    chars = sum(len(r) for r, _ in pairs)
    word_edits = sum(word_error_rate(r, h) * max(len(r.split()), 1) for r, h in pairs)
    words = sum(max(len(r.split()), 1) for r, _ in pairs)
    return CorpusMetrics(
        sample_count=len(pairs),
        corpus_cer=edits / max(chars, 1),
        corpus_wer=word_edits / max(words, 1),
        mean_per_line_cer=sum(character_error_rate(r, h) for r, h in pairs) / len(pairs),
        total_reference_chars=chars,
        total_edits=edits,
    )


class ConfidenceBucket(BaseModel):
    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    sample_count: int
    mean_cer: float | None
    median_cer: float | None


def confidence_buckets(
    records: list[tuple[float, float]], *, edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)
) -> list[ConfidenceBucket]:
    """`records` is `[(confidence, cer), ...]`. Empty buckets are reported with `None` statistics
    rather than silently dropped -- a gap in coverage is itself calibration evidence."""
    from statistics import mean, median

    out = []
    for lo, hi in zip(edges, edges[1:]):
        vals = [cer for conf, cer in records if lo <= conf < hi]
        out.append(ConfidenceBucket(
            lower=lo, upper=hi, sample_count=len(vals),
            mean_cer=mean(vals) if vals else None,
            median_cer=median(vals) if vals else None,
        ))
    return out


def rejection_coverage(records: list[tuple[float, float]], thresholds: tuple[float, ...]) -> list[dict]:
    """For each threshold: what fraction of lines would be auto-accepted, and the CER of that
    accepted subset. The practical question a confidence score has to answer."""
    out = []
    total = len(records)
    for t in thresholds:
        kept = [cer for conf, cer in records if conf >= t]
        out.append({
            "threshold": t,
            "coverage": len(kept) / total if total else 0.0,
            "accepted_count": len(kept),
            "mean_cer_of_accepted": (sum(kept) / len(kept)) if kept else None,
        })
    return out
