"""Where benchmark data lives, and which directories each stage may write.

    benchmark-data/                        (gitignored; override with ARCHIVETRUST_BENCHMARK_ROOT)
      incoming/<source_id>/                exactly as delivered -- the harness never writes here
      work/<source_id>/
        inspection/                        inspection.json, findings.jsonl, inspection.md
        candidate/                         manifest.jsonl, lines/, review_queue.jsonl, build.json
        decisions.jsonl                    human review decisions (append-only, hand-edited)
      benchmark/<benchmark_id>/            frozen: manifest.jsonl, lines/, FROZEN.json -- never overwritten
      reports/<run_id>/                    predictions/<model>.jsonl, scores, report.md, run_record.json
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT_ENV_VAR = "ARCHIVETRUST_BENCHMARK_ROOT"


class LayoutError(RuntimeError):
    pass


def default_root() -> Path:
    override = os.environ.get(ROOT_ENV_VAR)
    return Path(override) if override else REPO_ROOT / "benchmark-data"


@dataclass(frozen=True)
class BenchmarkLayout:
    root: Path

    @classmethod
    def default(cls) -> BenchmarkLayout:
        return cls(default_root())

    @property
    def incoming(self) -> Path:
        return self.root / "incoming"

    def incoming_source(self, source_id: str) -> Path:
        return self.incoming / source_id

    def work(self, source_id: str) -> Path:
        return self.root / "work" / source_id

    def inspection_dir(self, source_id: str) -> Path:
        return self.work(source_id) / "inspection"

    def candidate_dir(self, source_id: str) -> Path:
        return self.work(source_id) / "candidate"

    def decisions_file(self, source_id: str) -> Path:
        return self.work(source_id) / "decisions.jsonl"

    def frozen_dir(self, benchmark_id: str) -> Path:
        return self.root / "benchmark" / benchmark_id

    def report_dir(self, run_id: str) -> Path:
        return self.root / "reports" / run_id

    def dryrun_dir(self, dryrun_id: str) -> Path:
        """No-ground-truth mechanical dry runs (never a benchmark, never scored)."""
        return self.root / "dryrun" / dryrun_id


def assert_outside(target: Path, protected: Path) -> None:
    """Refuses any write whose target resolves inside `protected` (used to keep incoming/ read-only
    and frozen benchmarks immutable)."""
    target_r, protected_r = target.resolve(), protected.resolve()
    if target_r == protected_r or protected_r in target_r.parents:
        raise LayoutError(f"refusing to write {target} inside protected {protected}")
