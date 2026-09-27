"""Run provenance: code state, runtime, and time -- stamped into every frozen benchmark and report.

Dirty state is computed over the paths that decide benchmark behaviour (`src/`, `scripts/`,
`pyproject.toml`), not the whole tree: large gitignored training output has made a whole-tree
`git status` slow enough to time out before (see .gitignore notes), and an unrelated doc edit should
not taint a run. An "official" run refuses a dirty tree; any other run is labelled unofficial.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from archivetrust.htr.benchmark.layout import REPO_ROOT

CODE_PATHS = ("src", "scripts", "pyproject.toml")
PACKAGES = ("pillow", "pydantic", "pyarrow", "numpy", "torch", "transformers", "tensorflow")


class DirtyTreeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str, repo: Path) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def git_state(repo: Path = REPO_ROOT) -> dict:
    commit = _git("rev-parse", "HEAD", repo=repo)
    status = _git("status", "--porcelain", "--", *CODE_PATHS, repo=repo)
    dirty_paths = [line[3:] for line in (status or "").splitlines() if line.strip()]
    return {
        "commit": commit.strip() if commit else None,
        "dirty": bool(dirty_paths) if status is not None else None,
        "dirty_paths": dirty_paths[:50],
        "dirty_path_count": len(dirty_paths),
        "checked_paths": list(CODE_PATHS),
    }


def runtime_info() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions}


def provenance(*, official: bool, repo: Path = REPO_ROOT) -> dict:
    git = git_state(repo)
    if official and git["dirty"] is not False:
        raise DirtyTreeError(
            "official run refused: benchmark code has uncommitted changes (or git is unavailable): "
            f"{git['dirty_paths'][:10]} -- commit first, or run without --official (labelled UNOFFICIAL)")
    return {"official": official, "git": git, "runtime": runtime_info(), "timestamp_utc": utc_now()}


def label(record: dict) -> str:
    if record.get("official"):
        return "OFFICIAL"
    reasons = ["not run with --official"]
    if record.get("git", {}).get("dirty"):
        reasons.append(f"{record['git']['dirty_path_count']} uncommitted code paths")
    return "UNOFFICIAL (" + "; ".join(reasons) + ")"
