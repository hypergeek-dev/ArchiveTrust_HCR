"""Asserts the outcome of the real-model stop/resume demonstration.

`scripts/reliability_fixture_demo.py` is the part of this work that cannot be proven with fakes: it
runs real Florence-2 segmentation, the real SATRN subprocess and real Florence-2 recognition across
**two separate OS processes**, stopping in the first and resuming in the second. It writes what it
observed to two small JSON records, and this module is the assertion over them.

Written as read-and-assert rather than as a test that launches the demo, deliberately. Re-running it
inside `pytest` would put a multi-minute GPU job on the critical path of a suite that must stay fast
and must pass on a machine with no `.venv-satrn` -- and it would prove nothing the recorded evidence
does not already prove, since the recorded evidence *is* the run's own output. When the records are
absent the tests skip with a message naming the command that produces them; they are never silently
vacuous.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archivetrust.htr.screening.run_configuration import (
    FLORENCE2_METHOD_ID,
    SATRN_METHOD_ID,
    screening_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_ROOT = screening_directory(REPO_ROOT) / "fixture-demo"
COMMAND = "PYTHONPATH=src .venv/Scripts/python.exe scripts/reliability_fixture_demo.py"


def _load(name: str) -> dict:
    path = DEMO_ROOT / name
    if not path.is_file():
        pytest.skip(f"{path} absent -- produce it with: {COMMAND}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def phase1() -> dict:
    return _load("demo-phase1.json")


@pytest.fixture
def phase2() -> dict:
    return _load("demo-phase2.json")


def test_phase1_stopped_safely_partway_through(phase1):
    assert phase1["stopped"] is True
    completed = phase1["completed_by_method"]
    total = sum(completed.values())
    assert 0 < total < 8, "phase 1 must stop partway, not before starting or after finishing"
    # Every real adapter call it made produced a recorded completion -- no call is unaccounted for.
    assert phase1["adapter_calls"][SATRN_METHOD_ID] == completed[SATRN_METHOD_ID]
    assert phase1["adapter_calls"][FLORENCE2_METHOD_ID] == completed[FLORENCE2_METHOD_ID]


def test_phase2_resumed_in_a_new_process_without_repeating_any_work(phase2):
    assert phase2["run_id"] == _load("demo-phase1.json")["run_id"], (
        "phase 2 must have discovered the same run, with no run id typed"
    )
    assert phase2["completed_before"] == _load("demo-phase1.json")["completed_by_method"]
    assert phase2["tasks_skipped"] > 0, "phase 2 must have skipped phase 1's completed work"
    # The whole claim, in one line: the number of real model calls the resuming process made is
    # exactly the number of tasks that were still pending -- not one call more.
    assert phase2["adapter_calls"] == phase2["newly_completed"]
    assert phase2["no_work_repeated"] is True
    assert phase2["completed"] is True


def test_the_two_methods_finished_independently_and_completely(phase2):
    after = phase2["completed_after"]
    assert after[SATRN_METHOD_ID] == after[FLORENCE2_METHOD_ID]
    assert after[SATRN_METHOD_ID] > 0
