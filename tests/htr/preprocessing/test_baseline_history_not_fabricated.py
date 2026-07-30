"""No normalization event was fabricated for the pre-existing committed baseline.

The RGB-normalization stage did not exist when `docs/experiments/baseline-comparison/` was produced.
Its Transkribus `MethodRun` consumed a hand-authored PAGE XML fixture and -- as this file asserts from
the committed artifacts themselves -- had **no associated raw page image at all**: `input_crop_id` is
`None` and `tests/fixtures/transkribus/` contains only `.xml`/`.txt` files. There was therefore nothing
to normalize, no normalization was performed, and no normalization event may be added after the fact.

These tests are what makes that a checkable property of the repository rather than a promise in a
commit message. If someone later back-fills the baseline's telemetry to claim normalization occurred,
this file goes red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "docs" / "experiments" / "baseline-comparison"
RESEARCH_EVENTS = BASELINE_DIR / "htr_research_events.jsonl"
TRANSKRIBUS_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "transkribus"

NORMALIZATION_KINDS = frozenset(
    {
        "ImageNormalizationStarted",
        "ImageNormalizationCompleted",
        "ImageNormalizationFailed",
        "DerivedImageArtifactCreated",
    }
)

BASELINE_TELEMETRY_FILES = (
    "htr_research_events.jsonl",
    "htr_research_events.jsonl.chain.jsonl",
    "htr_knowledge_events.jsonl",
    "htr_knowledge_events.jsonl.chain.jsonl",
    "htr_knowledge_feedback_events.jsonl",
    "htr_knowledge_feedback_events.jsonl.chain.jsonl",
    "htr_coarse_entities.json",
)
"""Every committed telemetry artifact of the pre-existing baseline. None of these was modified by the
normalization work, and the tests below assert that none contains a normalization event."""


def _events():
    return [
        json.loads(line)
        for line in RESEARCH_EVENTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_committed_baseline_exists_so_these_assertions_are_meaningful():
    """Guards against this whole file silently passing because the baseline was moved or deleted."""
    assert RESEARCH_EVENTS.is_file(), f"{RESEARCH_EVENTS} is missing"
    assert len(_events()) > 50, "the committed baseline event log looks unexpectedly small"


@pytest.mark.parametrize("filename", BASELINE_TELEMETRY_FILES)
def test_no_normalization_event_appears_in_any_baseline_telemetry_file(filename):
    """The central assertion: not one of the four normalization event kinds appears anywhere in the
    pre-existing baseline's committed telemetry."""
    path = BASELINE_DIR / filename
    if not path.is_file():
        pytest.skip(f"{filename} is not part of the committed baseline")

    content = path.read_text(encoding="utf-8")

    for kind in sorted(NORMALIZATION_KINDS):
        assert kind not in content, (
            f"{filename} contains a {kind} event. The RGB-normalization stage did not exist when "
            "this baseline ran and its Transkribus run had no page image -- adding a normalization "
            "event to it retroactively fabricates history. See "
            "docs/methods/transkribus-swedish-lion-1.md §7."
        )


def test_the_baseline_transkribus_run_had_no_input_image_at_all():
    """Records the actual, verified reason no normalization event exists: there was no image.

    This is a stronger and more honest statement than "the fixture happened to already be RGB" -- the
    Transkribus baseline was a PAGE XML *text* import with no associated raster whatsoever.
    """
    transkribus_runs = [
        event["method_run"]
        for event in _events()
        if event["kind"] == "MethodRunStarted"
        and event.get("method_run", {}).get("method_id") == "transkribus_swedish_lion_1"
    ]

    assert len(transkribus_runs) == 1, "expected exactly one Transkribus MethodRun in the baseline"
    assert transkribus_runs[0]["input_crop_id"] is None, (
        "the baseline's Transkribus run is recorded as consuming no input crop"
    )


def test_no_page_image_exists_in_the_transkribus_fixture_directory():
    """Corroborates the above from the filesystem: there is no raster for that run to have used."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
    images = [
        path.name
        for path in TRANSKRIBUS_FIXTURES.iterdir()
        if path.suffix.lower() in image_suffixes
    ]

    assert images == [], (
        f"unexpected image file(s) {images} in tests/fixtures/transkribus/. The documented claim "
        "that the Transkribus baseline had no page image would no longer hold."
    )


def test_the_baseline_records_no_page_image_or_normalized_artifact_entities():
    """Replay-level check: reconstructing the committed baseline yields zero preprocessing entities.

    Complements the textual scan above -- this one goes through the real `HtrJournal`, so it would
    catch a normalization entity introduced under some other event kind.
    """
    from archivetrust.application.htr_journal import HtrJournal
    from archivetrust.domain.telemetry.events import parse_event

    store = HtrJournal().replay(parse_event(event) for event in _events())

    assert store.page_image_artifacts() == ()
    assert store.normalized_page_artifacts() == ()
    assert store.normalization_failures() == ()


def test_the_baseline_experiment_version_records_no_normalization_requirement():
    """The committed baseline's own pipeline configuration must not claim normalization."""
    version_events = [
        event for event in _events() if event["kind"] == "ExperimentVersionCreated"
    ]
    assert version_events, "expected an ExperimentVersionCreated event in the baseline"

    for event in version_events:
        ref = event["experiment_version"]["pipeline_configuration_ref"]
        payload = json.loads(ref)
        assert payload.get("image_color_normalization") in (None, {}), (
            "the committed baseline's pipeline configuration now claims an image-color-normalization "
            "requirement it never ran under"
        )


def test_the_demonstration_is_a_separate_directory_from_the_baseline():
    """The new demonstration must not live inside -- or write into -- the committed baseline."""
    demo = REPO_ROOT / "docs" / "experiments" / "rgb-normalization-demo"

    assert demo.is_dir(), "the demonstration run's output directory is missing"
    assert demo != BASELINE_DIR
    assert BASELINE_DIR not in demo.parents
