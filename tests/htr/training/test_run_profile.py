from __future__ import annotations

from archivetrust.htr.training.run_profile import (
    FULL_CORPUS_DEFAULT_BUDGET_HOURS,
    FULL_CORPUS_DEFAULT_EPOCH_CAP,
    cap_shorter_than_one_epoch_warning,
    full_corpus_profile,
    pilot_profile,
)


def test_pilot_profile_has_a_real_measured_not_estimated_timing():
    profile = pilot_profile()
    assert profile.is_estimated is False
    assert profile.default_wall_clock_hours == 3.0
    assert profile.default_patience == 5


def test_full_corpus_profile_refuses_a_silent_wall_clock_default():
    profile = full_corpus_profile(data_prepared=False)
    assert profile.default_wall_clock_hours is None
    assert profile.is_estimated is True
    assert profile.default_epoch_cap == FULL_CORPUS_DEFAULT_EPOCH_CAP
    assert profile.suggested_budget_hours == FULL_CORPUS_DEFAULT_BUDGET_HOURS


def test_full_corpus_profile_reports_whether_its_data_is_actually_prepared():
    assert full_corpus_profile(data_prepared=False).data_prepared is False
    assert full_corpus_profile(data_prepared=True).data_prepared is True


def test_no_warning_when_cap_comfortably_exceeds_one_epoch():
    profile = pilot_profile()  # ~0.152h/epoch
    assert cap_shorter_than_one_epoch_warning(profile=profile, wall_clock_hours=3.0) is None


def test_warns_when_cap_is_shorter_than_one_pilot_epoch():
    profile = pilot_profile()  # ~0.152h/epoch
    warning = cap_shorter_than_one_epoch_warning(profile=profile, wall_clock_hours=0.05)
    assert warning is not None
    assert "pilot" in warning
    assert "measured" in warning  # pilot's timing is real, not extrapolated


def test_warns_when_cap_is_shorter_than_one_full_corpus_epoch_and_labels_it_estimated():
    profile = full_corpus_profile(data_prepared=True)  # ~7.17h/epoch
    warning = cap_shorter_than_one_epoch_warning(profile=profile, wall_clock_hours=3.0)
    assert warning is not None
    assert "full_corpus" in warning
    assert "estimated" in warning  # must never claim this figure is measured


def test_warning_boundary_is_inclusive_exactly_one_epoch_is_not_a_warning():
    profile = pilot_profile()
    exact_hours = profile.estimated_seconds_per_epoch / 3600.0
    assert cap_shorter_than_one_epoch_warning(profile=profile, wall_clock_hours=exact_hours) is None
