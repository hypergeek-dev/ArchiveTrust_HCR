from __future__ import annotations

import pytest

from archivetrust.htr.training.lr_schedule import (
    LrScheduleParameters,
    learning_rate_at,
    resolve_schedule_parameters,
)


def test_resolves_decay_steps_of_negative_one_to_train_batches():
    """The real, recorded `config.json` convention: `decay_steps: -1` means "use this epoch's own
    batch count" -- confirmed against `create_learning_rate_schedule`'s real source."""
    params = resolve_schedule_parameters(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, warmup_ratio=0.0,
        train_batches_this_epoch=625,
    )
    assert params.decay_steps == 625


def test_leaves_an_explicit_positive_decay_steps_untouched():
    params = resolve_schedule_parameters(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=100, warmup_ratio=0.0,
        train_batches_this_epoch=625,
    )
    assert params.decay_steps == 100


def test_total_steps_matches_the_pinned_formula():
    """`total_steps = epochs_this_invocation * train_batches_this_epoch + 1` -- from
    `create_learning_rate_schedule`'s real construction."""
    params = resolve_schedule_parameters(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, warmup_ratio=0.0,
        train_batches_this_epoch=625, epochs_this_invocation=1,
    )
    assert params.total_steps == 626


def test_no_warmup_no_decay_at_step_zero_returns_the_initial_rate():
    """This project's real recorded config: `warmup_ratio: 0.0`, `decay_rate: 0.99`,
    `linear_decay: false` -- at step 0, `decay_rate ** 0 == 1`, so lr == initial_lr exactly."""
    params = resolve_schedule_parameters(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, warmup_ratio=0.0,
        train_batches_this_epoch=625,
    )
    assert learning_rate_at(0, params) == pytest.approx(0.0001)


def test_exponential_decay_matches_a_hand_computed_value():
    params = resolve_schedule_parameters(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, warmup_ratio=0.0,
        train_batches_this_epoch=100,
    )
    # step=100, decay_steps=100, warmup_steps=0 -> exponent = (100-0)/100 = 1 -> lr = initial * 0.99^1
    assert learning_rate_at(100, params) == pytest.approx(0.0001 * 0.99)


def test_exponential_decay_is_monotonically_decreasing_past_warmup():
    params = resolve_schedule_parameters(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, warmup_ratio=0.0,
        train_batches_this_epoch=100,
    )
    lr_early = learning_rate_at(50, params)
    lr_later = learning_rate_at(500, params)
    assert lr_later < lr_early


def test_warmup_phase_ramps_linearly_from_zero():
    params = resolve_schedule_parameters(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, warmup_ratio=0.5,
        train_batches_this_epoch=100,
    )
    # total_steps=101, warmup_steps=0.5*101=50.5 -> step 25 is mid-warmup
    lr_at_25 = learning_rate_at(25, params)
    lr_at_10 = learning_rate_at(10, params)
    assert lr_at_10 < lr_at_25 < 0.0001


def test_linear_decay_reaches_zero_at_total_steps():
    params = LrScheduleParameters(
        initial_learning_rate=0.0001, decay_rate=0.99, decay_steps=100, warmup_ratio=0.0,
        total_steps=101, decay_per_epoch=False, linear_decay=True,
    )
    assert learning_rate_at(101, params) == pytest.approx(0.0, abs=1e-12)


def test_linear_decay_never_goes_negative_past_total_steps():
    params = LrScheduleParameters(
        initial_learning_rate=0.0001, decay_rate=0.99, decay_steps=100, warmup_ratio=0.0,
        total_steps=101, decay_per_epoch=False, linear_decay=True,
    )
    assert learning_rate_at(500, params) == 0.0


def test_decay_per_epoch_uses_floor_of_step_over_decay_steps():
    params = LrScheduleParameters(
        initial_learning_rate=0.0001, decay_rate=0.99, decay_steps=100, warmup_ratio=0.0,
        total_steps=1001, decay_per_epoch=True, linear_decay=False,
    )
    # steps 100-199 all floor to exponent=1 -- same lr throughout that band
    assert learning_rate_at(100, params) == learning_rate_at(199, params)
    assert learning_rate_at(199, params) != learning_rate_at(200, params)
