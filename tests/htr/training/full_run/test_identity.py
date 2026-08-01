from __future__ import annotations

import pytest

from archivetrust.htr.training.full_run.identity import (
    PilotCheckpointRejected,
    RunDirectoryAlreadyExists,
    assert_parent_checkpoint_is_not_a_pilot_path,
    create_full_run_identity,
    generate_run_name,
)
from archivetrust.htr.training.training_identity import TRAINING_PHASE_FULL_CORPUS, TrainingConfiguration


def _config(**overrides) -> TrainingConfiguration:
    base = dict(
        train_manifest_hash="t", val_manifest_hash="v", charlist_hash="c",
        preprocessing_version="byte_identical_from_source_parquet", model_architecture="new10",
        parent_checkpoint_hash="p", learning_rate_policy="constant_0.0001", optimizer="adam",
        augmentation_policy="none",
    )
    base.update(overrides)
    return TrainingConfiguration(**base)


def test_generate_run_name_uses_the_explicit_name_verbatim():
    assert generate_run_name(base_name="my-full-run") == "my-full-run"


def test_generate_run_name_generates_a_timestamped_identifier_when_none_given():
    name = generate_run_name(base_name=None)
    assert name.startswith("full-corpus-")
    assert len(name) > len("full-corpus-")


def test_generate_run_name_is_unique_across_calls():
    import time

    name1 = generate_run_name(base_name=None)
    time.sleep(1.1)  # the format is second-resolution
    name2 = generate_run_name(base_name=None)
    assert name1 != name2


def test_rejects_a_parent_checkpoint_directly_under_a_pilot_run_dir(tmp_path):
    pilot_dir = tmp_path / "training" / "loghi-swedish-v1"
    bad_checkpoint = pilot_dir / "run-state" / "epoch_output" / "epoch_22" / "model_new10" / "best_val"
    bad_checkpoint.mkdir(parents=True)
    with pytest.raises(PilotCheckpointRejected):
        assert_parent_checkpoint_is_not_a_pilot_path(bad_checkpoint, known_pilot_run_dirs=(pilot_dir,))


def test_rejects_the_pilot_run_dir_itself(tmp_path):
    pilot_dir = tmp_path / "training" / "loghi-swedish-v1"
    pilot_dir.mkdir(parents=True)
    with pytest.raises(PilotCheckpointRejected):
        assert_parent_checkpoint_is_not_a_pilot_path(pilot_dir, known_pilot_run_dirs=(pilot_dir,))


def test_accepts_a_real_pinned_checkpoint_path_outside_any_pilot_dir(tmp_path):
    pilot_dir = tmp_path / "training" / "loghi-swedish-v1"
    real_pin = tmp_path / ".loghi-upstream" / "pretrained-models" / "loghi-htr" / "generic-2023-02-15"
    real_pin.mkdir(parents=True)
    assert_parent_checkpoint_is_not_a_pilot_path(real_pin, known_pilot_run_dirs=(pilot_dir,))  # does not raise


def test_accepts_a_similarly_named_but_genuinely_different_directory(tmp_path):
    """A real edge case: a directory whose name merely *contains* the pilot's name as a substring
    (not a path-hierarchy match) must not be falsely rejected."""
    pilot_dir = tmp_path / "training" / "loghi-swedish-v1"
    pilot_dir.mkdir(parents=True)
    lookalike = tmp_path / "training" / "loghi-swedish-v1-full-corpus"
    lookalike.mkdir(parents=True)
    assert_parent_checkpoint_is_not_a_pilot_path(lookalike, known_pilot_run_dirs=(pilot_dir,))  # does not raise


def test_create_full_run_identity_uses_the_full_corpus_phase(tmp_path):
    real_pin = tmp_path / ".loghi-upstream" / "generic-2023-02-15"
    real_pin.mkdir(parents=True)
    identity, config_hash = create_full_run_identity(
        run_state_dir=tmp_path / "run-state",
        parent_checkpoint="generic-2023-02-15@abc",
        parent_checkpoint_dir=real_pin,
        configuration=_config(),
    )
    assert identity.training_phase == TRAINING_PHASE_FULL_CORPUS
    assert identity.run_id.startswith("loghi_training_run_")
    assert config_hash


def test_create_full_run_identity_rejects_a_pilot_checkpoint(tmp_path):
    pilot_dir = tmp_path / "training" / "loghi-swedish-v1"
    bad_checkpoint = pilot_dir / "run-state" / "epoch_output" / "epoch_22"
    bad_checkpoint.mkdir(parents=True)
    with pytest.raises(PilotCheckpointRejected):
        create_full_run_identity(
            run_state_dir=tmp_path / "run-state",
            parent_checkpoint="pilot-epoch-22@abc",
            parent_checkpoint_dir=bad_checkpoint,
            configuration=_config(),
            known_pilot_run_dirs=(pilot_dir,),
        )


def test_create_full_run_identity_refuses_to_overwrite_an_existing_run_directory(tmp_path):
    real_pin = tmp_path / ".loghi-upstream" / "generic-2023-02-15"
    real_pin.mkdir(parents=True)
    run_state_dir = tmp_path / "run-state"
    create_full_run_identity(
        run_state_dir=run_state_dir, parent_checkpoint="generic-2023-02-15@abc",
        parent_checkpoint_dir=real_pin, configuration=_config(),
    )
    with pytest.raises(RunDirectoryAlreadyExists):
        create_full_run_identity(
            run_state_dir=run_state_dir, parent_checkpoint="generic-2023-02-15@abc",
            parent_checkpoint_dir=real_pin, configuration=_config(),
        )


def test_full_run_and_pilot_identity_mechanisms_share_the_same_hash_discipline(tmp_path):
    """Same `TrainingConfiguration.compute_hash()` -- no parallel, divergent hashing scheme for the
    full-run phase."""
    real_pin = tmp_path / ".loghi-upstream" / "generic-2023-02-15"
    real_pin.mkdir(parents=True)
    _, config_hash = create_full_run_identity(
        run_state_dir=tmp_path / "run-state", parent_checkpoint="generic-2023-02-15@abc",
        parent_checkpoint_dir=real_pin, configuration=_config(),
    )
    assert config_hash == _config().compute_hash()
