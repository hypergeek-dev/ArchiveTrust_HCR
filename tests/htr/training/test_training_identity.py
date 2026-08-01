from __future__ import annotations

import pytest

from archivetrust.htr.training.training_identity import (
    TrainingConfiguration,
    TrainingIdentityMismatch,
    create_or_load_identity,
)


def _config(**overrides) -> TrainingConfiguration:
    base = dict(
        train_manifest_hash="train_hash",
        val_manifest_hash="val_hash",
        charlist_hash="charlist_hash",
        preprocessing_version="byte_identical_from_source_parquet",
        model_architecture="new10",
        parent_checkpoint_hash="parent_hash",
        learning_rate_policy="constant_0.0001",
        optimizer="adam",
        augmentation_policy="none",
    )
    base.update(overrides)
    return TrainingConfiguration(**base)


def test_first_call_mints_a_new_run_id(tmp_path):
    identity, config_hash = create_or_load_identity(
        run_state_dir=tmp_path, parent_checkpoint="generic-2023-02-15@abc", configuration=_config()
    )
    assert identity.run_id.startswith("loghi_training_run_")
    assert identity.method_id == "loghi_swedish_finetuned_v1"
    assert identity.parent_method_id == "loghi"
    assert config_hash


def test_second_call_with_same_configuration_reuses_the_run_id(tmp_path):
    identity1, hash1 = create_or_load_identity(
        run_state_dir=tmp_path, parent_checkpoint="generic-2023-02-15@abc", configuration=_config()
    )
    identity2, hash2 = create_or_load_identity(
        run_state_dir=tmp_path, parent_checkpoint="generic-2023-02-15@abc", configuration=_config()
    )
    assert identity1.run_id == identity2.run_id
    assert hash1 == hash2


def test_a_session_is_not_a_new_experiment_changed_configuration_is_refused(tmp_path):
    create_or_load_identity(run_state_dir=tmp_path, parent_checkpoint="generic-2023-02-15@abc", configuration=_config())
    with pytest.raises(TrainingIdentityMismatch):
        create_or_load_identity(
            run_state_dir=tmp_path,
            parent_checkpoint="generic-2023-02-15@abc",
            configuration=_config(learning_rate_policy="different_policy"),
        )


def test_configuration_hash_changes_when_manifests_change():
    a = _config().compute_hash()
    b = _config(train_manifest_hash="different_hash").compute_hash()
    assert a != b


def test_configuration_hash_is_deterministic():
    assert _config().compute_hash() == _config().compute_hash()


@pytest.mark.parametrize(
    "field",
    [
        "val_manifest_hash", "charlist_hash", "preprocessing_version", "model_architecture",
        "parent_checkpoint_hash", "learning_rate_policy", "optimizer", "augmentation_policy",
    ],
)
def test_every_material_field_changes_the_hash(field):
    """The brief's own list of what makes a session a new experiment -- each one must move the hash."""
    a = _config().compute_hash()
    b = _config(**{field: "materially_different_value"}).compute_hash()
    assert a != b
