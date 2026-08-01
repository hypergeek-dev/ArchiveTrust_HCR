"""The stable training identity + configuration-hash gate for `loghi_swedish_finetuned_v1`
(docs/methods/loghi-swedish-finetuning.md, Work Package 8).

Mirrors `htr/screening/run_configuration.py::ResolvedConfiguration`'s discipline exactly: every field
that matters to "is this still the same run" is resolved from real, already-committed artifacts
(manifest hashes, the real pinned checkpoint, the real charlist) into one deterministic
`configuration_hash`. A **session** (one ~5-hour training invocation) reuses an unchanged `run_id` and
`configuration_hash`; a **new run** -- a different `Experiment` in the domain-design sense -- is
whatever this module's own docstring in the brief calls out: changed manifests, vocabulary,
preprocessing, architecture, parent checkpoint, LR policy, optimizer, or augmentation policy. Changing
any of those and reusing the old `run_id` is refused by `require_unchanged_configuration` below, the
same way `ResolvedConfiguration` refuses a resume under a changed corpus/sample/model revision.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.shared.ids import new_id

METHOD_ID = "loghi_swedish_finetuned_v1"
PARENT_METHOD_ID = "loghi"
TRAINING_DATASET_ID = "riksarkivet_swedish_lion_libre_training_data"
TRAINING_PHASE = "pilot_10k"
TRAINING_PHASE_FULL_CORPUS = "full_corpus"
"""`htr/training/full_run/identity.py`'s phase -- named here, alongside the pilot's own
`TRAINING_PHASE`, so both real phases this identity mechanism has ever needed to represent are
declared in one place rather than one being a bare string literal scattered at call sites."""


class TrainingIdentity(BaseModel):
    """The stable identity this training work belongs to -- unchanged across every resumed session."""

    model_config = ConfigDict(frozen=True)

    method_id: str = METHOD_ID
    parent_method_id: str = PARENT_METHOD_ID
    parent_checkpoint: str
    """`{model_checkpoint_id}@{model_checkpoint_hash}` -- the exact real pinned generic checkpoint
    this fine-tune started from (`providers/loghi/pinned_versions.py`)."""
    training_dataset: str = TRAINING_DATASET_ID
    training_phase: str = TRAINING_PHASE
    run_id: str
    created_at: str
    """Set once, at identity creation -- never touched again across resumed sessions. The
    dashboard's "start time" for this run (`run_status.py`); nothing else in this module's durable
    files previously recorded when a run first began."""


class TrainingConfiguration(BaseModel):
    """Every field whose change means "this is materially a different training run," per the brief's
    own list. Hashed, never compared field-by-field by a caller -- `configuration_hash` is the single
    source of truth for "did anything material change."
    """

    model_config = ConfigDict(frozen=True)

    train_manifest_hash: str
    val_manifest_hash: str
    charlist_hash: str
    """sha256 of the exact charlist file used -- covers vocabulary extension too: a repinned/extended
    charlist changes this hash."""
    preprocessing_version: str
    """Currently `"byte_identical_from_source_parquet"` -- see `loghi_training_data.py`'s module
    docstring: images are written unmodified from the source Parquet, no transform to version."""
    model_architecture: str
    """The VGSL-style architecture id the generic checkpoint itself was trained with (`"new10"`, read
    from `generic-2023-02-15/file.txt`'s own shipped config) -- this fine-tune does not change it."""
    parent_checkpoint_hash: str
    learning_rate_policy: str
    optimizer: str
    augmentation_policy: str

    def compute_hash(self) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TrainingIdentityMismatch(RuntimeError):
    """Raised when a resumed session's configuration hash disagrees with the persisted run's -- the
    brief's "a session is not a new model experiment" rule, enforced structurally rather than by
    convention."""


def _identity_path(run_state_dir: str | Path) -> Path:
    return Path(run_state_dir) / "training_identity.json"


def create_or_load_identity(
    *,
    run_state_dir: str | Path,
    parent_checkpoint: str,
    configuration: TrainingConfiguration,
    training_phase: str = TRAINING_PHASE,
) -> tuple[TrainingIdentity, str]:
    """First call for a fresh run: mints a new `run_id`, persists identity + configuration hash,
    returns `(identity, configuration_hash)`. Every later call (a resumed session) must pass the exact
    same `configuration`; a mismatch raises `TrainingIdentityMismatch` rather than silently continuing
    under a changed configuration, or silently overwriting the persisted one.

    `training_phase` defaults to the pilot's own `TRAINING_PHASE` -- every existing pilot call site
    (`scripts/train_loghi_swedish.py`) is unaffected. `htr/training/full_run/identity.py` passes
    `TRAINING_PHASE_FULL_CORPUS` explicitly.
    """
    run_state_dir = Path(run_state_dir)
    run_state_dir.mkdir(parents=True, exist_ok=True)
    path = _identity_path(run_state_dir)
    configuration_hash = configuration.compute_hash()

    if path.exists():
        persisted = json.loads(path.read_text(encoding="utf-8"))
        identity = TrainingIdentity.model_validate(persisted["identity"])
        if persisted["configuration_hash"] != configuration_hash:
            raise TrainingIdentityMismatch(
                f"Persisted configuration_hash {persisted['configuration_hash']!r} does not match "
                f"the configuration passed to this session ({configuration_hash!r}). A session must "
                "reuse the exact same training manifests/vocabulary/preprocessing/architecture/parent "
                "checkpoint/LR policy/optimizer/augmentation policy as the run it resumes -- if any of "
                "those genuinely changed, this is a new run (new run_id), not a resumed session."
            )
        return identity, configuration_hash

    identity = TrainingIdentity(
        parent_checkpoint=parent_checkpoint,
        training_phase=training_phase,
        run_id=new_id("loghi_training_run"),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    path.write_text(
        json.dumps(
            {"identity": identity.model_dump(), "configuration_hash": configuration_hash, "configuration": configuration.model_dump()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return identity, configuration_hash
