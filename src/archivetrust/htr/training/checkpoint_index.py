"""The durable, atomic checkpoint index for `loghi_swedish_finetuned_v1` (Work Package 13).

**Never determine the newest checkpoint by filename ordering** -- this index is the one place that
answers "which checkpoint is resumable / best / latest," written atomically (write-temp-then-rename)
so a crash mid-write can never leave a torn, half-written index behind. A checkpoint is only ever
marked `resumable=True` after every required state file has been saved *and* re-opened to verify it is
actually readable -- see `verify_checkpoint` -- matching the brief's "Never label a checkpoint
resumable before all required state has been saved and verified."
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class CheckpointEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_id: str
    run_id: str
    session_id: str
    source_checkpoint: str
    """The `--existing_model` this checkpoint was produced from -- `None`/empty string only for the
    very first checkpoint, which started from the real pinned generic checkpoint instead."""
    epoch: int
    """The cumulative, ArchiveTrust-tracked epoch number -- never Keras's own per-invocation epoch
    counter, which restarts at 0 every container invocation (see `training_session.py`'s module
    docstring for why)."""
    global_step: int
    cumulative_training_seconds: float
    session_training_seconds: float
    checkpoint_dir: str
    model_file_hash: str | None
    """sha256 of the `.keras` file -- `None` only if verification could not read it, in which case
    `verification_status` must not be `"verified"`."""
    model_state_present: bool
    optimizer_state_present: bool
    """Keras's `.save()` includes optimizer state by default and `LoghiCustomCallback` never
    overrides that -- `True` whenever the `.keras` file itself was written by that callback."""
    scheduler_state_present: bool
    """The LR schedule's own step counter lives inside the optimizer's saved state (`LoghiLearningRateSchedule`
    is a `tf.keras.optimizers.schedules.LearningRateSchedule` bound to the optimizer) -- tracked
    separately here because it is a distinct claim from "the optimizer variables exist," per the
    brief's explicit "do not confuse loading model weights with resuming training."""
    sampler_state_present: bool
    """Honestly `False` for every checkpoint produced by the pinned `loghi-htr` commit -- its
    `tf.data` shuffle pipeline is not checkpointed. Recorded as a fact, not silently omitted."""
    configuration_hash: str
    training_manifest_hash: str
    validation_manifest_hash: str
    validation_metrics: dict
    created_at: str
    verification_status: str
    """`"unverified" | "verified" | "verification_failed"`."""
    resumable: bool
    checkpoint_kind: str
    """`"latest" | "best_val" | "end_of_session"` -- WP13's three retained categories; one physical
    checkpoint directory can carry more than one kind (e.g. the last epoch of a session is often both
    `"latest"` and `"end_of_session"`), recorded as separate index entries pointing at the same
    `checkpoint_dir` rather than a single entry with an ambiguous kind."""


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-index-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_index(index_path: str | Path) -> list[CheckpointEntry]:
    path = Path(index_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CheckpointEntry.model_validate(e) for e in payload["entries"]]


def append_checkpoint_entry(index_path: str | Path, entry: CheckpointEntry) -> None:
    """Never overwrites an existing (immutable) entry -- WP13's "Do not overwrite immutable session
    checkpoints." Appends, then atomically rewrites the whole index file."""
    entries = load_index(index_path)
    if any(e.checkpoint_id == entry.checkpoint_id for e in entries):
        raise ValueError(f"checkpoint_id {entry.checkpoint_id!r} already exists in the index -- append-only")
    entries.append(entry)
    _atomic_write_json(Path(index_path), {"entries": [e.model_dump() for e in entries]})


def latest_resumable_checkpoint(index_path: str | Path) -> CheckpointEntry | None:
    """The real "which checkpoint do I resume from" query -- by recorded `epoch`/`created_at`, never
    by scanning a directory's filenames."""
    entries = [e for e in load_index(index_path) if e.resumable]
    if not entries:
        return None
    return max(entries, key=lambda e: (e.epoch, e.created_at))


def best_validation_checkpoint(index_path: str | Path, *, metric_key: str = "val_CER_metric") -> CheckpointEntry | None:
    entries = [e for e in load_index(index_path) if e.resumable and metric_key in e.validation_metrics]
    if not entries:
        return None
    return min(entries, key=lambda e: e.validation_metrics[metric_key])


def verify_checkpoint(checkpoint_dir: str | Path) -> tuple[bool, str | None, dict]:
    """Reopens the checkpoint directory's `.keras` file well enough to prove it is readable --
    computes its real hash and confirms the file is non-empty and parses as a valid zip (Keras's
    `.keras` format is a zip archive) without invoking TensorFlow, so this check is cheap and has no
    GPU/framework dependency. A full `tf.keras.models.load_model` round-trip is what the resume-proof
    (`training_session.py`) actually exercises; this is the lightweight, always-available check the
    checkpoint index itself can run right after every save.
    """
    import zipfile

    checkpoint_dir = Path(checkpoint_dir)
    model_files = list(checkpoint_dir.glob("*.keras"))
    if not model_files:
        return False, None, {}

    model_path = model_files[0]
    if model_path.stat().st_size == 0:
        return False, None, {}

    try:
        with zipfile.ZipFile(model_path) as zf:
            bad_entry = zf.testzip()
            if bad_entry is not None:
                return False, None, {}
    except zipfile.BadZipFile:
        return False, None, {}

    digest = hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)

    extra = {}
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if tokenizer_path.exists():
        extra["tokenizer_present"] = True
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        extra["config_present"] = True

    return True, digest.hexdigest(), extra
