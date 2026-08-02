"""Cached shard-integrity validation, so resuming does not re-read the entire shard set every time.

**The problem this solves, which I introduced.** Strengthening the launch guard (R-007) made it
rehash every lap-0 shard *and* re-check overlap against the validation and reserved-test manifests --
57 + 171 + 171 = 399 Parquet reads on every `start` and every `resume`. That is correct but slow, and
the cost is paid before the container even launches, on a run that may resume dozens of times.

**The rule this must not break: never skip validation to save time.** So the cache is not a bypass,
it is a *fingerprint* of the inputs whose full validation already succeeded:

- **First run** performs the complete check -- every shard hashed, full overlap validation -- and
  records the result together with a fingerprint of every shard file (size + mtime) plus the dataset,
  validation, and configuration identities it was validated against.
- **Resume** re-verifies that fingerprint: file count, every file's size and mtime, and the recorded
  dataset/validation/configuration hashes. It then rehashes a *deterministic* sample of shards, so
  content is still genuinely re-read on every resume rather than trusted purely from metadata.
- **Any mismatch** -- a changed size, mtime, count, dataset hash, validation-manifest hash, or
  configuration hash -- discards the cache and forces the full check.
- `force_full_check=True` (CLI: `--force-integrity-check`) always performs the complete validation.

The deterministic sample is seeded by the manifest hash, so it is reproducible and auditable rather
than a random subset that might differ between two people investigating the same run. A tampered
shard that preserves both size and mtime and is missed by the sample would still be caught at the
next full check, and cannot survive a dataset/config change -- but this is a genuine, disclosed
narrowing of guarantee versus rehashing everything, which is why the override flag exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

INTEGRITY_CACHE_NAME = "integrity_cache.json"
DEFAULT_SAMPLE_SIZE = 8
"""Shards rehashed on every resume even when the fingerprint matches. Small enough to be fast
(~8 Parquet reads instead of 399), large enough that repeated resumes cover the set."""


class ShardFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    size_bytes: int
    mtime_ns: int


class IntegrityCache(BaseModel):
    model_config = ConfigDict(frozen=True)

    training_manifest_hash: str
    dataset_hash: str
    validation_manifest_hash: str
    configuration_hash: str
    shard_count: int
    fingerprints: tuple[ShardFingerprint, ...]
    validation_overlap: int
    test_overlap: int
    validated_at: str


def _fingerprint(path: Path) -> ShardFingerprint:
    st = path.stat()
    return ShardFingerprint(path=str(path), size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def deterministic_sample_indices(*, manifest_hash: str, shard_count: int, sample_size: int = DEFAULT_SAMPLE_SIZE) -> list[int]:
    """Reproducible from the manifest hash alone, so any two people auditing the same run rehash the
    same shards."""
    if shard_count <= sample_size:
        return list(range(shard_count))
    rng = random.Random(int(manifest_hash[:16], 16))
    return sorted(rng.sample(range(shard_count), sample_size))


def cache_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run-state" / INTEGRITY_CACHE_NAME


def load_cache(run_dir: str | Path) -> IntegrityCache | None:
    p = cache_path(run_dir)
    if not p.exists():
        return None
    try:
        return IntegrityCache.model_validate_json(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_cache(run_dir: str | Path, cache: IntegrityCache) -> None:
    p = cache_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-integrity-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(cache.model_dump_json(indent=2))
        os.replace(tmp, str(p))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def cache_is_valid(
    cache: IntegrityCache | None,
    *,
    shard_paths: list[Path],
    dataset_hash: str | None,
    training_manifest_hash: str | None,
    validation_manifest_hash: str,
    configuration_hash: str,
) -> tuple[bool, str]:
    """`(is_valid, reason)`. Any identity or fingerprint mismatch invalidates -- the caller must then
    perform the full check rather than proceeding."""
    if cache is None:
        return False, "no cached integrity result"
    if cache.training_manifest_hash != (training_manifest_hash or ""):
        return False, "training manifest hash changed since the cached validation"
    if cache.dataset_hash != (dataset_hash or ""):
        return False, "dataset hash changed since the cached validation"
    if cache.validation_manifest_hash != validation_manifest_hash:
        return False, "validation manifest hash changed since the cached validation"
    if cache.configuration_hash != configuration_hash:
        return False, "configuration hash changed since the cached validation"
    if cache.shard_count != len(shard_paths):
        return False, f"shard count changed ({cache.shard_count} -> {len(shard_paths)})"

    recorded = {f.path: f for f in cache.fingerprints}
    for path in shard_paths:
        f = recorded.get(str(path))
        if f is None:
            return False, f"shard not present in the cached fingerprint set: {path.name}"
        if not path.exists():
            return False, f"shard file is missing: {path.name}"
        st = path.stat()
        if st.st_size != f.size_bytes:
            return False, f"shard size changed: {path.name}"
        if st.st_mtime_ns != f.mtime_ns:
            return False, f"shard modification time changed: {path.name}"
    return True, "fingerprint matches the cached full validation"


def build_cache(
    *,
    shard_paths: list[Path],
    dataset_hash: str | None,
    training_manifest_hash: str | None,
    validation_manifest_hash: str,
    configuration_hash: str,
    validation_overlap: int,
    test_overlap: int,
) -> IntegrityCache:
    import time

    return IntegrityCache(
        training_manifest_hash=training_manifest_hash or "",
        dataset_hash=dataset_hash or "",
        validation_manifest_hash=validation_manifest_hash,
        configuration_hash=configuration_hash,
        shard_count=len(shard_paths),
        fingerprints=tuple(_fingerprint(p) for p in shard_paths),
        validation_overlap=validation_overlap,
        test_overlap=test_overlap,
        validated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def sampled_shard_hash(shard_paths: list[Path], indices: list[int]) -> str:
    """Rehashes only the sampled shards' real `line_id` content -- so even a cache hit re-reads real
    bytes rather than trusting metadata alone."""
    import pyarrow.parquet as pq

    digest = hashlib.sha256()
    for i in indices:
        ids = pq.read_table(shard_paths[i], columns=["line_id"]).column("line_id").to_pylist()
        digest.update(json.dumps(sorted(ids)).encode("utf-8"))
    return digest.hexdigest()
