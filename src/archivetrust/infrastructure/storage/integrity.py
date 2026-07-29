"""Tamper-evident integrity manifests for append-only files and exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict


HASH_CHAIN_SCHEMA = "archivetrust.hash_chain.v1"
FILE_DIGEST_SCHEMA = "archivetrust.file_digest.v1"
HASH_ALGORITHM = "sha256"
ZERO_CHAIN_HASH = "0" * 64


class HashChainCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    line_number: int
    line_hash: str
    previous_chain_hash: str
    chain_hash: str


class HashChainManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_schema: str
    algorithm: str
    source_name: str
    created_at: str
    line_count: int
    root_hash: str
    checkpoints: tuple[HashChainCheckpoint, ...]


class FileDigestManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_schema: str
    algorithm: str
    source_name: str
    created_at: str
    size_bytes: int
    digest: str


class IntegrityVerification(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    reason: str | None = None
    expected: str | int | None = None
    actual: str | int | None = None


def default_hash_chain_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(f"{source.name}.chain.json")


def default_file_digest_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(f"{source.name}.integrity.json")


def build_hash_chain_manifest(path: str | Path) -> HashChainManifest:
    source = Path(path)
    previous = ZERO_CHAIN_HASH
    checkpoints: list[HashChainCheckpoint] = []

    for line_number, line in enumerate(_jsonl_record_bytes(source), start=1):
        line_hash = _sha256_hex(line)
        chain_hash = _sha256_hex(bytes.fromhex(previous) + bytes.fromhex(line_hash))
        checkpoints.append(
            HashChainCheckpoint(
                line_number=line_number,
                line_hash=line_hash,
                previous_chain_hash=previous,
                chain_hash=chain_hash,
            )
        )
        previous = chain_hash

    return HashChainManifest(
        manifest_schema=HASH_CHAIN_SCHEMA,
        algorithm=HASH_ALGORITHM,
        source_name=source.name,
        created_at=_now_utc(),
        line_count=len(checkpoints),
        root_hash=previous,
        checkpoints=tuple(checkpoints),
    )


def write_hash_chain_manifest(path: str | Path, manifest_path: str | Path | None = None) -> Path:
    source = Path(path)
    output = Path(manifest_path) if manifest_path is not None else default_hash_chain_manifest_path(source)
    output.write_text(
        build_hash_chain_manifest(source).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output


def load_hash_chain_manifest(path: str | Path) -> HashChainManifest:
    return HashChainManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def verify_hash_chain_manifest(path: str | Path, manifest_path: str | Path | None = None) -> IntegrityVerification:
    source = Path(path)
    sidecar = Path(manifest_path) if manifest_path is not None else default_hash_chain_manifest_path(source)
    if not sidecar.exists():
        return IntegrityVerification(ok=False, reason="manifest_missing")

    expected = load_hash_chain_manifest(sidecar)
    if expected.manifest_schema != HASH_CHAIN_SCHEMA or expected.algorithm != HASH_ALGORITHM:
        return IntegrityVerification(
            ok=False,
            reason="unsupported_manifest",
            expected=HASH_CHAIN_SCHEMA,
            actual=expected.manifest_schema,
        )

    actual = build_hash_chain_manifest(source)
    if expected.line_count != actual.line_count:
        return IntegrityVerification(
            ok=False,
            reason="line_count_mismatch",
            expected=expected.line_count,
            actual=actual.line_count,
        )
    if expected.root_hash != actual.root_hash:
        return IntegrityVerification(
            ok=False,
            reason="root_hash_mismatch",
            expected=expected.root_hash,
            actual=actual.root_hash,
        )

    expected_checkpoints = {checkpoint.line_number: checkpoint for checkpoint in expected.checkpoints}
    for checkpoint in actual.checkpoints:
        expected_checkpoint = expected_checkpoints.get(checkpoint.line_number)
        if expected_checkpoint != checkpoint:
            return IntegrityVerification(
                ok=False,
                reason="checkpoint_mismatch",
                expected=expected_checkpoint.chain_hash if expected_checkpoint is not None else None,
                actual=checkpoint.chain_hash,
            )
    return IntegrityVerification(ok=True)


HASH_CHAIN_LINES_SCHEMA = "archivetrust.hash_chain_lines.v1"
"""Release WS5: the append-only sidecar encoding of the same chain `HASH_CHAIN_SCHEMA` describes.
One JSON line per source record (`HashChainCheckpoint`'s fields), appended in O(1) as the source
grows — replacing the previous per-append full-manifest rewrite, which re-read and re-hashed the
entire source file on every single event (measured quadratic during the 2026-07-16 F3 campaign).
The sidecar is derived and disposable: a missing, short, or inconsistent sidecar is rebuilt from
the source, never trusted over it."""


def default_hash_chain_sidecar_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(f"{source.name}.chain.jsonl")


class HashChainAppender:
    """Maintains the append-only chain sidecar for one append-only JSONL source.

    Constructing it reconciles the sidecar with the source exactly once: a sidecar whose record
    count matches the source resumes from its last chain hash; anything else (missing, shorter,
    longer, corrupt) is rebuilt from the source in one streaming pass. After that, every
    `append_record(raw_line)` costs two SHA-256 updates and one sidecar line write.
    """

    def __init__(self, source_path: str | Path, sidecar_path: str | Path | None = None) -> None:
        self._source = Path(source_path)
        self._sidecar = (
            Path(sidecar_path) if sidecar_path is not None else default_hash_chain_sidecar_path(self._source)
        )
        self._line_count, self._last_chain_hash = self._reconcile()

    @property
    def line_count(self) -> int:
        return self._line_count

    @property
    def root_hash(self) -> str:
        return self._last_chain_hash

    def _reconcile(self) -> tuple[int, str]:
        source_records = sum(1 for _ in _jsonl_record_bytes(self._source))
        if self._sidecar.exists():
            count = 0
            last_hash = ZERO_CHAIN_HASH
            valid = True
            with self._sidecar.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        last_hash = record["chain_hash"]
                        count += 1
                    except Exception:  # noqa: BLE001 -- any damage means rebuild, never trust
                        valid = False
                        break
            if valid and count == source_records:
                return count, last_hash
        return self._rebuild()

    def _rebuild(self) -> tuple[int, str]:
        previous = ZERO_CHAIN_HASH
        count = 0
        with self._sidecar.open("w", encoding="utf-8") as sidecar:
            for raw in _jsonl_record_bytes(self._source):
                line_hash = _sha256_hex(raw)
                chain_hash = _sha256_hex(bytes.fromhex(previous) + bytes.fromhex(line_hash))
                count += 1
                sidecar.write(
                    json.dumps(
                        {
                            "line_number": count,
                            "line_hash": line_hash,
                            "previous_chain_hash": previous,
                            "chain_hash": chain_hash,
                        },
                        separators=(",", ":"),
                    )
                )
                sidecar.write("\n")
                previous = chain_hash
        return count, previous

    def append_record(self, raw_line: bytes) -> None:
        line_hash = _sha256_hex(raw_line)
        chain_hash = _sha256_hex(bytes.fromhex(self._last_chain_hash) + bytes.fromhex(line_hash))
        self._line_count += 1
        with self._sidecar.open("a", encoding="utf-8") as sidecar:
            sidecar.write(
                json.dumps(
                    {
                        "line_number": self._line_count,
                        "line_hash": line_hash,
                        "previous_chain_hash": self._last_chain_hash,
                        "chain_hash": chain_hash,
                    },
                    separators=(",", ":"),
                )
            )
            sidecar.write("\n")
        self._last_chain_hash = chain_hash


def verify_hash_chain_sidecar(
    path: str | Path, sidecar_path: str | Path | None = None
) -> IntegrityVerification:
    """Streams the source and the append-only sidecar in parallel, recomputing the chain — the
    JSONL-sidecar counterpart of `verify_hash_chain_manifest` (which continues to govern the
    legacy one-document manifest used by replay archives)."""
    source = Path(path)
    sidecar = Path(sidecar_path) if sidecar_path is not None else default_hash_chain_sidecar_path(source)
    if not sidecar.exists():
        return IntegrityVerification(ok=False, reason="manifest_missing")

    previous = ZERO_CHAIN_HASH
    count = 0
    with sidecar.open("r", encoding="utf-8") as recorded:
        recorded_lines = (line for line in recorded if line.strip())
        for raw in _jsonl_record_bytes(source):
            recorded_line = next(recorded_lines, None)
            if recorded_line is None:
                return IntegrityVerification(ok=False, reason="line_count_mismatch", actual=count)
            record = json.loads(recorded_line)
            line_hash = _sha256_hex(raw)
            chain_hash = _sha256_hex(bytes.fromhex(previous) + bytes.fromhex(line_hash))
            count += 1
            if record.get("line_hash") != line_hash or record.get("chain_hash") != chain_hash:
                return IntegrityVerification(
                    ok=False,
                    reason="checkpoint_mismatch",
                    expected=record.get("chain_hash"),
                    actual=chain_hash,
                )
            previous = chain_hash
        if next(recorded_lines, None) is not None:
            return IntegrityVerification(ok=False, reason="line_count_mismatch", expected=count)
    return IntegrityVerification(ok=True)


def build_file_digest_manifest(path: str | Path) -> FileDigestManifest:
    source = Path(path)
    payload = source.read_bytes()
    return FileDigestManifest(
        manifest_schema=FILE_DIGEST_SCHEMA,
        algorithm=HASH_ALGORITHM,
        source_name=source.name,
        created_at=_now_utc(),
        size_bytes=len(payload),
        digest=_sha256_hex(payload),
    )


def write_file_digest_manifest(path: str | Path, manifest_path: str | Path | None = None) -> Path:
    source = Path(path)
    output = Path(manifest_path) if manifest_path is not None else default_file_digest_manifest_path(source)
    output.write_text(
        build_file_digest_manifest(source).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output


def load_file_digest_manifest(path: str | Path) -> FileDigestManifest:
    return FileDigestManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def verify_file_digest_manifest(path: str | Path, manifest_path: str | Path | None = None) -> IntegrityVerification:
    source = Path(path)
    sidecar = Path(manifest_path) if manifest_path is not None else default_file_digest_manifest_path(source)
    if not sidecar.exists():
        return IntegrityVerification(ok=False, reason="manifest_missing")

    expected = load_file_digest_manifest(sidecar)
    actual = build_file_digest_manifest(source)
    if expected.size_bytes != actual.size_bytes:
        return IntegrityVerification(
            ok=False,
            reason="size_mismatch",
            expected=expected.size_bytes,
            actual=actual.size_bytes,
        )
    if expected.digest != actual.digest:
        return IntegrityVerification(
            ok=False,
            reason="digest_mismatch",
            expected=expected.digest,
            actual=actual.digest,
        )
    return IntegrityVerification(ok=True)


def _jsonl_record_bytes(path: Path) -> tuple[bytes, ...]:
    payload = path.read_bytes() if path.exists() else b""
    return tuple(line for line in payload.split(b"\n") if line.strip())


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
