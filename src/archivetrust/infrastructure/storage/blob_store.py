"""Content-addressed blob storage for large telemetry payloads."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ContentAddressedBlobStore:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def put_text(self, value: str) -> tuple[str, int]:
        data = value.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        path = self._path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return digest, len(data)

    def get_text(self, digest: str) -> str:
        data = self._path_for(digest).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise BlobIntegrityError(
                f"blob digest mismatch: expected {digest}, got {actual}"
            )
        return data.decode("utf-8")

    def _path_for(self, digest: str) -> Path:
        return self._root / digest[:2] / digest[2:]


class BlobIntegrityError(ValueError):
    pass
