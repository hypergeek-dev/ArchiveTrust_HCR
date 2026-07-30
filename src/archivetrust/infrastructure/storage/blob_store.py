"""Content-addressed blob storage for large telemetry payloads.

`put_bytes`/`get_bytes` were added alongside the RGB-normalization preprocessing stage
(`htr/preprocessing/`), which stores real page-image bytes -- original and normalized -- and needs
the identical content-addressing guarantee the text side already had. They are the *same* mechanism,
not a second one: one SHA-256 digest function, one `root/<first-2-hex>/<rest>` path layout, one
integrity check on read. `put_text`/`get_text` are now thin UTF-8 wrappers over them, so there is a
single code path that can drift rather than two that can disagree about where a digest lives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class ContentAddressedBlobStore:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes) -> tuple[str, int]:
        """Stores raw bytes under their SHA-256 digest. Returns `(digest, byte_size)`.

        Idempotent by construction: identical bytes always produce the same digest and therefore the
        same path, and an existing blob is never rewritten (the write is skipped, not repeated) --
        the same content is never stored twice, which is what makes a digest a stable identity for
        an image artifact rather than merely a checksum of one copy of it.
        """
        digest = hashlib.sha256(data).hexdigest()
        path = self._path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return digest, len(data)

    def get_bytes(self, digest: str) -> bytes:
        """Reads a blob back and *verifies* it hashes to the digest it was asked for.

        The verification is not belt-and-braces: a content-addressed store whose reads are unchecked
        cannot distinguish "this is the artifact the provenance record names" from "this is whatever
        is currently at that path", which is the entire evidentiary value of storing it by hash.
        """
        data = self._path_for(digest).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise BlobIntegrityError(
                f"blob digest mismatch: expected {digest}, got {actual}"
            )
        return data

    def put_text(self, value: str) -> tuple[str, int]:
        return self.put_bytes(value.encode("utf-8"))

    def get_text(self, digest: str) -> str:
        return self.get_bytes(digest).decode("utf-8")

    def path_for(self, digest: str) -> Path:
        """Where a digest's blob lives. Public because an export package needs to *copy* a stored
        artifact into a package directory without loading its bytes through this process."""
        return self._path_for(digest)

    def _path_for(self, digest: str) -> Path:
        return self._root / digest[:2] / digest[2:]


class BlobIntegrityError(ValueError):
    pass
