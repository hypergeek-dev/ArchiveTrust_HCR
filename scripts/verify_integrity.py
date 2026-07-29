from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archivetrust.infrastructure.storage.integrity import (
    verify_file_digest_manifest,
    verify_hash_chain_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify ArchiveTrust integrity sidecars.")
    parser.add_argument("path", type=Path, help="File to verify")
    parser.add_argument(
        "--kind",
        choices=("hash-chain", "file-digest"),
        default="hash-chain",
        help="Integrity sidecar type to verify",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit sidecar path")
    args = parser.parse_args(argv)

    if args.kind == "hash-chain":
        result = verify_hash_chain_manifest(args.path, args.manifest)
    else:
        result = verify_file_digest_manifest(args.path, args.manifest)

    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
