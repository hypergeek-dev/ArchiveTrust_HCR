from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archivetrust.replay.distribution import ReplayArchiveError, export_archive_json, summarize_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="archivetrust-replay",
        description="Replay ArchiveTrust telemetry archives without providers.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    summary = subcommands.add_parser("summary", help="Print replay-derived summary metrics as JSON")
    summary.add_argument("archive", type=Path)

    export = subcommands.add_parser("export", help="Write an A3-compatible canonical JSON export")
    export.add_argument("archive", type=Path)
    export.add_argument("--output", "-o", type=Path, required=True)
    export.add_argument("--workspace-id", default="replay-archive")
    export.add_argument("--workspace-name", default="Replay archive")
    export.add_argument("--integrity-sidecar", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.command == "summary":
            result = summarize_archive(args.archive)
            print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
            return 0

        output = export_archive_json(
            args.archive,
            args.output,
            workspace_id=args.workspace_id,
            workspace_name=args.workspace_name,
            integrity_sidecar=args.integrity_sidecar,
        )
        print(json.dumps({"output": str(output)}, indent=2, sort_keys=True))
        return 0
    except ReplayArchiveError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
