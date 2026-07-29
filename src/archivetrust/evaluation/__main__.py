"""`python -m archivetrust.evaluation` — evaluation-reference administration.

Subcommands (all against one Workspace's durable streams; no provider is ever invoked):

  list-documents             documents with recorded canonical results, for campaign selection
  render --ref R --page N    render one page to PNG so an annotator can read the original
  add --ref R --field F ...  append one annotation (validated against the immutable object)
  annotations                print the resolved (latest) annotation set
  export --output F          export the dataset with provenance + integrity sidecar
  evaluate --output F        score providers and canonical output against verified references

Evaluation references judge outputs; they never participate in producing those outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from archivetrust.acquisition.events import FileAcquisitionTelemetrySink
from archivetrust.acquisition.manager import AcquisitionManager
from archivetrust.application.progress import FileProcessingProgressSink
from archivetrust.application.current_state import CurrentStateService
from archivetrust.domain.shared.ids import new_id
from archivetrust.evaluation.evaluate import evaluate, write_evaluation
from archivetrust.evaluation.ground_truth import (
    AdjudicationStatus,
    FileGroundTruthStore,
    GroundTruthAnnotation,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.workspace.store import WorkspaceStore


def _open_workspace(args):
    store = WorkspaceStore(app_root=Path(args.deployment_root) / "workspaces")
    workspaces = store.list()
    match = next((w for w in workspaces if w.name == args.workspace or w.id == args.workspace), None)
    if match is None:
        names = ", ".join(f"{w.name!r}" for w in workspaces)
        raise SystemExit(f"workspace {args.workspace!r} not found; available: {names}")
    layout = store.layout_for(match.id)
    telemetry = FileTelemetrySink(layout.telemetry_dir / "events.jsonl")
    manager = AcquisitionManager(
        match.id,
        layout,
        FileAcquisitionTelemetrySink(layout.telemetry_dir / "acquisition.jsonl"),
        FileProcessingProgressSink(layout.telemetry_dir / "processing.jsonl"),
    )
    gt_store = FileGroundTruthStore(layout.root / "evaluation" / "annotations.jsonl")
    return match, layout, telemetry, manager, gt_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="archivetrust-evaluate", description=__doc__)
    parser.add_argument("--deployment-root", default="archivetrust_data")
    parser.add_argument("--workspace", required=True, help="Workspace name or id")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-documents")
    sub.add_parser("annotations")

    render = sub.add_parser("render")
    render.add_argument("--ref", required=True)
    render.add_argument("--page", type=int, default=1)
    render.add_argument("--output", required=True)

    add = sub.add_parser("add")
    add.add_argument("--ref", required=True, help="archive_object ref")
    add.add_argument("--page", type=int, default=1)
    add.add_argument("--field", required=True, help="e.g. page1_heading")
    add.add_argument("--observation-type", required=True, help="e.g. heading, paragraph")
    add.add_argument("--text", default=None)
    add.add_argument("--illegible", action="store_true")
    add.add_argument("--uncertain", action="store_true")
    add.add_argument("--annotator", required=True)
    add.add_argument("--method", required=True)
    add.add_argument("--source", required=True)
    add.add_argument("--supersedes", default=None)
    add.add_argument("--adjudication", default="unadjudicated", choices=[s.value for s in AdjudicationStatus])
    add.add_argument("--notes", default=None)

    export = sub.add_parser("export")
    export.add_argument("--output", required=True)

    run = sub.add_parser("evaluate")
    run.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    workspace, layout, telemetry, manager, gt_store = _open_workspace(args)

    if args.command == "list-documents":
        for state in CurrentStateService(telemetry).documents():
            if state.latest_canonical_document is not None:
                archive_object = manager.archive_object_by_ref(state.document_ref)
                name = archive_object.original_filename if archive_object else "(unknown file)"
                print(f"{state.document_ref}\t{name}")
        return 0

    if args.command == "render":
        from archivetrust.infrastructure.rendering.pdf_renderer import render_pdf_page

        archive_object = manager.archive_object_by_ref(args.ref)
        if archive_object is None:
            raise SystemExit(f"unknown archive object {args.ref}")
        pdf_path = layout.archive_dir / archive_object.storage_path
        rendered = render_pdf_page(pdf_path, args.page)
        rendered.image.save(args.output)
        print(json.dumps({"output": args.output, "file": archive_object.original_filename}))
        return 0

    if args.command == "add":
        archive_object = manager.archive_object_by_ref(args.ref)
        if archive_object is None:
            raise SystemExit(f"unknown archive object {args.ref}")
        annotation = GroundTruthAnnotation(
            annotation_id=new_id("gt"),
            archive_object_ref=args.ref,
            content_hash=archive_object.content_hash,
            page=args.page,
            field=args.field,
            observation_type=args.observation_type,
            text=args.text,
            uncertain=args.uncertain,
            illegible=args.illegible,
            annotator=args.annotator,
            method=args.method,
            source=args.source,
            created_at=datetime.now(timezone.utc).isoformat(),
            revision=1,
            supersedes=args.supersedes,
            adjudication_status=AdjudicationStatus(args.adjudication),
            notes=args.notes,
        )
        gt_store.append(annotation, expected_content_hash=archive_object.content_hash)
        print(json.dumps({"annotation_id": annotation.annotation_id, "store": str(gt_store.path)}))
        return 0

    if args.command == "annotations":
        for record in gt_store.latest():
            print(record.model_dump_json(by_alias=True))
        return 0

    if args.command == "export":
        output = gt_store.export_with_provenance(args.output)
        print(json.dumps({"output": str(output)}))
        return 0

    if args.command == "evaluate":
        result = evaluate(store=gt_store, telemetry_source=telemetry)
        output = write_evaluation(result, args.output)
        print(json.dumps({"output": str(output), "central_result": result.central_result}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
