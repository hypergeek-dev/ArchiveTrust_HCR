"""Converts the Florence-2 line segmentation recorded by the 2026-07-31 technical-reliability run into
the benchmark harness's segmentation-import JSONL, for the NO_GROUND_TRUTH mechanical dry run on
`dataset-rgb/`. This is not an accuracy benchmark.

It joins TextLineDetected (line box, reading order) -> RegionDetected (page_id) -> the run report's
page list (page_id -> dataset-rgb relative path, raster size). It is read-only on the run directory.

    python scripts/dryrun_segmentation_from_reliability.py <out.jsonl>
    python -m archivetrust.htr.benchmark dryrun-build dataset-rgb <out.jsonl> dataset-rgb-dryrun
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "docs/experiments/technical-reliability-screening/full-run/reliability-2026-07-31"


def main(out_path: str) -> int:
    report = json.loads((RUN / "reliability-run-report.json").read_text(encoding="utf-8"))
    pages = {p["page_id"]: p for p in report["configuration"]["pages"]}
    adapter = {p["page_id"]: p["adapter"] for p in report["segmentation"]["pages"]}
    region_page: dict[str, str] = {}
    lines = []
    with (RUN / "events.jsonl").open(encoding="utf-8") as events:
        for raw in events:
            event = json.loads(raw)
            if event["kind"] == "RegionDetected":
                region_page[event["region"]["region_id"]] = event["region"]["page_id"]
            elif event["kind"] == "TextLineDetected":
                lines.append(event["text_line"])
    rows = []
    for line in lines:
        page = pages[region_page[line["region_id"]]]
        box = line["bounding_box"]
        rows.append({
            "page_image": page["relative_path"].removeprefix("dataset-rgb/"),
            "page_width": page["width"], "page_height": page["height"],
            "line_key": line["text_line_id"].removeprefix("text_line_"),
            "bbox": [box["x0"], box["y0"], box["x1"], box["y1"]],
            "reading_order": line["reading_order_index"],
            "segmentation_source": f"{adapter[page['page_id']]} ({report['run_id']})",
        })
    rows.sort(key=lambda r: (r["page_image"], r["reading_order"], r["line_key"]))
    Path(out_path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(f"{len(rows)} lines on {len({r['page_image'] for r in rows})} pages -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
