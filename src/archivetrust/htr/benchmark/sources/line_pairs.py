"""Line image + sidecar transcription (`<stem>.gt.txt` preferred, else `<stem>.txt`).

The layout used by Kraken, OCR-D, Calamari and many hand-made line datasets. The containing
directory is taken as the document; page grouping is unknown in this format and recorded as such.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from archivetrust.htr.benchmark.findings import Finding
from archivetrust.htr.benchmark.sources.base import CandidateLine, Extraction, is_image, read_text_file

ROOT_DOCUMENT = "root"
UNKNOWN_PAGE = "lines"


def natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _sidecars(image: str, names: set[str]) -> list[str]:
    path = PurePosixPath(image)
    stem = path.with_suffix("")
    return [c for c in (f"{stem}.gt.txt", f"{stem}.txt") if c in names]


class LinePairsAdapter:
    adapter_id = "line_pairs"
    version = "1"

    def detect(self, root: Path, files: list[str]) -> float:
        names = set(files)
        images = [f for f in files if is_image(f)]
        if not images:
            return 0.0
        return sum(1 for image in images if _sidecars(image, names)) / len(images)

    def extract(self, root: Path, files: list[str]) -> Extraction:
        out = Extraction(adapter_id=self.adapter_id, adapter_version=self.version)
        names = set(files)
        by_dir: dict[str, list[str]] = defaultdict(list)
        for name in files:
            if is_image(name):
                by_dir[str(PurePosixPath(name).parent)].append(name)

        used_txt: set[str] = set()
        for directory in sorted(by_dir):
            document = ROOT_DOCUMENT if directory == "." else directory
            for order, image in enumerate(sorted(by_dir[directory], key=lambda n: natural_key(PurePosixPath(n).name))):
                sidecars = _sidecars(image, names)
                gt_path, gt_text = "", None
                if sidecars:
                    gt_path = sidecars[0]
                    used_txt.update(sidecars)
                    if len(sidecars) > 1:
                        out.findings.append(Finding(
                            severity="warning", code="gt.ambiguous_sidecar", path=image,
                            message=f"both {sidecars[0]} and {sidecars[1]} exist; using {sidecars[0]} (the .gt.txt convention)",
                        ))
                    gt_text, text_findings = read_text_file(root, gt_path)
                    out.findings.extend(text_findings)
                    out.consumed_files.add(gt_path)
                else:
                    out.findings.append(Finding(severity="needs_review", code="gt.missing", path=image,
                                                message="line image has no .gt.txt/.txt transcription"))
                out.consumed_files.add(image)
                out.lines.append(CandidateLine(
                    document_raw=document, page_raw=UNKNOWN_PAGE, line_raw=PurePosixPath(image).stem, line_order=order,
                    image_kind="line", image_path=image, gt_path=gt_path, gt_source=gt_text, gt_from_line_file=True,
                    source_line_ref=image,
                ))

        for orphan in sorted(n for n in names if n.endswith(".txt") and n not in used_txt):
            out.findings.append(Finding(severity="warning", code="gt.orphan_transcription", path=orphan,
                                        message="transcription file with no matching image"))
        if out.lines:
            out.findings.append(Finding(severity="info", code="layout.page_unknown",
                                        message="line-pair format carries no page grouping; the directory is the document, all lines share page 'lines'"))
        return out
