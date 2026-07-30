"""Stage 1 dataset inventory for the Swedish Historical HTR Technical Reliability Screening.

**Pure metadata inspection. No model inference, no GPU, no network.** The only third-party import is
Pillow, used exactly as `htr/preprocessing/rgb_normalization.py::_observe` uses it -- to record a
source image's honest, pre-transformation properties. Nothing here writes to `dataset-rgb/`; the
dataset is opened read-only and treated as immutable evidence.

Run (from the checkout that holds `dataset-rgb/`):

    PYTHONPATH=src .venv/Scripts/python.exe scripts/inventory_technical_reliability_screening.py

Outputs `dataset-manifest.json` and `dataset-inventory.csv` into
`docs/experiments/technical-reliability-screening/`.

## Identity scheme (documented because it was a decision, not a default)

`domain/shared/ids.py` offers two mechanisms. `new_id()` is a random uuid4 -- correct for domain
objects whose identity is not their content, and **wrong here**: an inventory that is re-run must
produce the same ids, or the manifest is not diffable and no reviewer can tell a re-scan from a
change. So this script uses the codebase's *other*, deterministic mechanism, `content_address()`,
for every assigned identifier:

* `dataset_id`      = content_address(DATASET_NAME, prefix="dataset")
* `dataset_version` = content_address(DATASET_NAME, corpus digest, prefix="dataset_version")
                      -- derived from the sorted content hashes of every file inventoried, so any
                      change to the corpus yields a new version id automatically.
* `document_id`     = content_address(dataset_id, collection, volume, prefix="document")
* `page_id`         = content_address(document_id, repo-relative path, prefix="page")

`page_id` is path-derived rather than content-derived on purpose: a page is a *slot* in an archival
volume. Two byte-identical scans are still two pages (and are reported as duplicates below), and
re-scanning the same slot with corrected bytes must not silently become a different page.

The **content hash** is separate from all of the above and reuses
`htr/preprocessing/models.py::PageImageArtifact.compute_hash` verbatim -- these files *are* original,
un-normalized page images, which is exactly what that model addresses (`page_image_<sha256>`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from archivetrust.domain.shared.ids import content_address
from archivetrust.htr.preprocessing.models import PageImageArtifact

# Large archival scans (~20 MP median) legitimately exceed Pillow's decompression-bomb guard, which
# defaults to ~89 MP. Raising it is required to read real material; it is not a safety bypass for
# untrusted input -- this is a curated local archival corpus.
Image.MAX_IMAGE_PIXELS = 300_000_000

DATASET_NAME = "swedish-witchcraft-trial-records-rgb"
INVENTORY_SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------------------------
# Classification policy. Every threshold below is derived from an observed reconnaissance pass
# over this exact corpus, and each is recorded on the record it fires for, so no classification
# is ever an unexplained verdict.
# --------------------------------------------------------------------------------------------

SPREAD_ASPECT_RATIO = 1.15
"""width/height at or above which a page image is treated as a **double-page spread** (an open bound
volume photographed flat: two facing text blocks, a central gutter, two independent reading orders).

Visually validated on real files from this corpus before adoption, not assumed from the number. The
observed population is cleanly bimodal -- portrait images cluster at ratio <= 1.0 and landscape ones
at >= 1.26, with **no file anywhere between 1.0 and 1.26** -- so this threshold sits inside an empty
gap and no file is a borderline call."""

LOW_BYTES_PER_PIXEL = 0.31
HIGH_BYTES_PER_PIXEL = 1.42
"""Outlier bounds on encoded bytes per pixel (observed p01 / p99 of this corpus).

**These are not compression-artifact detectors and are not described as such.** Every file here is
lossless PNG, so bytes-per-pixel measures raster *entropy*, not codec loss. A very low value means a
smooth, low-detail raster (consistent with a faint, low-contrast, or heavily denoised scan); a very
high value means a dense, noisy raster (consistent with heavy grain, show-through, or bleed-through
from the reverse of the leaf). Both extremes are worth a human look; neither is proof of a defect,
which is why they produce a flag and a recorded reason rather than an exclusion."""

LARGE_PIXEL_COUNT = 40_000_000
"""Rasters above this size stress per-page memory during any future segmentation/tiling stage."""

_VOLUME_RE = re.compile(r"__([A-Za-z]\d{4,})_(\d+)$")


@dataclass
class PageRecord:
    """One row of the inventory. Field order is the CSV column order."""

    dataset_id: str
    dataset_version: str
    page_id: str
    document_id: str
    collection: str
    archival_volume: str
    page_number: str
    relative_path: str
    content_hash: str
    file_size_bytes: int
    readability_status: str
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: float | None = None
    pixel_count: int | None = None
    color_mode: str | None = None
    channel_count: int | None = None
    bit_depth: int | None = None
    bytes_per_pixel: float | None = None
    icc_profile_present: bool | None = None
    alpha_present: bool | None = None
    exif_orientation: int | None = None
    duplicate_status: str = "unique"
    duplicate_of_page_id: str | None = None
    inclusion_status: str = "included"
    exclusion_reason: str | None = None
    automatic_category: str | None = None
    classification_reason: str | None = None
    supervised_final_category: str | None = None
    """Deliberately empty. Stage 1 produces a *suggestion*; the supervised final classification is a
    Checkpoint 1 human decision and this script never fills it in."""
    flag_double_page_spread: bool = False
    flag_low_entropy_raster: bool = False
    flag_high_entropy_raster: bool = False
    flag_large_raster: bool = False
    notes: str = ""
    proposed_sample_member: bool = False


def _channel_count(mode: str) -> int:
    """Mirrors `rgb_normalization._channel_count` so the inventory and the normalization stage
    describe the same image the same way."""
    return {"1": 1, "L": 1, "La": 2, "LA": 2, "P": 1, "PA": 2, "I": 1, "F": 1, "CMYK": 4}.get(
        mode, 4 if mode in ("RGBA", "RGBa") else 3 if mode == "RGB" else 1
    )


def _bit_depth(mode: str) -> int:
    """Mirrors `rgb_normalization._bit_depth`."""
    if mode == "1":
        return 1
    if mode in {"I", "I;16", "I;16B", "I;16L", "I;16N"}:
        return 16
    return 8


_ALPHA_MODES = frozenset({"LA", "RGBA", "PA", "La", "RGBa"})
_EXIF_ORIENTATION_TAG = 274


def inspect(path: Path, root: Path, dataset_id: str) -> PageRecord:
    """Reads one file and records what is actually true of it. Never raises: an unreadable file is a
    recorded fact with a reason, which is the whole point of a readability audit."""
    rel = path.relative_to(root.parent).as_posix()
    collection = path.parent.name
    stem = path.stem
    match = _VOLUME_RE.search(stem)
    volume = match.group(1) if match else collection
    page_number = match.group(2) if match else stem

    document_id = content_address(dataset_id, collection, volume, prefix="document")
    page_id = content_address(document_id, rel, prefix="page")

    size = path.stat().st_size
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        return PageRecord(
            dataset_id=dataset_id,
            dataset_version="",
            page_id=page_id,
            document_id=document_id,
            collection=collection,
            archival_volume=volume,
            page_number=page_number,
            relative_path=rel,
            content_hash="",
            file_size_bytes=size,
            readability_status="unreadable",
            inclusion_status="excluded",
            exclusion_reason=f"file could not be read: {type(exc).__name__}: {exc}",
        )

    content_hash = PageImageArtifact.compute_hash(image_bytes)

    record = PageRecord(
        dataset_id=dataset_id,
        dataset_version="",
        page_id=page_id,
        document_id=document_id,
        collection=collection,
        archival_volume=volume,
        page_number=page_number,
        relative_path=rel,
        content_hash=content_hash,
        file_size_bytes=size,
        readability_status="readable",
    )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            # Force a full decode, exactly as the normalization stage does, so a truncated or
            # corrupt raster fails here rather than in a later pipeline stage.
            image.load()
            record.image_format = image.format
            record.width = image.width
            record.height = image.height
            record.color_mode = image.mode
            record.channel_count = _channel_count(image.mode)
            record.bit_depth = _bit_depth(image.mode)
            record.icc_profile_present = bool(image.info.get("icc_profile"))
            record.alpha_present = image.mode in _ALPHA_MODES or "transparency" in image.info
            try:
                orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
                record.exif_orientation = (
                    int(orientation) if isinstance(orientation, int) else None
                )
            except Exception:  # noqa: BLE001 -- unreadable EXIF is an absence, not a failure
                record.notes = "EXIF block present but unreadable"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        record.readability_status = "undecodable"
        record.inclusion_status = "excluded"
        record.exclusion_reason = f"image could not be decoded: {type(exc).__name__}: {exc}"
        return record

    if not record.width or not record.height:
        record.readability_status = "undecodable"
        record.inclusion_status = "excluded"
        record.exclusion_reason = "decoded image reported a zero dimension"
        return record

    record.pixel_count = record.width * record.height
    record.aspect_ratio = round(record.width / record.height, 4)
    record.bytes_per_pixel = round(size / record.pixel_count, 4)
    return record


def classify(record: PageRecord) -> None:
    """Assigns the *automatic* primary category and records why.

    Precedence is `Structurally difficult` > `Technically difficult` > `Ordinary`, and that order is
    a reasoned choice: the structural signal is directly observed page geometry (and was visually
    validated on this corpus), whereas the technical signals are distribution-relative proxies that
    no automatic pass can confirm. Stronger evidence therefore wins the single primary slot.

    Collapsing to one primary category is lossy -- the axes are orthogonal -- so every signal is
    *also* written to its own `flag_*` column, letting a reviewer re-derive any other precedence
    from the manifest without re-running this script.
    """
    if record.inclusion_status == "excluded":
        record.automatic_category = None
        record.classification_reason = "not classified: excluded from the usable population"
        return

    reasons: list[str] = []

    if record.aspect_ratio is not None and record.aspect_ratio >= SPREAD_ASPECT_RATIO:
        record.flag_double_page_spread = True
        reasons.append(
            f"aspect ratio {record.aspect_ratio} >= {SPREAD_ASPECT_RATIO}: double-page spread "
            f"(two facing text blocks, central gutter, two independent reading orders)"
        )

    if record.bytes_per_pixel is not None and record.bytes_per_pixel < LOW_BYTES_PER_PIXEL:
        record.flag_low_entropy_raster = True
        reasons.append(
            f"{record.bytes_per_pixel} encoded bytes/pixel < {LOW_BYTES_PER_PIXEL}: unusually "
            f"low-entropy lossless raster (faint, low-contrast or heavily denoised scan)"
        )
    if record.bytes_per_pixel is not None and record.bytes_per_pixel > HIGH_BYTES_PER_PIXEL:
        record.flag_high_entropy_raster = True
        reasons.append(
            f"{record.bytes_per_pixel} encoded bytes/pixel > {HIGH_BYTES_PER_PIXEL}: unusually "
            f"high-entropy lossless raster (heavy grain, show-through or bleed-through)"
        )
    if record.pixel_count is not None and record.pixel_count > LARGE_PIXEL_COUNT:
        record.flag_large_raster = True
        reasons.append(
            f"{record.pixel_count / 1e6:.1f} MP > {LARGE_PIXEL_COUNT / 1e6:.0f} MP: large raster, "
            f"per-page memory pressure in any future segmentation stage"
        )
    if record.color_mode != "RGB" or record.bit_depth != 8:
        reasons.append(
            f"non-baseline raster representation: mode={record.color_mode}, "
            f"bit_depth={record.bit_depth}"
        )
    if record.icc_profile_present:
        reasons.append("embedded ICC profile present")
    if record.alpha_present:
        reasons.append("alpha channel present")
    if record.exif_orientation not in (None, 1):
        reasons.append(f"non-trivial EXIF orientation {record.exif_orientation}")

    technical = (
        record.flag_low_entropy_raster
        or record.flag_high_entropy_raster
        or record.flag_large_raster
        or record.color_mode != "RGB"
        or record.bit_depth != 8
        or bool(record.icc_profile_present)
        or bool(record.alpha_present)
        or record.exif_orientation not in (None, 1)
    )

    if record.flag_double_page_spread:
        record.automatic_category = "Structurally difficult"
    elif technical:
        record.automatic_category = "Technically difficult"
    else:
        record.automatic_category = "Ordinary"
        reasons.append(
            "single-page portrait raster with no detected technical anomaly; note that "
            "structural difficulty beyond page-spread geometry (dense marginalia, tables, "
            "multi-column layout, insertions, damage) is NOT automatically detectable without "
            "a layout model and is explicitly out of scope for this pass -- this page may be "
            "reclassified 'Structurally difficult' by supervised human correction"
        )

    record.classification_reason = "; ".join(reasons)


def build_stratified_sample(
    records: list[PageRecord], *, target: int, seed: int
) -> dict:
    """A **proposed**, reproducible stratified sample -- not a binding selection.

    Algorithm, stated so it can be re-executed independently:

    1. Candidate population = every record with `inclusion_status == "included"`.
    2. Strata = the automatic primary category, in the fixed order
       (`Ordinary`, `Technically difficult`, `Structurally difficult`).
    3. Allocation: every non-empty stratum is first guaranteed `min(size, MIN_PER_STRATUM)` slots,
       so a small stratum cannot be rounded out of existence. The remaining quota is distributed
       across strata proportionally to stratum size by the largest-remainder method, ties broken by
       the fixed stratum order.
    4. Selection within a stratum spreads across archival collections: collections are visited in
       sorted order, round-robin, and at each visit one page is drawn uniformly at random from that
       collection's remaining pool, until the stratum quota is met. This maximizes collection
       coverage while keeping the draw random.
    5. All randomness comes from a single `random.Random(seed)` consumed in that deterministic
       order, so the same seed always yields the same sample.
    """
    min_per_stratum = 5
    order = ("Ordinary", "Technically difficult", "Structurally difficult")
    rng = random.Random(seed)

    candidates = [r for r in records if r.inclusion_status == "included"]
    strata: dict[str, list[PageRecord]] = {name: [] for name in order}
    for record in candidates:
        strata.setdefault(record.automatic_category or "Ordinary", []).append(record)

    sizes = {name: len(strata.get(name, [])) for name in order}
    total = sum(sizes.values())

    allocation = {name: min(sizes[name], min_per_stratum) for name in order if sizes[name]}
    remaining = target - sum(allocation.values())

    if remaining > 0 and total:
        headroom = {n: sizes[n] - allocation.get(n, 0) for n in order}
        exact = {n: remaining * (sizes[n] / total) for n in order if sizes[n]}
        floors = {n: min(int(v), headroom[n]) for n, v in exact.items()}
        for name, value in floors.items():
            allocation[name] = allocation.get(name, 0) + value
        leftover = target - sum(allocation.values())
        remainders = sorted(
            ((exact[n] - int(exact[n]), order.index(n), n) for n in exact),
            key=lambda t: (-t[0], t[1]),
        )
        index = 0
        while leftover > 0 and index < len(remainders) * 4:
            name = remainders[index % len(remainders)][2]
            if allocation.get(name, 0) < sizes[name]:
                allocation[name] = allocation.get(name, 0) + 1
                leftover -= 1
            index += 1

    selected: list[PageRecord] = []
    per_stratum: dict[str, list[str]] = {}
    for name in order:
        pool = strata.get(name, [])
        quota = min(allocation.get(name, 0), len(pool))
        by_collection: dict[str, list[PageRecord]] = defaultdict(list)
        for record in pool:
            by_collection[record.collection].append(record)
        for bucket in by_collection.values():
            bucket.sort(key=lambda r: r.relative_path)

        chosen: list[PageRecord] = []
        collections = sorted(by_collection)
        while len(chosen) < quota and any(by_collection[c] for c in collections):
            for collection in collections:
                if len(chosen) >= quota:
                    break
                bucket = by_collection[collection]
                if not bucket:
                    continue
                chosen.append(bucket.pop(rng.randrange(len(bucket))))
        chosen.sort(key=lambda r: r.relative_path)
        per_stratum[name] = [r.page_id for r in chosen]
        selected.extend(chosen)

    chosen_ids = {r.page_id for r in selected}
    for record in records:
        record.proposed_sample_member = record.page_id in chosen_ids

    return {
        "status": "PROPOSED -- not binding; requires supervisor approval at Checkpoint 1",
        "reason_not_binding": (
            "The strata are automatic suggestions only. Structural difficulty beyond page-spread "
            "geometry is not automatically detectable without a layout model, so the categories "
            "this sample stratifies on are expected to change under supervised human correction. "
            "Approving this sample before correcting the categories would freeze a stratification "
            "known to be incomplete."
        ),
        "target_size": target,
        "seed": seed,
        "min_per_stratum": min_per_stratum,
        "candidate_population": len(candidates),
        "stratum_sizes": sizes,
        "allocation": allocation,
        "selected_count": len(selected),
        "selected_page_ids_by_stratum": per_stratum,
        "excluded_from_sample_count": len(candidates) - len(selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset-rgb")
    parser.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "docs"
            / "experiments"
            / "technical-reliability-screening"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.is_dir():
        print(f"dataset root not found: {root}", file=sys.stderr)
        return 2

    dataset_id = content_address(DATASET_NAME, prefix="dataset")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"total files found: {len(files)}")

    records = [inspect(path, root, dataset_id) for path in files]

    # Duplicate detection by content hash. The first path in sorted order is the retained original;
    # every later file with the same bytes is marked and excluded, never deleted.
    seen: dict[str, PageRecord] = {}
    for record in records:
        if not record.content_hash:
            continue
        original = seen.get(record.content_hash)
        if original is None:
            seen[record.content_hash] = record
        else:
            record.duplicate_status = "duplicate"
            record.duplicate_of_page_id = original.page_id
            record.inclusion_status = "excluded"
            record.exclusion_reason = (
                f"byte-identical duplicate of {original.relative_path}"
            )
            if original.duplicate_status == "unique":
                original.duplicate_status = "duplicate_original"

    corpus_digest = hashlib.sha256(
        "\x00".join(sorted(r.content_hash for r in records if r.content_hash)).encode()
    ).hexdigest()
    dataset_version = content_address(DATASET_NAME, corpus_digest, prefix="dataset_version")
    for record in records:
        record.dataset_version = dataset_version

    for record in records:
        classify(record)

    usable = [r for r in records if r.inclusion_status == "included"]
    unreadable = [r for r in records if r.readability_status != "readable"]
    duplicates = [r for r in records if r.duplicate_status == "duplicate"]
    categories = Counter(r.automatic_category for r in usable)

    print(f"usable (included): {len(usable)}")
    print(f"unreadable/undecodable: {len(unreadable)}")
    print(f"duplicate files (excluded): {len(duplicates)}")
    print("category distribution (automatic suggestion):")
    for name in ("Ordinary", "Technically difficult", "Structurally difficult"):
        print(f"  {name}: {categories.get(name, 0)}")

    if len(usable) <= args.sample_size:
        sample = {
            "status": "all usable pages included -- no sampling required",
            "rule": f"usable ({len(usable)}) <= threshold ({args.sample_size})",
            "candidate_population": len(usable),
            "selected_count": len(usable),
        }
        for record in usable:
            record.proposed_sample_member = True
        print(
            f"proposed sample size: {len(usable)} (<= {args.sample_size}: all usable pages used)"
        )
    else:
        sample = build_stratified_sample(records, target=args.sample_size, seed=args.seed)
        print(
            f"proposed sample size: {sample['selected_count']} "
            f"(> {args.sample_size} usable: stratified sample required)"
        )
        print(f"  seed: {sample['seed']}, allocation: {sample['allocation']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in records]
    columns = list(rows[0].keys()) if rows else []

    csv_path = out_dir / "dataset-inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/inventory_technical_reliability_screening.py",
        "stage": "Stage 1 -- dataset inventory (Checkpoint 1 material, not approved)",
        "no_inference_performed": True,
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_name": DATASET_NAME,
            "dataset_version": dataset_version,
            "corpus_digest_sha256": corpus_digest,
            "root_relative_path": root.name,
            "description": (
                "Real page images from Swedish witchcraft-trial court records "
                "(Riksarkivet-style collections, 1584-1764), never committed to git."
            ),
        },
        "counts": {
            "total_files": len(records),
            "usable": len(usable),
            "unreadable_or_undecodable": len(unreadable),
            "duplicates_excluded": len(duplicates),
            "distinct_content_hashes": len(seen),
            "collections": len({r.collection for r in records}),
            "documents": len({r.document_id for r in records}),
        },
        "category_distribution_automatic": {
            name: categories.get(name, 0)
            for name in ("Ordinary", "Technically difficult", "Structurally difficult")
        },
        "classification_policy": {
            "precedence": [
                "Structurally difficult",
                "Technically difficult",
                "Ordinary",
            ],
            "spread_aspect_ratio_threshold": SPREAD_ASPECT_RATIO,
            "low_bytes_per_pixel_threshold": LOW_BYTES_PER_PIXEL,
            "high_bytes_per_pixel_threshold": HIGH_BYTES_PER_PIXEL,
            "large_pixel_count_threshold": LARGE_PIXEL_COUNT,
            "known_limitation": (
                "Structural difficulty other than double-page-spread geometry (marginalia "
                "density, tables, multi-column layout, insertions, physical damage) is NOT "
                "automatically detectable without a layout model, which is out of scope for this "
                "pass. Pages carrying such difficulty are conservatively reported as 'Ordinary' "
                "and require supervised human correction."
            ),
            "bytes_per_pixel_caveat": (
                "Every file is lossless PNG, so bytes-per-pixel measures raster entropy, not "
                "compression artifacts. It is reported as a detail/noise proxy only."
            ),
        },
        "identity_scheme": {
            "mechanism": "archivetrust.domain.shared.ids.content_address (deterministic)",
            "why_not_new_id": (
                "new_id() is uuid4-random; re-running the inventory would mint new ids and make "
                "the manifest undiffable. Deterministic ids make a re-scan verifiable."
            ),
            "content_hash": (
                "htr/preprocessing/models.py::PageImageArtifact.compute_hash "
                "(page_image_<sha256>), reused verbatim"
            ),
        },
        "proposed_sample": sample,
        "supervised_decisions_deferred_to_checkpoint_1": [
            "Final sample size and whether the proposed stratified sample is adopted.",
            "Correction of every automatic category, especially 'Ordinary' pages that are in "
            "fact structurally difficult.",
            "Whether double-page spreads are split into single pages before recognition, or "
            "processed whole.",
            "Review sampling rate and the human-plausibility rating scale.",
        ],
    }

    manifest_path = out_dir / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {csv_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
