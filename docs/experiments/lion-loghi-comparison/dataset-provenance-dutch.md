# Dutch Corpus Provenance — `dataset-dutch-rgb/`

Status: Current
Governs: Where the Dutch corpus came from, what is and is not known about its licensing, and what was
done to prepare it for this repository.

## What was provided, and what was done with it

`D:\Downloads\republic7.zip` (3.6 GB, `sha256` not separately recorded — the archive itself is not
committed, only its extracted, gitignored contents under `dataset-dutch-rgb/`) was provided directly by
the user during this integration. It was extracted, unmodified, into `dataset-dutch-rgb/republic7/`,
preserving its own internal `train/`/`val/` split and `page/` ground-truth subdirectories exactly as
packaged. Nothing inside it was renamed, recompressed, or altered. `scripts/inventory_dutch_dataset.py`
computed the real per-page content hashes and image properties recorded in
`dataset-dutch-manifest.json`/`dataset-dutch-inventory.csv` — measured from the actual extracted files,
not estimated.

## What the archive itself states

Read directly from the extracted contents (515 pages: 465 `train/`, 50 `val/`; 514 of 515 paired with a
`page/*.xml` ground-truth file — `NL-HaNA_1.10.94_437_0028.jpg` has no matching XML):

* **Image filenames** follow the pattern `NL-HaNA_<archive-code>_<inventory-number>_<page-number>.jpg`.
  `NL-HaNA` is the standard reference prefix for the Nationaal Archief (Dutch National Archive, The
  Hague). The two archive codes present are `1.01.02` (the overwhelming majority) and `1.10.94`. `1.01.02`
  is the Nationaal Archief's published reference code for the **Staten-Generaal** fonds (States General
  of the Dutch Republic) — this identification is read off the well-known, publicly documented Nationaal
  Archief finding-aid numbering convention, not asserted from the image content itself.
* **PAGE XML ground truth** (schema `http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15`)
  carries `<TranskribusMetadata docId="..." pageId="..." status="GT" ...>` on every page that has one —
  i.e. these are Transkribus-platform-annotated ground-truth transcriptions, `status="GT"` being
  Transkribus's own "ground truth" marker. `<Creator>` metadata additionally names
  `de.uros.citlab.segmentation.CITlab_LA_ML` (University of Rostock / CITlab layout-analysis tooling),
  indicating the region/line segmentation was produced or assisted by that tool before transcription.
* **No license file, README, or citation file is present inside the archive.** No SPDX identifier, no
  `LICENSE`, no `CITATION.cff`, nothing.

## What is inferred, clearly labeled as inference, and why

The filename/schema conventions above (`NL-HaNA_1.01.02_...`, PAGE XML with Transkribus GT metadata,
CITlab segmentation) match the publicly known shape of the **REPUBLIC project**'s training data — a
KNAW Humanities Cluster digital-humanities project that used exactly this Nationaal Archief
Staten-Generaal material to build ground-truth transcription datasets, and whose data has in turn been
used to train HTR models including Loghi's own Dutch models. This identification is offered as a
**plausible, non-verified inference** based on matching a well-known public pattern, not as a confirmed
fact read from the archive itself — no file inside the zip names "REPUBLIC," Loghi, or a specific
publication.

## Licensing — explicitly unresolved, not assumed

**No license is asserted for this corpus.** The brief's instruction is unambiguous: "Do not download or
choose a Dutch dataset automatically without recording provenance and licensing." Provenance is recorded
above to the extent it is knowable from the files themselves. Licensing is **not** recorded because it
is not stated anywhere in the archive, and this document does not guess one.

**Before any external use, publication, or redistribution of this corpus or derivatives of it (including
model outputs trained or evaluated on it), the exact license terms must be confirmed from an
authoritative source** — most plausibly the Nationaal Archief's own reuse terms for `1.01.02`/`1.10.94`
scans, and/or the REPUBLIC project's own published dataset license if the inference above is correct
(their public releases have historically used open licenses, but this document does not cite one it has
not verified for this specific archive). Until confirmed, this corpus should be treated as **internal
research use only** within this repository.

## Known limitation

The archive's origin was not independently verified against a live network source (no network access
was used to confirm the REPUBLIC-project inference or to locate an authoritative license) — this
document records what is directly observable in the provided file and clearly separates that from
inference. A future pass with network access should confirm or correct the inference above and locate
the authoritative license before this corpus is used beyond internal research.
