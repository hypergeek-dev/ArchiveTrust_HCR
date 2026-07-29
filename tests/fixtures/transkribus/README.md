# Transkribus test fixtures

**Provenance: every file in this directory is hand-authored for this test suite.** None of these
are real Transkribus export output -- no network access to a real Transkribus account/job was
available to produce genuine exports for this phase, and the task brief explicitly permits this:
"If you can find genuinely representative small public sample files ... use those and document
provenance; otherwise hand-author small but schema-valid PAGE XML / ALTO XML fixtures yourself
... clearly label them as hand-authored test fixtures, not real Transkribus output." These fixtures
are schema-conformant to the real PAGE XML (`schema.primaresearch.org/PAGE/gts/pagecontent/
2019-07-15`) and ALTO (`www.loc.gov/standards/alto/ns-v4#`) namespaces and element/attribute
vocabularies (`TextRegion`/`TextLine`/`Coords`/`Baseline`/`TextEquiv`/`Unicode`/`MetadataItem`/
`ReadingOrder` for PAGE; `TextBlock`/`TextLine`/`String`/`WC`/`processingDateTime` for ALTO), not
invented syntax -- but the *content* (the transcribed Swedish text, the ids, the metadata values)
is fabricated for testing, not transcribed from a real archival document.

The Swedish-language line text (`"Anno 1712 den 3 Januarii holltes ting medh allmogen aff Sochnen"`,
`"NB dombook"`) is a plausible-looking but invented 18th-century Swedish court-record phrase, not
copied from any real source -- written to exercise non-ASCII characters and realistic line lengths,
not to represent genuine historical content.

## Files

- `sample_page.xml` -- valid PAGE XML: 2 `TextRegion`s, 3 `TextLine`s total, explicit
  `<ReadingOrder>` block, per-line `TextEquiv/@conf`, `Coords`/`Baseline` geometry, and a
  `Metadata/MetadataItem` block carrying `vendorReportedAccuracy`, `modelVersion`, `jobId`,
  `transkribusDocId` -- the standard PAGE XML mechanism for arbitrary custom metadata, used here to
  exercise this adapter's vendor-accuracy/model-hint/job-id extraction.
- `sample_alto.xml` -- valid ALTO XML: 2 `TextBlock`s, 3 `TextLine`s, per-`String` `WC` word
  confidence, `HPOS`/`VPOS`/`WIDTH`/`HEIGHT` geometry, `Description/OCRProcessing/
  ocrProcessingStep/processingDateTime`.
- `sample_plain.txt` -- the same three lines as plain text, no markup.
- `malformed_page.xml` -- PAGE XML with an unterminated `<Unicode>` element and a missing closing
  `</PcGts>` tag -- a genuine XML well-formedness error (`xml.etree.ElementTree.ParseError`), not
  merely semantically odd-but-parseable content.
- `malformed_alto.xml` -- ALTO XML with an unclosed `<String .../>` tag (missing `/>`) -- same
  category of genuine well-formedness error.
- `missing_page_element.xml` -- well-formed PAGE XML (valid `<PcGts>` root, valid `<Metadata>`)
  that omits the required `<Page>` element entirely -- exercises the "well-formed XML, wrong/
  incomplete structure" failure category, distinct from a parse error.
- `no_confidence_page.xml` -- valid PAGE XML with a `<TextLine>` whose `<TextEquiv>` carries no
  `conf` attribute at all -- exercises the "confidence genuinely absent, must not be fabricated"
  case.
- `empty_file.txt` -- a zero-byte file, used for the plain-text `empty_file` failure-category test
  (also reused as a stand-in empty PAGE/ALTO input in unit tests that pass empty strings directly).
