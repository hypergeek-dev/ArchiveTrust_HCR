# HTR test fixtures

## `trolldomskommissionen_sample_line.jpg` / `.txt`

**Provenance**: one row (image + transcription) extracted from the
[`Riksarkivet/trolldomskommissionen_lines`](https://huggingface.co/datasets/Riksarkivet/trolldomskommissionen_lines)
dataset published by the Swedish National Archives (Riksarkivet) on Hugging Face. The source
material is *Trolldomskommissionen* (the Witchcraft Commission), 17th-century Swedish court
records -- real historical handwriting, not synthetic. This is the same family of material
`Riksarkivet/satrn_htr` was trained on (see its model card's 1661/1664/1688 test sets).

**License**: Riksarkivet publishes its HTR line-image datasets on the Hugging Face Hub for
research/training use (public-sector archival material); no separate license file was attached
to the dataset repo at time of extraction, so this fixture is used here strictly as a small
(<1MB), attributed, non-redistributed-in-bulk research/test sample -- one line out of ~19,000 in
the source dataset -- for adapter smoke-testing, not as a redistribution of the dataset itself.

**Extraction**: downloaded via `huggingface_hub.hf_hub_download(repo_type="dataset")`, row index
5 among transcriptions of length 20-60 chars, read from the dataset's parquet shard with
`pyarrow`, saved as-is (JPEG bytes) plus its paired ground-truth transcription text file. No
cropping, resizing, or other modification was applied -- this is exactly the image + text pair as
published upstream.

**Dimensions**: 2568x231 px, RGB.

**Ground truth transcription** (`trolldomskommissionen_sample_line.txt`, UTF-8, as published --
the replacement-character glyphs reflect the upstream dataset's own transcription convention for
uncertain/special characters, not a corruption introduced here):

```
bekiendt. Sager och deth hon Minnes hoon Tua ganger waritt
```
(see the .txt file itself for the exact UTF-8 bytes, including the upstream special-character
markers -- this README re-types it in ASCII for portability of this document only.)

Used by `tests/providers/satrn/test_adapter.py`'s real-inference test to assert SATRN's raw
transcription is non-empty and plausible, not to require an exact CER-zero match (SATRN is a
deterministic model with real recognition error, not a lookup table).
