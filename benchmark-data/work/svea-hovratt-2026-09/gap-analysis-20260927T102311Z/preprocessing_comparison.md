# Preprocessing comparison (Loghi vs Lion), from repository code

Diagnostic only, derived by reading the harness code, not by running anything.
Sources: `src/archivetrust/htr/benchmark/imaging.py`, `inference.py`, `model_registry.py`.

## Shared step (identical for both models)

Both models are pointed at the exact same frozen crop file for a given `line_id`
(`benchmark-data/benchmark/.../lines/...png`), produced once during `build` by
`imaging.py`'s deterministic crop policy (`bbox_v1`): cut from the full-resolution page as decoded
(EXIF orientation never silently applied), encoded as PNG with pinned encoder settings
(`optimize=False, compress_level=6`), byte-identical for every consumer. Neither backend re-derives
or re-crops the image; `inference.py` explicitly re-verifies the frozen manifest hashes before and
after each run so this cannot silently drift.

## Loghi (`LoghiBackend`, `main.py` inside the pinned container)

- Reads the PNG as delivered (RGB or whatever mode it was saved in), the container's own loader
  converts to grayscale.
- Height-normalized to a **fixed 64px height** (VGSL input `None, None, 64, 1`); width is **not**
  fixed — it scales proportionally with the crop's own aspect ratio, so a very short/squarish crop
  becomes a very narrow input (e.g. a crop with aspect ratio ~1.3 becomes roughly 64x83px after
  resize).
- CTC output: the number of decodable timesteps is a function of the resized *width* (after the
  network's own width-downsampling stride). A narrower resized width structurally yields fewer
  timesteps. CTC requires at least one timestep per emitted (non-blank) symbol before duplicate-
  collapse; a crop resized to very few timesteps can make the correct multi-character sequence
  unreachable regardless of decoding search width, and beam search cannot recover a sequence the
  timestep budget cannot represent.
- Batched inference (`batch_size=16`); durations are batch-amortized, not per-line.
- No confidence is meaningfully exposed as a separate signal beyond what the CLI already returns per
  line (used only diagnostically here, not as a benchmark metric).

## Lion (`LionBackend`, in-process TrOCR/ViT via the Swedish Lion facade)

- The processor resizes/pads every crop into a **fixed 384x384** grid regardless of the original
  aspect ratio (TrOCR's standard ViT preprocessing), converts to RGB, normalizes with mean/std 0.5.
- Because the patch grid is fixed size, a Lion crop always gets the same amount of "spatial budget"
  (384/16 = 24x24 patches for the standard ViT-base patch size) independent of how narrow or wide the
  original line was. A tiny crop is stretched to fill the same grid a normal-width crop would use;
  it does not lose timesteps/patches the way Loghi's variable-width CTC input can.
- Autoregressive decoding (`generate`, num_beams=4) with `max_length=256`; no architectural
  "budget too small to represent the target" failure mode analogous to Loghi's CTC timestep limit.

## Why this matters for the empty-output and length-bias findings

This is a **structural, architecture-level difference in how the two models consume the exact same
crop geometry**, not a difference in what crop each model was given (the crop itself is identical and
byte-verified). It is a plausible, code-grounded explanation for why the 71 Loghi empty outputs and
the broader corpus-wide under-production concentrate so sharply on short/narrow/low-aspect-ratio
crops (see `geometry_sensitivity.json`, `empty_output_analysis.json`): Loghi's fixed-height/variable-
width CTC input can run out of usable timesteps on very small crops in a way TrOCR's fixed-size ViT
grid structurally cannot. This is offered as a **strong, code-supported hypothesis**, not a confirmed
root cause — confirming it further would require inspecting actual per-line timestep counts from the
Loghi container's internal output (not exposed by the current CLI/results.txt format) or a targeted
padding experiment (see the retraining/next-step recommendation in `GAP_ANALYSIS.md`).
