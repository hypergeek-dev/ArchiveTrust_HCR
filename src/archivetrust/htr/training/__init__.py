"""Training-specific HTR modules -- the Swedish Loghi fine-tuning pilot
(docs/methods/loghi-swedish-finetuning.md).

A materially different concern from `htr/screening/` (which drives inference-only benchmark runs over
already-trained methods): this package builds, validates, and trains a *new* candidate checkpoint
(`loghi_swedish_finetuned_v1`) from the pinned generic Loghi-HTR model. Nothing here alters
`htr/research_status.py`'s active-method set -- a fine-tuned checkpoint is a future candidate model
version under the existing `loghi` pipeline family, not a new active method.
"""
