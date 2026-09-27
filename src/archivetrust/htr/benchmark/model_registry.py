"""Frozen identities of the two benchmarked recognizers, and their decoding profiles.

Nothing here is looked up at run time from "latest": every model file is pinned by SHA-256 or by
repository revision, and every decoding parameter is spelled out, so the decoding a model ran with is
never an inherited default nobody chose. A run refuses a model whose files do not hash to these
values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from archivetrust.htr.benchmark.contract import sha256_file
from archivetrust.htr.benchmark.layout import REPO_ROOT
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

LOGHI_CHECKPOINT_ENV_VAR = "ARCHIVETRUST_LOGHI_CHECKPOINT"


class ModelIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodingProfile:
    profile_id: str
    parameters: dict
    rationale: str


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    family: str
    description: str
    pinned: dict
    profiles: dict[str, DecodingProfile]
    default_profile: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def profile(self, profile_id: str | None) -> DecodingProfile:
        key = profile_id or self.default_profile
        if key not in self.profiles:
            raise ModelIdentityError(f"{self.model_id}: unknown decoding profile {key!r}; known {sorted(self.profiles)}")
        return self.profiles[key]

    def record(self, profile_id: str | None = None) -> dict:
        profile = self.profile(profile_id)
        return {"model_id": self.model_id, "family": self.family, "pinned": self.pinned,
                "decoding_profile": profile.profile_id, "decoding": profile.parameters}


LOGHI = ModelIdentity(
    model_id="loghi-swedish-scratch-exp2-epoch7",
    family="loghi-htr",
    description="Scratch-trained Loghi-HTR (VGSL 'recommended'), Experiment 2 epoch 7 best_val. Frozen; not to be retrained.",
    pinned={
        "checkpoint_relative_path": "training/experiment-2-epoch7-from-epoch6-checkpoint-20260813T124955Z/run-state/"
                                    "epoch_output/epoch_1/recommended/best_val",
        "file_sha256": {
            "model.keras": "7a11df10b943330fe93c7fbf2ddb0584a75539ecb6b1da06dcaa3a5c39f17572",
            "config.json": "54b6f6856a2441f1e6fb0e22521d06d290977df296be2b029d7eede0276508c7",
            "tokenizer.json": "a6cb9dfe416c36d0d04caa35f8e3381606ea0e473acab56e18d5ec3be825d01e",
        },
        "container_image": f"{CURRENT_PINNED_VERSIONS.docker_image_tag}@{CURRENT_PINNED_VERSIONS.docker_image_digest}",
        "loghi_htr_commit": CURRENT_PINNED_VERSIONS.loghi_htr_commit,
        "input": "grayscale, height 64 (VGSL input None,None,64,1); resizing done by the container's own loader",
        "reference_validation": {"cer": 0.0998, "wer": 0.3273, "lines": 1000,
                                 "decoding": "beam_width 10, greedy false (lap-evaluation/lap1_best_val/config.json)",
                                 "caveat": "in-distribution validation split; not comparable to an external benchmark"},
        "backup_release": "https://github.com/hypergeek-dev/ArchiveTrust_HCR/releases/tag/"
                          "model-loghi-swedish-scratch-exp2-epoch7",
    },
    profiles={
        "validated": DecodingProfile("validated", {"beam_width": 10, "greedy": False, "batch_size": 16, "seed": 42},
                                     "The exact decoding that produced val CER 0.0998 (Loghi CLI default beam_width=10). "
                                     "The checkpoint's own config.json says beam_width 1 but is only read with --config_file, "
                                     "which the evaluation did not pass."),
        "greedy": DecodingProfile("greedy", {"beam_width": 1, "greedy": True, "batch_size": 16, "seed": 42},
                                  "Diagnostic: plain greedy CTC decoding."),
    },
    default_profile="validated",
    notes=("No language model / corpus file is used (corpus_file null).",
           "Loghi timing is per batch; per-line durations are batch-amortized."),
)

LION = ModelIdentity(
    model_id="riksarkivet-swedish-lion-libre",
    family="trocr",
    description="Riksarkivet 'Swedish Lion Libre' TrOCR line recognizer, pinned revision.",
    pinned={
        "hf_model": "Riksarkivet/trocr-base-handwritten-hist-swe-2",
        "hf_model_revision": "aa79fcb1850bf3155ebc442570d6c6bfc0ac8100",
        "hf_processor": "microsoft/trocr-base-handwritten",
        "hf_processor_revision": "eaacaf452b06415df8f10bb6fad3a4c11e609406",
        "processor_equivalence": "image processor, vocab.json and merges.txt verified identical to the model repo's own",
        "input": "RGB, TrOCR ViT processor 384x384, mean/std 0.5",
        "training_data_note": "trained on the same public Riksarkivet HF line collections as the Loghi training corpus",
    },
    profiles={
        "generation_config": DecodingProfile(
            "generation_config",
            {"num_beams": 4, "no_repeat_ngram_size": 3, "length_penalty": 2.0, "early_stopping": True, "max_length": 256,
             "do_sample": False},
            "What the model card's `model.generate(pixel_values)` does: the pinned generation_config.json, spelled out."),
        "htrflow": DecodingProfile(
            "htrflow",
            {"num_beams": 1, "no_repeat_ngram_size": 3, "max_length": 256, "do_sample": False},
            "num_beams=1 as in the htrflow pipeline.yaml the existing adapter followed; generate(num_beams=1) "
            "inherits the rest of generation_config.json (no_repeat_ngram_size=3, max_length=256), spelled out here. "
            "This is exactly what the existing ArchiveTrust Lion adapter ran."),
        "greedy": DecodingProfile(
            "greedy", {"num_beams": 1, "no_repeat_ngram_size": 0, "max_length": 256, "do_sample": False},
            "Diagnostic: plain greedy, with the inherited repetition ban explicitly switched off."),
    },
    default_profile="generation_config",
    notes=("no_repeat_ngram_size=3 forbids any repeated 3-token BPE sequence within a line; lines with genuine repetition "
           "can be damaged by it -- compare against the 'greedy' profile.",
           "TrOCR exposes no calibrated per-line confidence; none is recorded."),
)

MODELS: dict[str, ModelIdentity] = {LOGHI.model_id: LOGHI, LION.model_id: LION}
SHORT_NAMES = {"loghi": LOGHI.model_id, "lion": LION.model_id}


def resolve_model(name: str) -> ModelIdentity:
    key = SHORT_NAMES.get(name, name)
    if key not in MODELS:
        raise ModelIdentityError(f"unknown model {name!r}; known {sorted(SHORT_NAMES)} / {sorted(MODELS)}")
    return MODELS[key]


def loghi_checkpoint_dir() -> Path:
    override = os.environ.get(LOGHI_CHECKPOINT_ENV_VAR)
    return Path(override) if override else REPO_ROOT / LOGHI.pinned["checkpoint_relative_path"]


def verify_loghi_checkpoint(checkpoint_dir: Path | None = None) -> dict:
    """Hashes the three checkpoint files; raises unless all match the pinned identity."""
    directory = Path(checkpoint_dir) if checkpoint_dir else loghi_checkpoint_dir()
    observed: dict[str, str | None] = {}
    for name in LOGHI.pinned["file_sha256"]:
        path = directory / name
        observed[name] = sha256_file(path) if path.is_file() else None
    mismatched = {n: h for n, h in observed.items() if h != LOGHI.pinned["file_sha256"][n]}
    if mismatched:
        raise ModelIdentityError(f"Loghi checkpoint at {directory} does not match {LOGHI.model_id}: "
                                 f"{ {n: (h or 'missing') for n, h in mismatched.items()} }")
    return {"checkpoint_dir": str(directory), "file_sha256": observed, "verified": True}


def load_loghi_charset(checkpoint_dir: Path | None = None) -> set[str]:
    """The characters the Loghi model can emit (tokenizer.json `{index: token}` minus [PAD]/[UNK])."""
    directory = Path(checkpoint_dir) if checkpoint_dir else loghi_checkpoint_dir()
    tokens = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    return {t for t in tokens.values() if not (t.startswith("[") and t.endswith("]") and len(t) > 1)}
