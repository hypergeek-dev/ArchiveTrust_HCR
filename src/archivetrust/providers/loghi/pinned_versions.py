"""The Loghi pinned-version configuration this ArchiveTrust checkout is set up to use.

**Updated 2026-08-01: real values, resolved from an actually-installed environment.** Every field
below was read from a real, local artifact this session actually produced -- never typed from memory,
never guessed from upstream documentation. Provenance for each:

- `loghi_repo_commit`/`submodule_commits`/`laypa_commit`/`loghi_tooling_commit`/`loghi_htr_commit`:
  `git rev-parse HEAD` and `git submodule status`, run against a real `git clone --recurse-submodules
  https://github.com/knaw-huc/loghi.git` into `.loghi-upstream/` (gitignored, repo-root-relative,
  mirroring `.venv-satrn`'s isolated-external-dependency convention).
- `model_checkpoint_id`/`model_checkpoint_hash`: `generic-2023-02-15`, the real pretrained Loghi-HTR
  checkpoint (`model.keras`, 372,591,597 bytes) downloaded from the project's published SURFdrive share
  (`surfdrive.surf.nl/files/index.php/s/YA8HJuukIUKznSP`, confirmed via that share's public WebDAV
  listing) and hashed with sha256 -- see `training/loghi-swedish-v1/manifests/
  pretrained_checkpoint_hashes.json` for the hashing run's full output (all four downloaded files,
  including the Laypa `baseline2` layout model pinned alongside it for completeness even though HTR
  fine-tuning alone does not exercise Laypa).
- `docker_image_tag`/`docker_image_digest` (+ the Laypa/Loghi-Tooling pairs): `docker pull loghi/
  docker.htr`, `loghi/docker.laypa`, `loghi/docker.loghi-tooling` (real Docker Hub images -- not GHCR,
  confirmed via the upstream README), each resolved via `docker inspect --format '{{index
  .RepoDigests 0}}'` after pulling.
- GPU passthrough was verified empirically, not assumed: `docker run --rm --gpus all loghi/docker.htr
  python3 -c "import tensorflow as tf; ..."` printed a real `PhysicalDevice(name='/physical_device:
  GPU:0', ...)` -- Docker Desktop's WSL2 integration exposes the RTX 3070 to containers with no
  `nvidia-ctk` install needed on the Windows host, confirming what `docs/loghi-integration-audit.md`
  §13 only expected.
- `container_runtime_version`: `docker --version` on this host (`Docker 29.6.1`); the container image
  itself reports `CUDA Version 12.8.0` and `TensorFlow 2.20.0-dev0+selfbuilt` (both printed by the
  verification run above, not looked up from documentation).
- `reading_order_settings`/`language_detection_settings`: honestly **not independently configured** in
  this pass -- both are Loghi Tooling inference-pipeline stages, and this integration's first real use
  of the environment is fine-tuning `loghi-htr`'s recognition model directly on pre-cropped line images
  (`F:\\huggingface_dataset`), which never invokes Loghi Tooling. Recorded as such, not filled with an
  invented Loghi Tooling configuration nobody chose.
- `beam_width=1`: the real value read from `generic-2023-02-15`'s own shipped training configuration
  (`.loghi-upstream/pretrained-models/loghi-htr/generic-2023-02-15/file.txt`, a JSON dump of the exact
  config used to produce that checkpoint) -- not a default this integration invented.

`PLACEHOLDER_SENTINEL` and `is_placeholder()` are unchanged and still load-bearing: any *future* pin
change (a different checkpoint, a repinned commit) goes back through this same discipline -- resolved
from a real installed artifact, never floating, never invented -- and a caller who constructs
`LoghiComponentVersions` with any field still equal to the sentinel is still correctly refused by
`validate_environment()`.
"""

from __future__ import annotations

from archivetrust.providers.loghi.models import LoghiComponentVersions

PLACEHOLDER_SENTINEL = "UNPINNED -- set before any research run, see docs/methods/loghi.md"

CURRENT_PINNED_VERSIONS = LoghiComponentVersions(
    loghi_repo_commit="90305b91b793ff81e99dbdc486881aace7650035",
    submodule_commits={
        "laypa": "d0ed632ab6ca4e595d7224f0bc046d337b1bcb0e",
        "loghi-htr": "c24c745e9aeb287bedeb6b4edee670ae5c81a3cd",
        "loghi-tooling": "71217767ed017a0dea6b66b681d8fa4a5729b634",
        "prima-core-libs": "6328dd79914876c7a6f18c8f82d12db506ac60f9",
    },
    laypa_commit="d0ed632ab6ca4e595d7224f0bc046d337b1bcb0e",
    loghi_tooling_commit="71217767ed017a0dea6b66b681d8fa4a5729b634",
    loghi_htr_commit="c24c745e9aeb287bedeb6b4edee670ae5c81a3cd",
    model_checkpoint_id="generic-2023-02-15",
    model_checkpoint_hash="0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95",
    docker_image_tag="loghi/docker.htr:latest",
    docker_image_digest="sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8",
    laypa_docker_image_tag="loghi/docker.laypa:latest",
    laypa_docker_image_digest="sha256:5933040e8f58abacf2d35f5cb49e3bb2bc0c500c57c77b9ab646cdd7c6fe9c83",
    loghi_tooling_docker_image_tag="loghi/docker.loghi-tooling:latest",
    loghi_tooling_docker_image_digest="sha256:828d011d7faa30d3ad8b338cc2759d4378d8fd80c7c948ce701d709091fb8e13",
    inference_script_version="loghi-htr@c24c745e9aeb287bedeb6b4edee670ae5c81a3cd:src/main.py",
    beam_width=1,
    reading_order_settings="not_applicable_to_htr_only_finetuning",
    language_detection_settings="not_applicable_to_htr_only_finetuning",
    gpu_selection="0",
    container_runtime_version="Docker 29.6.1 (host); CUDA 12.8.0, TensorFlow 2.20.0-dev0+selfbuilt (container, loghi/docker.htr)",
)
"""The real, currently-pinned Loghi environment this checkout uses. `is_placeholder()` is `False` for
this instance -- `validate_environment()` reports ready (pins-wise) once Docker/WSL2 execution mode
also resolves. See this module's docstring for exactly how each field was obtained."""
