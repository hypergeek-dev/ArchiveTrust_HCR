"""Version-pinning and environment-probe types for the Loghi pipeline (docs/methods/loghi.md).

**Why one `LoghiComponentVersions` object instead of a single `model_revision` string.** Loghi is not
one model -- it is three independently-versioned components (Laypa for layout analysis, Loghi Tooling
for line extraction/reading order, Loghi HTR for recognition), each with its own repository commit,
plus a separately-versioned model checkpoint and Docker image. `MethodMetadata.model_revision`
(`providers/htr_adapter.py`) is a single string, which would force collapsing all of that into one
opaque value; `LoghiAdapter.get_metadata()` instead renders a summary string from this richer object,
and the object itself is what `ReproducibilityManifest.software_environment` actually carries (brief:
"Do not represent the complete Loghi pipeline as one model revision if multiple independently
versioned components participate").

**Why no field has a default.** Every field here is a real pin a research run depends on. A default
would either be a fabricated value (a specific commit nobody actually chose) or a floating one
(`"latest"`/`"main"`), both of which the brief explicitly forbids in a research run. Callers that have
not yet pinned real values use `pinned_versions.UNPINNED_PLACEHOLDER` (its sentinel strings, not a
`None`/omitted field) so `validate_environment()` can detect and refuse them explicitly rather than a
missing field silently passing pydantic validation with `None`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class LoghiComponentVersions(BaseModel):
    """Every independently-versioned piece of one Loghi pipeline configuration. One instance of this
    is what "a Loghi pipeline version" means in this codebase -- never a single string."""

    model_config = ConfigDict(frozen=True)

    loghi_repo_commit: str
    """Pinned commit of the top-level `knaw-huc/loghi` repository (never `main`/`HEAD`)."""
    submodule_commits: dict[str, str]
    """`{submodule_path: pinned_commit}` for every Git submodule the top-level repo references at
    `loghi_repo_commit` -- recorded explicitly rather than assumed to follow the parent commit, since
    `git submodule update --remote` (which would float them) is exactly what the brief forbids."""
    laypa_commit: str
    loghi_tooling_commit: str
    loghi_htr_commit: str
    model_checkpoint_id: str
    """Identifies which trained HTR checkpoint this configuration uses (e.g. a Loghi-published Dutch
    model release name) -- distinct from the repository commits above, which version the *code*."""
    model_checkpoint_hash: str
    """Content hash of the checkpoint files actually used, so "the same checkpoint" is a verifiable
    fact rather than a trusted label."""
    docker_image_tag: str
    """The Loghi HTR recognition image (`loghi/docker.htr`) -- the one this adapter's inference/
    fine-tuning facade actually invokes. Named singular (not pluralized to all three pipeline images)
    to match this field's original scope; Laypa's and Loghi Tooling's images are the two fields below,
    added when this integration actually pulled and pinned all three real images rather than assuming
    only one mattered."""
    docker_image_digest: str | None
    """`sha256:...` digest, when the registry provides one -- more specific than a tag, which can be
    force-moved by its publisher. `None` only when a digest is genuinely unavailable, never omitted
    silently."""
    laypa_docker_image_tag: str | None = None
    laypa_docker_image_digest: str | None = None
    loghi_tooling_docker_image_tag: str | None = None
    loghi_tooling_docker_image_digest: str | None = None
    """Optional -- `None` for any pipeline that pins only the recognition image (e.g. a fine-tuning-
    only setup that never runs layout analysis/line extraction). Real, pinned values once the full
    three-container pipeline is installed, never placeholders left silently absent."""
    inference_script_version: str
    beam_width: int
    reading_order_settings: str
    language_detection_settings: str
    gpu_selection: str
    container_runtime_version: str

    def summary(self) -> str:
        """One-line rendering for `MethodMetadata.model_revision` and log lines -- a compressed
        pointer at this object, never a replacement for reading the full pinned fields (which is
        what `ReproducibilityManifest` carries in full). Every component is truncated to a fixed
        width -- including `model_checkpoint_id`/`docker_image_tag`, which are normally short but can
        be the long `PLACEHOLDER_SENTINEL` string before pinning; truncating keeps this summary a
        genuinely one-line pointer either way rather than repeating a long sentinel six times."""
        return (
            f"loghi@{self.loghi_repo_commit[:12]} "
            f"laypa@{self.laypa_commit[:12]} "
            f"tooling@{self.loghi_tooling_commit[:12]} "
            f"htr@{self.loghi_htr_commit[:12]} "
            f"checkpoint={self.model_checkpoint_id[:24]} "
            f"image={self.docker_image_tag[:24]}"
        )

    def is_placeholder(self) -> bool:
        """True when any field still carries the `pinned_versions.UNPINNED_PLACEHOLDER` sentinel --
        the check `validate_environment()` runs before ever reporting this configuration as usable."""
        from archivetrust.providers.loghi.pinned_versions import PLACEHOLDER_SENTINEL

        return any(
            (isinstance(value, str) and value == PLACEHOLDER_SENTINEL)
            or (isinstance(value, dict) and any(v == PLACEHOLDER_SENTINEL for v in value.values()))
            for value in (
                self.loghi_repo_commit,
                self.submodule_commits,
                self.laypa_commit,
                self.loghi_tooling_commit,
                self.loghi_htr_commit,
                self.model_checkpoint_id,
                self.model_checkpoint_hash,
                self.docker_image_tag,
                self.inference_script_version,
            )
        )


class LoghiExecutionMode(str, Enum):
    """The one explicitly-documented execution boundary this adapter supports -- never a claim of
    native Windows execution, since Loghi's own tooling assumes Bash/Linux paths/NVIDIA container
    tooling (brief: "Do not pretend native Windows execution is supported")."""

    DOCKER_LINUX = "docker_linux"
    DOCKER_WSL2 = "docker_wsl2"
    NATIVE_LINUX = "native_linux"


class LoghiEnvironmentReport(BaseModel):
    """What `environment.probe_loghi_environment()` actually observed on this machine -- read-only
    probes, never a container execution. Every field is `None` when that specific probe could not
    determine an answer (command not found, distro absent, etc.), never a guessed value."""

    model_config = ConfigDict(frozen=True)

    host_os: str
    docker_cli_present: bool
    docker_version: str | None
    wsl_present: bool
    wsl_distros: tuple[str, ...]
    """Every WSL distro name `wsl.exe -l -v` reported, verbatim -- not filtered to "the one we want",
    so a caller can see exactly what exists, including a stopped `docker-desktop` distro."""
    linux_distribution: str | None
    nvidia_toolkit_version: str | None
    cuda_visible_devices: str | None
    execution_mode: LoghiExecutionMode | None
    """The execution mode this report supports, if any -- `None` when no supported mode is currently
    available (e.g. Docker CLI present but no distro running), which is a real, reportable state, not
    an error."""
