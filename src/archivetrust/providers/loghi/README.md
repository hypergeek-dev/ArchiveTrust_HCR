# Loghi HTR adapter

`method_id = "loghi"`. Page-level, containerized, multi-component pipeline: **Laypa** (layout
analysis) → **Loghi Tooling** (line extraction, reading order) → **Loghi HTR** (recognition) →
PAGE XML. See `docs/methods/loghi.md` for the full integration writeup, `docs/loghi-integration-audit.md`
for how this fits the rest of the codebase, and `providers/htr_capability_matrix.py`'s generated
capability table for this method's flags as machine-verified fact.

## Execution model

Runs through Docker — either **Docker on Linux**, **Docker through WSL2**, or **native Linux**
(`environment.py::probe_loghi_environment`). Native Windows execution is not supported and is never
claimed: Loghi's own tooling assumes Bash, Linux paths and NVIDIA container tooling.
`validate_environment()` reports exactly which mode (if any) is available on the current host — see
its output for a live read of this machine's state.

## Version pinning

Every field on `LoghiComponentVersions` (`models.py`) must be an explicit, real pin before a research
run — repository commits (top-level `loghi`, `laypa`, `loghi-tooling`, `loghi-htr`, every Git
submodule), model checkpoint id/hash, Docker image tag/digest, inference-script version, beam width,
reading-order/language-detection settings, GPU selection, container runtime version. `pinned_versions.
CURRENT_PINNED_VERSIONS` ships with `PLACEHOLDER_SENTINEL` values (never a fabricated hash, never
`"latest"`/`"main"`) until a real environment is set up — `validate_environment()` refuses to report
this adapter ready while any field is still a placeholder.

## Install / Setup

1. Install Docker Desktop (with WSL2 backend on Windows) or a native Docker install on Linux.
2. Clone `knaw-huc/loghi` and its submodules (`laypa`, `loghi-htr`, `loghi-tooling`) at chosen, real
   commits — never `git submodule update --remote`.
3. Pull or build the Docker image at the chosen tag/digest; download and hash the model checkpoint.
4. Update `providers/loghi/pinned_versions.py::CURRENT_PINNED_VERSIONS` with the real values.
5. Re-run `validate_environment()` — it should report `valid=True` with the resolved execution mode.

## Known limitations

- Line-level input is not supported (`line_level_supported=False`) even though Loghi internally cuts
  lines: ArchiveTrust cannot yet intentionally hand Loghi's recognition stage a single line crop and
  get a traceable result back under a supported workflow.
- No real Loghi environment has been installed as part of this integration — `validate_environment()`
  will report `valid=False` until Docker/WSL2 is actually set up and `pinned_versions.py` carries real
  pins, not placeholders.
- The exact `docker run` entrypoint/argument names this adapter's facade constructs
  (`facade.py::_build_docker_argv`) are this integration's placeholder convention, not yet confirmed
  against a real pulled image's documented usage — confirm and adjust before the first real run.
- Confidence values are only populated when Loghi's own PAGE XML output states a `conf` attribute;
  many configurations may not emit one, in which case this is honestly `None`.
- The Docker/subprocess-argv construction is tested for structure (mount syntax, deterministic
  argument order, Windows→WSL path translation) but not against a real container in this repository's
  test suite — see `tests/providers/loghi/` for what is and is not exercised.
