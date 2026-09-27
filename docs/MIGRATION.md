# Migrating ArchiveTrust to a New Machine

Status: written 2026-08-07 against the actual state of this checkout (`D:\ArchiveTrust_HCR` on
the source machine, Windows 11, RTX 3070). Follow this top to bottom on the **new** machine.

## 0. What kind of move this is

This checkout has two very different kinds of content:

1. **Everything tracked in git** — code, docs, committed corpora (`corpora/`), configs, manifests.
   This is already portable: it lives on GitHub (`hypergeek-dev/ArchiveTrust_HCR`, confirmed
   fully pushed — `main` is 0 commits ahead/behind `origin/main` as of this writing) and a plain
   `git clone` reproduces it exactly, byte for byte.
2. **Large, gitignored, locally-assembled content** — Python virtual environments, a Loghi
   upstream clone + checkpoints, raw datasets, and 150+ GB of training run output. None of this
   is in git (by design — see the comments in `.gitignore`). Some of it is cheap to regenerate
   from scratch on the new machine; some of it is real, expensive-to-reproduce work (actual
   trained checkpoints, hours of GPU compute, a manually-provided dataset with no confirmed
   re-download source) that you decided to carry over physically via external drive instead.

This guide covers both: what to `git clone`, what to copy by hand (and to exactly which path),
and what to just let regenerate.

## 1. Before you unplug: sanity checks on the source machine

```powershell
cd D:\ArchiveTrust_HCR
git status                     # should be clean
git fetch origin
git rev-list --left-right --count origin/main...main   # should print "0  0"
```

If `git status` isn't clean or the counts aren't `0 0`, commit/push first — anything not pushed
does not survive a plain clone on the new machine.

## 2. What to copy via external drive, and where it goes

Copy these directories **as-is**, preserving the relative layout, to the same path on the new
machine: `D:\ArchiveTrust_HCR\<dir>` (see §5 on why the exact path matters). Sizes are as measured
on the source machine on 2026-08-07 — budget accordingly on the drive and the destination disk.

| Source-relative path | Size | What it is |
|---|---|---|
| `dataset-rgb/` | ~14 GB | Primary Swedish RGB-normalized dataset |
| `dataset-dutch-rgb/` | ~3.5 GB | Dutch corpus, extracted from a manually-provided archive with no confirmed re-download source (`docs/experiments/lion-loghi-comparison/dataset-provenance-dutch.md`) — **treat this one as irreplaceable, back it up separately too** |
| `training/` | ~156 GB | All experiment runs: configs, manifests, prepared shards, checkpoints, reports |
| `.loghi-upstream/` | ~3.5 GB | Pinned `knaw-huc/loghi` clone + downloaded pretrained checkpoints (optional — see §4, easy to regenerate instead) |
| `docs/experiments/technical-reliability-screening/**/{blobs,crops,export-packages}/` | ~965 MB | Derived reliability-screening artifacts (gitignored subpaths inside an otherwise-tracked `docs/` tree) |
| `.env` | 46 bytes | Contains `HF_TOKEN` — copy directly rather than retyping, since it's a local file transfer under your control |

Everything else that's gitignored (`.venv/`, `.venv-satrn/`, `.pytest_cache/`, `.history/`,
`test-artifacts/`, `archivetrust_data/`) is either regenerated in §3/§4 or purely transient — don't
bother copying it. `archivetrust_data/` in particular is an empty, app-managed runtime scaffold
(cache/config/identity/logs/models/plugins/telemetry/workspaces) that gets recreated as needed.

Two empty, oddly-named directories exist on the source machine — `tmp_probe;C` and
`training\_experiment_2_preflight_scratch;C` — both 0 bytes, almost certainly an accidental
artifact of a stray `;C:\...`-shaped argument at some point. Skip them; they carry nothing.

## 3. New machine: clone and install

```powershell
git clone https://github.com/hypergeek-dev/ArchiveTrust_HCR.git D:\ArchiveTrust_HCR
cd D:\ArchiveTrust_HCR
```

Run the existing preflight checker first — it verifies Python version, Git, Docker Desktop, the
Docker daemon, Docker Compose, WSL2, and NVIDIA GPU/driver presence before you invest time in the
rest:

```powershell
python -m archivetrust.bootstrap.cli preflight
```

(`docs/WINDOWS_BOOTSTRAP.md` has the full detail on what this does and does not check today —
venv creation and dependency install are not yet automated by it, hence the manual steps below.)

### 3.1 Main environment (`.venv`)

Source machine has Python 3.13.13, torch 2.13.0+cu130 (CUDA 13.0), on an RTX 3070 (driver
610.88). If the new machine's GPU/driver differs, the `--index-url` below may need a different
CUDA tag — check what your installed driver supports before forcing this exact one.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,gui,transformers,watch,dashboard]"
```

`docs/setup/requirements-freeze-main.txt` is the exact `pip freeze` of the working source-machine
environment — use it as a reference if anything resolves to a different version than expected
(it is not meant to be installed directly with `pip install -r`, see the file's header).

### 3.2 SATRN isolated environment (`.venv-satrn`)

Needed only for the SATRN provider. Full ordered install sequence (order matters — several
packages install with `--no-deps`) is in
[`src/archivetrust/providers/satrn/README.md`](../src/archivetrust/providers/satrn/README.md#setting-up-venv-satrn-one-time-windows).
`docs/setup/requirements-freeze-satrn.txt` is the reference freeze (Python 3.10.11) for the same
"does this match what worked before" purpose as §3.1.

### 3.3 `.env`

If you copied `.env` from the source machine in §2, nothing else to do. Otherwise, recreate it:

```
HF_TOKEN=<your Hugging Face token>
```

## 4. Regenerating what you didn't copy

- **`.loghi-upstream/`**, if you skipped copying it: re-clone at the exact pinned commit recorded
  in `src/archivetrust/providers/loghi/pinned_versions.py`
  (`CURRENT_PINNED_VERSIONS.loghi_repo_commit = "90305b91b793ff81e99dbdc486881aace7650035"`), then
  `git checkout` that commit and update submodules. The pretrained checkpoint
  (`generic-2023-02-15`, sha256 recorded in the same file) comes from the project's published
  SURFdrive share — see that file's module docstring for the exact URL and provenance.
- **Docker/vLLM providers** (`paddleocr-vl`, `surya`): not stored on disk under this repo at all —
  `.\scripts\providers.ps1 start <name>` pulls the pinned image and downloads the model checkpoint
  (~1.8 GB / ~1.3 GB) on first run, once Docker Desktop + WSL2 + GPU passthrough are working
  (confirmed by the preflight check in §3).
- **Loghi's own Docker images** (`loghi/docker.htr`, `docker.laypa`, `docker.loghi-tooling`): pull
  the exact pinned tags/digests from `pinned_versions.py` with `docker pull`, don't take `latest`
  at face value even though that's the tag name — the digest is what's actually pinned.

## 5. Why the path has to match

Nothing in `src/` or `scripts/` hardcodes `D:\ArchiveTrust_HCR` — only historical, non-executing
records do (`training/**/LAUNCH_COMMANDS.txt`, `training/**/lap_evaluation.md`, etc. — operator
reference text, never auto-run). Functionally you *can* install to a different drive/path. But
keeping `D:\ArchiveTrust_HCR` on the new machine too means those existing historical records stay
literally accurate instead of just close, so it's the path of least surprise — worth doing unless
you have a specific reason not to.

## 6. Verify

```powershell
cd D:\ArchiveTrust_HCR
.venv\Scripts\python.exe -m pytest -q
python -m archivetrust.bootstrap.cli status
.\scripts\providers.ps1 status
```

Full suite was 1216 passed / 2 skipped on the source machine (`docs/WINDOWS_BOOTSTRAP.md`) — a
materially lower pass count on the new machine means something in §3 didn't reproduce cleanly,
not that the new machine is "different" in some acceptable way.

If you copied `training/`, spot-check one run's state is intact:

```powershell
.venv\Scripts\python.exe -m archivetrust.htr.training.full_run status --run "D:\ArchiveTrust_HCR\training\<some-run-dir>"
```
