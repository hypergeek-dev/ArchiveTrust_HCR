"""Method overview ViewModel (Stage 11, brief's "User interface" -> method overview).

Framework-independent, like every other ViewModel in this package. Reads the three real
`HtrMethodAdapter` implementations directly -- `get_metadata()`, `get_capabilities()`,
`validate_environment()`, `health_check()` -- and the "Known limitations" section of each adapter's
own `README.md`. Nothing here is a hand-written summary of what a method does: if a capability flag
changes in the adapter, this surface changes with it.

**Why the READMEs.** `MethodMetadata` carries no limitations field (see `providers/htr_adapter.py`
-- it is deliberately a four-field identity record), and each adapter's README already has a
maintained, specific "Known limitations" section written by whoever implemented it. Parsing that
section is the honest way to surface real limitations rather than paraphrasing them into a second
copy that would immediately drift. `known_limitations` is empty -- never fabricated -- when an
adapter's README has no such section.

**Environment probing is opt-in.** `validate_environment()` and `health_check()` touch the
filesystem (and, for SATRN, look for an isolated interpreter); `method_rows()` therefore takes an
explicit `probe_environment` flag. With `probe_environment=False` (the default) the availability
fields are `None`, meaning "not probed" -- distinct from `False`, meaning "probed and unavailable".
That distinction is Constitution Article 18's discipline (an absence is a fact, not a zero) applied
to this surface.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.experiment.models import MethodRun
from archivetrust.presentation.display_names import capability_label, method_label
from archivetrust.providers.htr_adapter import HtrMethodAdapter

_PROVIDERS_ROOT = Path(__file__).resolve().parents[1] / "providers"

_README_BY_METHOD_ID = {
    "satrn": _PROVIDERS_ROOT / "satrn" / "README.md",
    "florence2_htr": _PROVIDERS_ROOT / "florence2_htr" / "README.md",
    "transkribus_swedish_lion_1": _PROVIDERS_ROOT / "transkribus" / "README.md",
}

_ADAPTER_VERSION_BY_METHOD_ID = {
    "satrn": "archivetrust.providers.satrn.adapter",
    "florence2_htr": "archivetrust.providers.florence2_htr.adapter",
    "transkribus_swedish_lion_1": "archivetrust.providers.transkribus.adapter",
}


class MethodCapabilityRow(BaseModel):
    """One capability flag, as a labeled true/false pair. Every flag is listed, including the false
    ones -- `MethodCapabilities` declares them all as required booleans precisely so an unsupported
    capability is visible rather than omitted (`providers/htr_adapter.py`'s module docstring)."""

    model_config = ConfigDict(frozen=True)

    field: str
    label: str
    supported: bool


class MethodRow(BaseModel):
    """Everything the method overview shows for one HTR method."""

    model_config = ConfigDict(frozen=True)

    method_id: str
    display_name: str
    vendor: str
    model_revision: str
    adapter_version: str | None
    execution_location: str
    """`"Local"` or `"External (manual import)"` -- derived from the adapter's own
    `local_execution_supported`/`external_upload_required` flags, never hardcoded per method."""
    is_local: bool
    capabilities: tuple[MethodCapabilityRow, ...]
    known_limitations: tuple[str, ...]
    """Verbatim bullets from the adapter README's "Known limitations" section. Empty when the
    README has no such section -- never filled with a generic placeholder."""
    environment_valid: bool | None = None
    """`None` = not probed this refresh; `True`/`False` = actually probed. Never conflated."""
    environment_messages: tuple[str, ...] = ()
    healthy: bool | None = None
    health_message: str | None = None
    device: str | None = None
    """Execution device last actually observed on a `MethodRun`'s Evidence, when run history was
    supplied. `None` when this method has never run here -- not defaulted to "CPU"."""
    last_success_at: str | None = None
    last_failure_at: str | None = None
    last_failure_reason: str | None = None
    total_runs: int = 0
    succeeded_runs: int = 0
    failed_runs: int = 0


def parse_known_limitations(readme_text: str) -> tuple[str, ...]:
    """Extracts the bullets under a README's `## Known limitations...` heading, joining wrapped
    continuation lines back into one bullet each. Stops at the next `##` heading. Returns `()` when
    no such heading exists -- an honest empty, not an invented limitation.
    """
    lines = readme_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(r"^##\s+Known limitations", line.strip(), flags=re.IGNORECASE):
            start = index + 1
            break
    if start is None:
        return ()

    bullets: list[str] = []
    current: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                bullets.append(" ".join(current))
            current = [stripped[2:].strip()]
        elif stripped and current:
            current.append(stripped)
        elif not stripped and current:
            bullets.append(" ".join(current))
            current = []
    if current:
        bullets.append(" ".join(current))
    return tuple(bullet for bullet in bullets if bullet)


def _known_limitations_for(method_id: str) -> tuple[str, ...]:
    path = _README_BY_METHOD_ID.get(method_id)
    if path is None or not path.exists():
        return ()
    return parse_known_limitations(path.read_text(encoding="utf-8"))


def _adapter_version_for(method_id: str) -> str | None:
    """Reads the adapter module's own `ADAPTER_VERSION` constant. `None` if the module cannot be
    imported (an optional dependency missing, say) -- never a guessed "1.0.0"."""
    module_name = _ADAPTER_VERSION_BY_METHOD_ID.get(method_id)
    if module_name is None:
        return None
    try:
        import importlib

        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - an unimportable adapter is a reportable absence, not a crash
        return None
    version = getattr(module, "ADAPTER_VERSION", None)
    return version if isinstance(version, str) else None


class MethodOverviewViewModel:
    """Projects the registered `HtrMethodAdapter`s into `MethodRow`s.

    Adapters are supplied by the caller (the composition root), not discovered here: this ViewModel
    must stay constructible in a test with a stub adapter, and must never be the thing that decides
    which methods exist.
    """

    def __init__(
        self,
        adapters: tuple[HtrMethodAdapter, ...],
        *,
        method_runs: tuple[MethodRun, ...] = (),
        run_devices: dict[str, str] | None = None,
        failure_reasons: dict[str, str] | None = None,
    ) -> None:
        self._adapters = adapters
        self._method_runs = method_runs
        self._run_devices = run_devices or {}
        """`method_run_id -> execution device` as recorded on that run's Evidence. Passed in rather
        than looked up here so this ViewModel never reaches into the telemetry store itself."""
        self._failure_reasons = failure_reasons or {}
        """`method_run_id -> FailureRecord.reason`."""

    def method_rows(self, *, probe_environment: bool = False) -> tuple[MethodRow, ...]:
        rows = [self._row(adapter, probe_environment=probe_environment) for adapter in self._adapters]
        return tuple(sorted(rows, key=lambda row: row.display_name))

    def _row(self, adapter: HtrMethodAdapter, *, probe_environment: bool) -> MethodRow:
        metadata = adapter.get_metadata()
        capabilities = adapter.get_capabilities()

        capability_rows = tuple(
            MethodCapabilityRow(
                field=field,
                label=capability_label(field),
                supported=bool(getattr(capabilities, field)),
            )
            for field in type(capabilities).model_fields
        )

        environment_valid: bool | None = None
        environment_messages: tuple[str, ...] = ()
        healthy: bool | None = None
        health_message: str | None = None
        if probe_environment:
            validation = adapter.validate_environment()
            environment_valid = validation.valid
            environment_messages = validation.messages
            health = adapter.health_check()
            healthy = health.healthy
            health_message = health.message

        runs = tuple(run for run in self._method_runs if run.method_id == metadata.method_id)
        succeeded = tuple(run for run in runs if run.outcome == "succeeded")
        failed = tuple(run for run in runs if run.outcome == "failed")

        last_success = max((r.completed_at or r.started_at for r in succeeded), default=None)
        last_failure_run = max(failed, key=lambda r: r.completed_at or r.started_at, default=None)

        device = None
        for run in sorted(runs, key=lambda r: r.started_at, reverse=True):
            if run.method_run_id in self._run_devices:
                device = self._run_devices[run.method_run_id]
                break

        return MethodRow(
            method_id=metadata.method_id,
            display_name=method_label(metadata.method_id)
            if metadata.method_id in _README_BY_METHOD_ID
            else metadata.method_name,
            vendor=metadata.vendor,
            model_revision=metadata.model_revision,
            adapter_version=_adapter_version_for(metadata.method_id),
            execution_location=(
                "Local" if capabilities.local_execution_supported else "External (manual import)"
            ),
            is_local=capabilities.local_execution_supported,
            capabilities=capability_rows,
            known_limitations=_known_limitations_for(metadata.method_id),
            environment_valid=environment_valid,
            environment_messages=environment_messages,
            healthy=healthy,
            health_message=health_message,
            device=device,
            last_success_at=last_success,
            last_failure_at=(
                (last_failure_run.completed_at or last_failure_run.started_at)
                if last_failure_run is not None
                else None
            ),
            last_failure_reason=(
                self._failure_reasons.get(last_failure_run.method_run_id)
                if last_failure_run is not None
                else None
            ),
            total_runs=len(runs),
            succeeded_runs=len(succeeded),
            failed_runs=len(failed),
        )
