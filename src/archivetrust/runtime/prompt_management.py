"""Prompt Management (Part 10): named, versioned, provider-specific prompt templates as
configurable assets, never requiring recompilation.

`PromptLibrary` is a small in-memory (optionally file-backed, `load_from_directory`) collection —
prompt templates are data, so they belong in `config/` (`DeploymentLayout.config_dir`), not in a
provider's source file. `providers.qwen_vl.adapter` still owns its own default system prompt for
backward compatibility (unchanged, per this milestone's non-goals); a runtime-backed provider
(`providers/qwen_vl/runtime_backend.py`) can source its prompt from a `PromptLibrary` instead,
recording which named/versioned template it used in the resulting Evidence's provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PromptTemplate(BaseModel):
    """One named, versioned prompt. `provider_id` scopes a template to the provider it was written
    for (e.g. Qwen's item-schema instructions make no sense for GLM-OCR) — `None` means
    provider-agnostic.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    provider_id: str | None
    text: str

    @property
    def qualified_name(self) -> str:
        """A stable identifier for telemetry provenance (Part 13): `"name@version"`."""
        return f"{self.name}@{self.version}"


class UnknownPromptTemplateError(KeyError):
    pass


class PromptLibrary:
    """A named collection of `PromptTemplate`s, keyed by `(name, version)`. Adding a new template or
    a new version never requires recompilation — this class holds only data.
    """

    def __init__(self) -> None:
        self._templates: dict[tuple[str, int], PromptTemplate] = {}

    def add(self, template: PromptTemplate) -> None:
        self._templates[(template.name, template.version)] = template

    def get(self, name: str, version: int) -> PromptTemplate:
        try:
            return self._templates[(name, version)]
        except KeyError:
            raise UnknownPromptTemplateError(f"{name}@{version}") from None

    def latest(self, name: str) -> PromptTemplate:
        """The highest version registered under `name`."""
        matches = [t for (n, _v), t in self._templates.items() if n == name]
        if not matches:
            raise UnknownPromptTemplateError(name)
        return max(matches, key=lambda t: t.version)

    def for_provider(self, provider_id: str) -> tuple[PromptTemplate, ...]:
        return tuple(
            t for t in self._templates.values() if t.provider_id in (provider_id, None)
        )

    def all(self) -> tuple[PromptTemplate, ...]:
        return tuple(self._templates.values())

    def save_to_directory(self, directory: Path) -> None:
        """Persists every template as one JSON file under `directory` (typically
        `DeploymentLayout.config_dir / "prompts"`) — plain data, human-readable, editable without
        recompilation."""
        directory.mkdir(parents=True, exist_ok=True)
        for template in self._templates.values():
            path = directory / f"{template.name}.v{template.version}.json"
            path.write_text(template.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_from_directory(cls, directory: Path) -> "PromptLibrary":
        library = cls()
        if not directory.exists():
            return library
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            library.add(PromptTemplate(**data))
        return library
