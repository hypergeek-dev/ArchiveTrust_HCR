from __future__ import annotations

import pytest

from archivetrust.runtime.prompt_management import (
    PromptLibrary,
    PromptTemplate,
    UnknownPromptTemplateError,
)


def test_add_and_get_by_name_and_version() -> None:
    library = PromptLibrary()
    library.add(PromptTemplate(name="default", version=1, provider_id="qwen2.5-vl", text="Describe."))
    assert library.get("default", 1).text == "Describe."


def test_latest_returns_highest_version() -> None:
    library = PromptLibrary()
    library.add(PromptTemplate(name="default", version=1, provider_id=None, text="v1"))
    library.add(PromptTemplate(name="default", version=3, provider_id=None, text="v3"))
    library.add(PromptTemplate(name="default", version=2, provider_id=None, text="v2"))
    assert library.latest("default").text == "v3"


def test_unknown_template_raises() -> None:
    library = PromptLibrary()
    with pytest.raises(UnknownPromptTemplateError):
        library.get("nonexistent", 1)


def test_for_provider_includes_provider_specific_and_provider_agnostic() -> None:
    library = PromptLibrary()
    library.add(PromptTemplate(name="qwen-only", version=1, provider_id="qwen2.5-vl", text="a"))
    library.add(PromptTemplate(name="glm-only", version=1, provider_id="glm-ocr", text="b"))
    library.add(PromptTemplate(name="shared", version=1, provider_id=None, text="c"))
    names = {t.name for t in library.for_provider("qwen2.5-vl")}
    assert names == {"qwen-only", "shared"}


def test_qualified_name_format() -> None:
    template = PromptTemplate(name="default", version=2, provider_id=None, text="x")
    assert template.qualified_name == "default@2"


def test_save_and_load_round_trip(tmp_path) -> None:
    library = PromptLibrary()
    library.add(PromptTemplate(name="default", version=1, provider_id="qwen2.5-vl", text="Describe the page."))
    directory = tmp_path / "prompts"
    library.save_to_directory(directory)

    reloaded = PromptLibrary.load_from_directory(directory)
    assert reloaded.get("default", 1).text == "Describe the page."


def test_load_from_missing_directory_returns_empty_library(tmp_path) -> None:
    library = PromptLibrary.load_from_directory(tmp_path / "does-not-exist")
    assert library.all() == ()
