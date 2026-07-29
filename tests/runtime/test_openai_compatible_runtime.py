from __future__ import annotations

from archivetrust.runtime.contracts import InferenceRequest
from archivetrust.runtime.openai_compatible_runtime import OpenAICompatibleRuntime

from tests.runtime._fakes import FakeHttpTransport


def _runtime(transport, api_key=None) -> OpenAICompatibleRuntime:
    return OpenAICompatibleRuntime(
        base_url="https://enterprise.example/v1",
        model_id="glm-ocr",
        api_key=api_key,
        transport=transport,
    )


def test_infer_posts_chat_completion_and_parses_response() -> None:
    transport = FakeHttpTransport(
        response={"model": "glm-ocr-2024", "choices": [{"message": {"content": '{"page_number":1}'}}]}
    )
    runtime = _runtime(transport)
    result = runtime.infer(InferenceRequest(page_image_ref="http://x/page1.png", prompt="describe"))
    assert result.raw_text == '{"page_number":1}'
    assert result.model_version == "glm-ocr-2024"
    assert result.device_used == "remote"


def test_api_key_becomes_bearer_header() -> None:
    transport = FakeHttpTransport(response={"choices": [{"message": {"content": "x"}}]})
    runtime = _runtime(transport, api_key="secret-token")
    runtime.infer(InferenceRequest(page_image_ref="ref", prompt="p"))
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_no_api_key_omits_authorization_header() -> None:
    transport = FakeHttpTransport(response={"choices": [{"message": {"content": "x"}}]})
    runtime = _runtime(transport, api_key=None)
    runtime.infer(InferenceRequest(page_image_ref="ref", prompt="p"))
    assert "Authorization" not in transport.calls[0]["headers"]


def test_warm_up_is_idempotent() -> None:
    transport = FakeHttpTransport(response={})
    runtime = _runtime(transport)
    runtime.warm_up()
    runtime.warm_up()
    assert len(transport.calls) == 1


def test_different_runtime_kind_from_transformers() -> None:
    transport = FakeHttpTransport(response={})
    runtime = _runtime(transport)
    assert runtime.runtime_kind == "openai_compatible"
