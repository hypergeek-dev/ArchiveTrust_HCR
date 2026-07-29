from __future__ import annotations

from archivetrust.providers.htr_adapter import (
    EnvironmentValidation,
    HealthCheckResult,
    HtrMethodAdapter,
    MethodCapabilities,
    MethodMetadata,
    RecognitionInput,
    RecognitionResult,
)


class _FakeHtrMethodAdapter:
    """Minimal fake satisfying the `HtrMethodAdapter` Protocol -- deliberately declares at least
    one capability as explicitly False (external_upload_required), proving unsupported
    capabilities are represented, not omitted."""

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id="method_fake",
            method_name="Fake Recognizer",
            vendor="test-fixture",
            model_revision="v0",
        )

    def get_capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            confidence_supported=True,
            geometry_supported=False,
            line_level_supported=True,
            page_level_supported=False,
            local_execution_supported=True,
            external_upload_required=False,
        )

    def validate_environment(self) -> EnvironmentValidation:
        return EnvironmentValidation(valid=True)

    def recognize(self, input: RecognitionInput) -> RecognitionResult:
        return RecognitionResult(text="hello", confidence=0.9, raw_response={"engine": "fake"})

    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True)


def test_fake_adapter_satisfies_the_protocol():
    fake = _FakeHtrMethodAdapter()
    assert isinstance(fake, HtrMethodAdapter)


def test_capabilities_represent_unsupported_capability_explicitly():
    capabilities = _FakeHtrMethodAdapter().get_capabilities()
    # geometry_supported is explicitly False -- present as real data, not an omitted field.
    assert capabilities.geometry_supported is False
    assert "geometry_supported" in type(capabilities).model_fields
    assert capabilities.model_dump()["geometry_supported"] is False


def test_recognition_result_preserves_provider_specific_raw_data():
    result = _FakeHtrMethodAdapter().recognize(RecognitionInput(input_crop_id="input_crop_1"))
    assert result.raw_response == {"engine": "fake"}
    assert result.text == "hello"
