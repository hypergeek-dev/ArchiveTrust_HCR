from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_repository_declares_apache_license() -> None:
    license_text = read("LICENSE")
    pyproject = read("pyproject.toml")

    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert 'license = { file = "LICENSE" }' in pyproject
    assert "Apache Software License" in pyproject


def test_contributing_guide_names_architecture_and_data_guardrails() -> None:
    guide = read("CONTRIBUTING.md")

    assert "ARCHITECTURAL_CONSTITUTION.md" in guide
    assert "provider independence" in guide.lower()
    assert "pytest" in guide
    assert "Dataset contributions are different" in guide


def test_constitution_has_contributor_amendment_governance() -> None:
    constitution = read("ARCHITECTURAL_CONSTITUTION.md")

    assert "Contributor governance" in constitution
    assert "governance note" in constitution
    assert "audit/incident/benchmark/roadmap" in constitution


def test_data_policy_distinguishes_code_license_from_dataset_license_and_pii() -> None:
    policy = read("docs/DATA_HANDLING_POLICY.md")

    assert "Code License Is Not A Data License" in policy
    assert "PII" in policy
    assert "dataset license" in policy
    assert "Telemetry can contain raw provider output" in policy
