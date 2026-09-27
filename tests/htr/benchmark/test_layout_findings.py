import pytest

from archivetrust.htr.benchmark.findings import Finding, findings_to_jsonl, findings_to_markdown, summarize
from archivetrust.htr.benchmark.layout import BenchmarkLayout, LayoutError, assert_outside, default_root


def test_default_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHIVETRUST_BENCHMARK_ROOT", str(tmp_path))
    assert default_root() == tmp_path
    monkeypatch.delenv("ARCHIVETRUST_BENCHMARK_ROOT")
    assert default_root().name == "benchmark-data" and (default_root().parent / "pyproject.toml").is_file()


def test_assert_outside_protects_incoming(tmp_path):
    layout = BenchmarkLayout(tmp_path)
    with pytest.raises(LayoutError):
        assert_outside(layout.incoming_source("s") / "x.json", layout.incoming)
    assert_outside(layout.inspection_dir("s") / "x.json", layout.incoming)


def test_findings_summary_and_rendering():
    findings = [
        Finding(severity="info", code="format.detected", message="PAGE XML"),
        Finding(severity="blocker", code="image.unreadable", message="bad", path="a.png"),
    ]
    assert summarize(findings)["blocking"] is True
    assert findings_to_jsonl(findings).splitlines()[0].startswith('{"code": "image.unreadable"')
    md = findings_to_markdown(findings)
    assert md.index("blocker") < md.index("info") and "`a.png`" in md
