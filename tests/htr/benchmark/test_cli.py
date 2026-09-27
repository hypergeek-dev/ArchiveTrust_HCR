from archivetrust.htr.benchmark import cli
from archivetrust.htr.benchmark.layout import BenchmarkLayout


def test_docker_check_fails_when_engine_pipe_answers_without_a_server(monkeypatch, tmp_path):
    def probe(argv, timeout=60):
        if argv[:2] == ["docker", "info"]:
            return True, ""  # Docker Desktop's pipe exists but the backend is down
        return False, "not probed in this test"

    monkeypatch.setattr(cli, "_probe", probe)
    rows = {check: (status, detail) for status, check, detail in cli.readiness(BenchmarkLayout(tmp_path))}
    assert rows["docker daemon"][0] == "FAIL"
    assert "Loghi container image" not in rows
