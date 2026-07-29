from __future__ import annotations

from archivetrust.admin.health import DeploymentHealth, HealthCheck
from archivetrust.operations.monitoring import AlertStore, record_health_alerts


def test_alert_history_is_durable_deduplicated_and_acknowledge_is_append_only(tmp_path) -> None:
    health = DeploymentHealth(
        checked_at="2026-07-17T00:00:00+00:00",
        overall="critical",
        checks=(HealthCheck(name="storage_capacity", status="critical", detail="4% free"),),
    )
    store = AlertStore(tmp_path / "alerts.jsonl")
    created = record_health_alerts(health, store)
    assert len(created) == 1
    assert record_health_alerts(health, store) == ()
    store.acknowledge(created[0].alert_id, actor="operator.ada")

    assert store.active() == ()
    facts = AlertStore(store.path).facts()
    assert [fact.kind for fact in facts] == ["raised", "acknowledged"]
    assert facts[1].actor == "operator.ada"
