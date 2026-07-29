from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from archivetrust.admin.health import DeploymentHealth


class AlertFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    alert_id: str
    kind: str
    key: str
    severity: str
    message: str
    guidance: str
    actor: str | None = None
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AlertStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def facts(self) -> tuple[AlertFact, ...]:
        return tuple(
            AlertFact.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def append(self, fact: AlertFact) -> AlertFact:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(fact.model_dump_json() + "\n")
        return fact

    def active(self) -> tuple[AlertFact, ...]:
        facts = self.facts()
        acknowledged = {fact.alert_id for fact in facts if fact.kind == "acknowledged"}
        return tuple(fact for fact in facts if fact.kind == "raised" and fact.alert_id not in acknowledged)

    def acknowledge(self, alert_id: str, *, actor: str) -> AlertFact:
        raised = next((fact for fact in self.facts() if fact.alert_id == alert_id and fact.kind == "raised"), None)
        if raised is None:
            raise KeyError(alert_id)
        if any(fact.alert_id == alert_id and fact.kind == "acknowledged" for fact in self.facts()):
            raise ValueError("alert already acknowledged")
        return self.append(
            AlertFact(
                alert_id=alert_id,
                kind="acknowledged",
                key=raised.key,
                severity=raised.severity,
                message="alert acknowledged",
                guidance=raised.guidance,
                actor=actor,
            )
        )


_GUIDANCE = {
    "storage_capacity": "Free storage or move the deployment before starting another run.",
    "backup_freshness": "Stop processing and create a verified deployment backup.",
    "queue_health": "Inspect worker command results and reconcile interrupted runs.",
}


def record_health_alerts(health: DeploymentHealth, store: AlertStore) -> tuple[AlertFact, ...]:
    active_keys = {fact.key for fact in store.active()}
    created = []
    for check in health.checks:
        if check.status not in {"warning", "critical"} or check.name in active_keys:
            continue
        key = check.name
        created.append(
            store.append(
                AlertFact(
                    alert_id=uuid.uuid4().hex,
                    kind="raised",
                    key=key,
                    severity=check.status,
                    message=check.detail,
                    guidance=_GUIDANCE.get(key.split(":", 1)[0], "Open Administration diagnostics and resolve this check before continuing."),
                )
            )
        )
    return tuple(created)
