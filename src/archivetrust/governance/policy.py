from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED_PERSONAL_DATA = "restricted_personal_data"


class ExternalExportPolicy(str, Enum):
    PROHIBITED = "prohibited"
    PII_REVIEW_REQUIRED = "pii_review_required"
    APPROVED = "approved"


class GovernanceViolation(PermissionError):
    pass


class WorkspaceGovernance(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_version: int = 1
    classification: DataClassification = DataClassification.INTERNAL
    retention_until: datetime | None = None
    legal_hold: bool = False
    legal_hold_reason: str | None = None
    external_export_policy: ExternalExportPolicy = ExternalExportPolicy.PROHIBITED
    encryption_required: bool = False

    def assert_permanent_deletion_allowed(self, *, now: datetime | None = None) -> None:
        if self.legal_hold:
            raise GovernanceViolation("workspace is under legal hold and cannot be permanently deleted")
        effective_now = now or datetime.now(timezone.utc)
        if self.retention_until is not None and effective_now < self.retention_until:
            raise GovernanceViolation(
                f"workspace retention period does not expire until {self.retention_until.isoformat()}"
            )

    def assert_external_export_allowed(self, *, pii_review_approved: bool) -> None:
        if self.external_export_policy is ExternalExportPolicy.PROHIBITED:
            raise GovernanceViolation("external/public export is prohibited for this workspace")
        if self.external_export_policy is ExternalExportPolicy.PII_REVIEW_REQUIRED and not pii_review_approved:
            raise GovernanceViolation("external export requires an approved PII review")
