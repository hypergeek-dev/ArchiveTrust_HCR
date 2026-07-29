from __future__ import annotations

import pytest

from archivetrust.admin.identity import (
    AdminAction,
    AdminAuditLog,
    IdentityRole,
    LocalAccount,
    LocalAccountStore,
    PermissionDenied,
    AuthenticatedSession,
    require_role,
)


def _admin() -> LocalAccount:
    return LocalAccount(
        display_name="Ada Administrator",
        reviewer_ref="reviewer.ada",
        roles=(IdentityRole.ADMINISTRATOR, IdentityRole.REVIEWER),
    )


def test_require_role_allows_matching_active_account() -> None:
    require_role(_admin(), IdentityRole.ADMINISTRATOR)


def test_require_role_rejects_missing_role_or_inactive_account() -> None:
    operator = LocalAccount(
        display_name="Olivia Operator",
        reviewer_ref="reviewer.olivia",
        roles=(IdentityRole.OPERATOR,),
    )
    inactive = _admin().model_copy(update={"active": False})

    with pytest.raises(PermissionDenied):
        require_role(operator, IdentityRole.ADMINISTRATOR)
    with pytest.raises(PermissionDenied):
        require_role(inactive, IdentityRole.ADMINISTRATOR)


def test_local_account_store_persists_accounts_and_rejects_duplicate_reviewer_ref(tmp_path) -> None:
    store = LocalAccountStore(tmp_path / "accounts.json")
    account = _admin()

    store.add(account)
    reloaded = LocalAccountStore(tmp_path / "accounts.json")

    assert reloaded.get(account.account_id) == account
    with pytest.raises(ValueError):
        reloaded.add(
            LocalAccount(
                display_name="Duplicate",
                reviewer_ref=account.reviewer_ref,
                roles=(IdentityRole.REVIEWER,),
            )
        )


def test_admin_audit_log_is_append_only_and_reloads(tmp_path) -> None:
    path = tmp_path / "admin-audit.jsonl"
    log = AdminAuditLog(path)
    actor = _admin()

    event = log.record(
        actor=actor,
        action=AdminAction.WORKSPACE_DELETED,
        target_ref="workspace-1",
        target_label="Municipal archive",
    )
    reloaded = AdminAuditLog(path)

    assert reloaded.all_events() == (event,)
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert reloaded.verify_integrity()[0] is True


def test_password_authentication_session_expiry_disable_and_reset(tmp_path) -> None:
    store = LocalAccountStore(tmp_path / "accounts.json")
    account = LocalAccount.with_password(
        password="Municipal-Archive-2026",
        display_name="Ada Administrator",
        reviewer_ref="reviewer.ada",
        roles=(IdentityRole.ADMINISTRATOR,),
    )
    store.add(account)

    session = store.authenticate("reviewer.ada", "Municipal-Archive-2026")
    assert session.require_active().account_id == account.account_id
    with pytest.raises(PermissionDenied):
        store.authenticate("reviewer.ada", "wrong")

    disabled = account.model_copy(update={"active": False})
    store.replace(disabled)
    with pytest.raises(PermissionDenied):
        store.authenticate("reviewer.ada", "Municipal-Archive-2026")


def test_audit_hash_chain_detects_tampering(tmp_path) -> None:
    path = tmp_path / "admin-audit.jsonl"
    log = AdminAuditLog(path)
    log.record(actor=_admin(), action=AdminAction.WORKSPACE_DELETED, target_ref="workspace-1")
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("workspace-1", "workspace-2"), encoding="utf-8")

    reloaded = AdminAuditLog(path)
    ok, reason = reloaded.verify_integrity()
    assert ok is False
    assert "hash mismatch" in reason
