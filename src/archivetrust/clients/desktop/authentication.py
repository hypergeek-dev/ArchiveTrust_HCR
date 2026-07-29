"""Small, supported local-account login flow for the single-site desktop profile."""

from __future__ import annotations

from pathlib import Path

from archivetrust.admin.identity import (
    AdminAction,
    AdminAuditLog,
    AuthenticatedSession,
    IdentityRole,
    LocalAccount,
    LocalAccountStore,
    PermissionDenied,
)


def authenticate_desktop(deployment_root: Path) -> AuthenticatedSession | None:
    """Authenticate before production composition is constructed.

    The first run bootstraps exactly one local administrator through an explicit password prompt;
    later runs never fall back to an implicit identity.  Returning ``None`` means the user
    cancelled startup.
    """

    from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

    identity_root = deployment_root / "identity"
    store = LocalAccountStore(identity_root / "accounts.json")
    audit = AdminAuditLog(identity_root / "admin-audit.jsonl")

    if not store.list():
        QMessageBox.information(
            None,
            "Create ArchiveTrust administrator",
            "No local accounts exist. Create the initial administrator before opening any workspace.",
        )
        display_name, ok = QInputDialog.getText(None, "Administrator", "Display name:")
        if not ok or not display_name.strip():
            return None
        reviewer_ref, ok = QInputDialog.getText(None, "Administrator", "Account name:")
        if not ok or not reviewer_ref.strip():
            return None
        password, ok = QInputDialog.getText(None, "Administrator", "Password:", QLineEdit.EchoMode.Password)
        if not ok:
            return None
        confirmation, ok = QInputDialog.getText(
            None, "Administrator", "Repeat password:", QLineEdit.EchoMode.Password
        )
        if not ok or password != confirmation:
            QMessageBox.critical(None, "Administrator", "Passwords did not match.")
            return None
        try:
            account = LocalAccount.with_password(
                password=password,
                display_name=display_name.strip(),
                reviewer_ref=reviewer_ref.strip(),
                roles=tuple(IdentityRole),
            )
        except ValueError as exc:
            QMessageBox.critical(None, "Administrator", str(exc))
            return None
        store.add(account)
        session = AuthenticatedSession.start(account)
        audit.record(
            actor=account,
            action=AdminAction.ACCOUNT_CHANGED,
            target_ref=account.account_id,
            target_label=account.display_name,
            resulting_state_ref="initial-administrator-created",
            reason="explicit first-run bootstrap",
            session_id=session.session_id,
        )
        return session

    for _attempt in range(3):
        account_name, ok = QInputDialog.getText(None, "ArchiveTrust sign in", "Account name:")
        if not ok:
            return None
        password, ok = QInputDialog.getText(
            None, "ArchiveTrust sign in", "Password:", QLineEdit.EchoMode.Password
        )
        if not ok:
            return None
        try:
            return store.authenticate(account_name.strip(), password)
        except PermissionDenied:
            QMessageBox.warning(None, "ArchiveTrust sign in", "Invalid account name or password.")
    QMessageBox.critical(None, "ArchiveTrust sign in", "Authentication failed three times.")
    return None
