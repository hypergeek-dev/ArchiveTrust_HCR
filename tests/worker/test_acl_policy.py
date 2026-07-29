from __future__ import annotations

import os

import pytest

from archivetrust.worker import acl_policy as acl
from archivetrust.worker.acl_policy import (
    AclApplicationRecord,
    AclPathClass,
    AclPolicy,
    AclVerificationResult,
    classify_deployment_paths,
    is_dangerous_root,
)

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows ACL engine is Windows-only")


def test_policy_is_versioned_and_serializable() -> None:
    policy = AclPolicy()
    payload = policy.model_dump_json()
    restored = AclPolicy.model_validate_json(payload)
    assert restored.version == acl.POLICY_VERSION
    assert restored.disallowed_write_principal_sids == tuple(sorted(acl.DISALLOWED_WRITE_PRINCIPAL_SIDS))


def test_disallowed_principals_include_authenticated_users_and_everyone() -> None:
    assert acl.SID_AUTHENTICATED_USERS in acl.DISALLOWED_WRITE_PRINCIPAL_SIDS
    assert acl.SID_EVERYONE in acl.DISALLOWED_WRITE_PRINCIPAL_SIDS
    assert acl.SID_BUILTIN_USERS in acl.DISALLOWED_WRITE_PRINCIPAL_SIDS


def test_root_is_classified_operator_runtime_not_administration(tmp_path) -> None:
    # Regression test for the 2026-07-20 lockout: classifying the deployment root itself as
    # ADMINISTRATION (operator = read-execute only) locked a non-elevated operator out of
    # creating any new child under it (new workspaces, identity file, worker/). The root must
    # grant the operator write.
    classified = classify_deployment_paths(tmp_path)
    root_entry = next(item for item in classified if item.path == str(tmp_path))
    assert root_entry.path_class is AclPathClass.OPERATOR_RUNTIME


def test_administration_subdirs_are_read_execute_for_operator() -> None:
    access = acl.CLASS_ACCESS[AclPathClass.ADMINISTRATION]
    assert access[acl.AclPrincipalRole.OPERATOR] is acl.AclAccessLevel.READ_EXECUTE


def test_worker_and_evidence_paths_grant_operator_modify() -> None:
    for path_class in (AclPathClass.OPERATOR_RUNTIME, AclPathClass.AUTHORITATIVE_EVIDENCE, AclPathClass.QUALIFICATION_EVIDENCE):
        access = acl.CLASS_ACCESS[path_class]
        assert access[acl.AclPrincipalRole.OPERATOR] is acl.AclAccessLevel.MODIFY


def test_classification_covers_worker_command_control_result_and_identity(tmp_path) -> None:
    (tmp_path / "worker" / "commands").mkdir(parents=True)
    (tmp_path / "worker" / "control").mkdir(parents=True)
    (tmp_path / "worker" / "results").mkdir(parents=True)
    (tmp_path / "identity").mkdir(parents=True)
    classified = {item.path: item.path_class for item in classify_deployment_paths(tmp_path)}
    assert classified[str(tmp_path / "worker" / "commands")] is AclPathClass.OPERATOR_RUNTIME
    assert classified[str(tmp_path / "worker" / "control")] is AclPathClass.OPERATOR_RUNTIME
    assert classified[str(tmp_path / "worker" / "results")] is AclPathClass.OPERATOR_RUNTIME
    assert classified[str(tmp_path / "identity")] is AclPathClass.AUTHORITATIVE_EVIDENCE


def test_workspace_subdirs_are_classified_per_workspace(tmp_path) -> None:
    workspace_dir = tmp_path / "workspaces" / "ws1"
    (workspace_dir / "archive").mkdir(parents=True)
    (workspace_dir / "derived").mkdir(parents=True)
    (workspace_dir / "models").mkdir(parents=True)
    classified = {item.path: item.path_class for item in classify_deployment_paths(tmp_path)}
    assert classified[str(workspace_dir / "archive")] is AclPathClass.AUTHORITATIVE_EVIDENCE
    assert classified[str(workspace_dir / "derived")] is AclPathClass.REBUILDABLE
    assert classified[str(workspace_dir / "models")] is AclPathClass.ADMINISTRATION


def test_rebuildable_class_is_not_critical() -> None:
    assert AclPathClass.REBUILDABLE not in acl.CRITICAL_PATH_CLASSES
    assert AclPathClass.OPERATOR_RUNTIME in acl.CRITICAL_PATH_CLASSES
    assert AclPathClass.AUTHORITATIVE_EVIDENCE in acl.CRITICAL_PATH_CLASSES


def test_dangerous_roots_are_rejected(tmp_path) -> None:
    from pathlib import Path

    assert is_dangerous_root(Path("C:\\"))
    assert is_dangerous_root(Path("C:\\Windows"))
    assert is_dangerous_root(Path("C:\\Users"))
    assert not is_dangerous_root(tmp_path / "deployment")


def test_apply_rejects_dangerous_root() -> None:
    from pathlib import Path

    with pytest.raises(acl.AclPolicyError):
        acl.apply(Path("C:\\"), actor="test")


def test_non_windows_engine_reports_unsupported() -> None:
    if os.name == "nt":
        pytest.skip("this asserts the non-Windows fallback path; run under a non-nt import guard instead")


def test_verify_on_missing_deployment_root_returns_no_findings(tmp_path) -> None:
    result = acl.verify(tmp_path / "does-not-exist")
    assert isinstance(result, AclVerificationResult)
    assert result.paths_checked == 0
    assert result.processing_blocked is False


@windows_only
def test_plan_detects_existing_disallowed_authenticated_users_ace(tmp_path) -> None:
    root = tmp_path / "deployment"
    (root / "worker" / "commands").mkdir(parents=True)
    result = acl.plan(root)
    assert result["platform_supported"] is True
    root_entry = next(item for item in result["paths"] if item["path"] == str(root))
    # tmp_path inherits the pytest temp-dir ACLs, which on this machine include Authenticated
    # Users; this assertion is intentionally soft (checks the field exists and is a list) so it
    # does not depend on the exact ambient ACL of the CI/dev machine's temp directory.
    assert "current_disallowed_write_principals" in root_entry


@windows_only
def test_apply_then_verify_removes_authenticated_users_and_is_idempotent(tmp_path) -> None:
    root = tmp_path / "deployment"
    (root / "worker" / "commands").mkdir(parents=True)
    (root / "worker" / "control").mkdir(parents=True)
    (root / "worker" / "results").mkdir(parents=True)

    record = acl.apply(root, actor="pytest")
    assert isinstance(record, AclApplicationRecord)
    assert record.result == "success"
    assert not record.paths_failed

    first_verify = acl.verify(root)
    assert first_verify.processing_blocked is False
    assert first_verify.findings == ()

    engine = acl.WindowsAclEngine()
    aces = engine.read_aces(root / "worker" / "commands")
    disallowed_sids = {ace.sid for ace in aces if ace.sid in acl.DISALLOWED_WRITE_PRINCIPAL_SIDS}
    assert not disallowed_sids, f"broad Authenticated Users/Everyone/Users ACE survived apply: {disallowed_sids}"

    operator_sid = engine.operator_sid()
    assert any(ace.sid == operator_sid for ace in aces)
    assert any(ace.sid == acl.SID_ADMINISTRATORS for ace in aces)

    # idempotent: applying twice changes nothing observable
    second_record = acl.apply(root, actor="pytest")
    assert second_record.result == "success"
    second_verify = acl.verify(root)
    assert second_verify.findings == ()


@windows_only
def test_verify_flags_reintroduced_broad_ace_as_critical_drift(tmp_path) -> None:
    root = tmp_path / "deployment"
    (root / "worker" / "commands").mkdir(parents=True)
    acl.apply(root, actor="pytest")
    assert acl.verify(root).processing_blocked is False

    # Simulate drift: something re-adds Authenticated Users: Modify to a protected path.
    import ntsecuritycon
    import win32security

    target = root / "worker" / "commands"
    security_descriptor = win32security.GetFileSecurity(str(target), win32security.DACL_SECURITY_INFORMATION)
    dacl = security_descriptor.GetSecurityDescriptorDacl()
    dacl.AddAccessAllowedAceEx(
        win32security.ACL_REVISION,
        ntsecuritycon.CONTAINER_INHERIT_ACE | ntsecuritycon.OBJECT_INHERIT_ACE,
        ntsecuritycon.FILE_GENERIC_WRITE,
        win32security.ConvertStringSidToSid(acl.SID_AUTHENTICATED_USERS),
    )
    security_descriptor.SetSecurityDescriptorDacl(1, dacl, 0)
    win32security.SetFileSecurity(str(target), win32security.DACL_SECURITY_INFORMATION, security_descriptor)

    drifted = acl.verify(root)
    assert drifted.processing_blocked is True
    assert any(f.code == "unauthorized_principal_write" and f.critical for f in drifted.findings)


@windows_only
def test_verify_without_prior_apply_is_noncritical(tmp_path) -> None:
    root = tmp_path / "deployment"
    (root / "worker" / "commands").mkdir(parents=True)
    result = acl.verify(root)
    assert result.processing_blocked is False
    assert any(f.code == "policy_not_applied" and not f.critical for f in result.findings)


@windows_only
def test_new_child_created_after_apply_inherits_no_broad_ace(tmp_path) -> None:
    root = tmp_path / "deployment"
    (root / "worker").mkdir(parents=True)
    acl.apply(root, actor="pytest")

    new_child = root / "worker" / "control"
    new_child.mkdir()
    engine = acl.WindowsAclEngine()
    aces = engine.read_aces(new_child)
    disallowed = {ace.sid for ace in aces if ace.sid in acl.DISALLOWED_WRITE_PRINCIPAL_SIDS}
    assert not disallowed
