"""Entry point for the desktop Human Review application.

Run with ``python -m archivetrust.clients.desktop`` (optionally ``--demo`` to launch with a small
seeded sample archive in the Processing Center). Requires the optional ``gui`` extra
(``pip install -e ".[gui]"``).

**First Launch (Operational Completion milestone: one coherent onboarding flow).** The non-demo
path no longer shows the old, narrower First Launch model-setup dialog on its own — a Workspace is
the real unit of first-run configuration (ROADMAP.md §5.13.1), and its Wizard's own Providers/
Models steps already call into `FirstLaunchViewModel.complete_setup` (see
`presentation/workspace_wizard_viewmodel.py`). If no Workspace exists yet, the Workspace Wizard is
shown before the main window; if the operator closes it without finishing, `AppContext` still falls
back to a "Default Workspace" so the app is never locked out — first launch is a convenience, never
a gate. The `--demo` path skips onboarding entirely (a demo Workspace is seeded directly).

**Local `.env` support.** A `.env` file in the working directory (never committed — see
`.gitignore`) is loaded into the process environment before anything else runs, so operator secrets
like `HF_TOKEN` (raises Hugging Face Hub's unauthenticated rate limit, per `huggingface_hub`'s own
convention) reach `transformers`/`huggingface_hub` without needing a system-wide environment
variable. Never overrides a variable already set in the real environment; silently a no-op if no
`.env` file exists.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from archivetrust.composition import AppContext, build_demo_context


def load_local_env() -> bool:
    """Loads a `.env` file from the working directory into `os.environ`, if one exists (see the
    module docstring). Returns whether a file was found and loaded -- factored out of `main()` so
    it's testable without constructing a `QApplication`.
    """
    from dotenv import load_dotenv

    # Only the launch directory is trusted. Searching parent directories can silently import a
    # developer/repository secret file into a deployment started from a child working directory.
    path = Path.cwd() / ".env"
    if not path.is_file():
        return False
    return load_dotenv(path)


def main(argv: list[str] | None = None) -> int:
    load_local_env()  # HF_TOKEN and friends, from a local .env -- see module docstring

    parser = argparse.ArgumentParser(description="ArchiveTrust Human Review (desktop)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch with a seeded sample archive open in the Processing Center.",
    )
    args = parser.parse_args(argv)

    # Imported here, not at module top, so `main.py` stays importable for tooling without a Qt
    # display; constructing QApplication is what actually requires the platform.
    from PySide6.QtWidgets import QApplication

    from archivetrust.clients.desktop.workspace_wizard import WorkspaceWizard

    app = QApplication.instance() or QApplication(sys.argv[:1])

    if args.demo:
        context, _documents = build_demo_context()
    else:
        deployment_root = Path.cwd() / "archivetrust_data"
        from archivetrust.clients.desktop.authentication import authenticate_desktop

        session = authenticate_desktop(deployment_root)
        if session is None:
            return 1
        context = AppContext(
            deployment_root=deployment_root,
            auto_create_default_workspace=False,
            use_process_worker=True,
            authenticated_session=session,
            require_authentication=True,
        )
        if not context.workspace_store.list():
            wizard = WorkspaceWizard(context)
            wizard.exec()
            if not wizard.was_completed():
                # Never a lock-out: proceed with a Default Workspace the operator can rename or
                # replace later from the Workspaces page.
                context.create_workspace("Default Workspace")

    # v2 is the only desktop client (release WS2: "one client"). The v1 shell was retired after
    # its required surfaces (evidence inspection, provider management, diagnostics) were ported;
    # its page widgets live only in git history now, not behind an environment variable.
    from archivetrust.clients.desktop_v2.app import MainWindowV2

    window = MainWindowV2(context)
    window.resize(1000, 640)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
