"""The PySide6 desktop Human Review application — the Architect Client
(HUMAN_REVIEW_SPECIFICATION.md §16.2).

This subpackage is the only framework-dependent layer in the codebase. Its `__init__` is kept
import-light on purpose: importing `archivetrust.clients.desktop` must not pull in PySide6, so the
framework-independent composition root (`composition.py`) can be imported and tested headless. The
Qt widgets (`app.py`, `pages/`) import PySide6 and are loaded only when the application is actually
launched (`python -m archivetrust.clients.desktop`).
"""
