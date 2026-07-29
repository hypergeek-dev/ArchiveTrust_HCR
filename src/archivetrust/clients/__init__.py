"""Client applications for the Learning Platform (HUMAN_REVIEW_SPECIFICATION.md §16).

Clients are interchangeable implementations of the same Learning Platform service contracts; they
consume the ViewModels (`archivetrust.presentation`) and, through them, the services
(`archivetrust.review`). No client contains review or analytics *behavior* — that lives below it
(the implementation-phase layering; HR-11). The first client is `clients.desktop`, the PySide6
Architect Client (§16.2).

Nothing in this package may be imported by any lower layer (Trust Engine, learning, review,
presentation) — enforced by `tests/review/test_separation.py`.
"""
