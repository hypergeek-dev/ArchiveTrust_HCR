"""Framework-independent presentation layer — ViewModels and their binding primitives.

This package is the **ViewModels** tier of the GUI → ViewModels → Services → Trust Engine stack
(the implementation-phase layering). It contains no GUI-framework code: no PySide6, no Qt, nothing
that ties it to a particular client. That is deliberate and load-bearing —

- **ViewModels hold behavior; Views only orchestrate interaction.** Every reviewer action, timing
  measurement, and state transition lives here, expressed against the `archivetrust.review` service
  contracts. A Qt (or future web/tablet) View binds to these ViewModels and renders them; it never
  re-implements review logic or reaches past them into the Trust Engine (the implementation-phase
  rule "the GUI should only orchestrate interaction").
- **Consistency across clients (HUMAN_REVIEW_SPECIFICATION.md §16.4, HR-11, UX-INV-10).** Because
  the behavior lives in a framework-independent ViewModel, every client that binds to it behaves
  identically with respect to the invariants and the decision model, while presenting a UX
  appropriate to its audience.

Binding to a View is done through the minimal `Observable` primitive (`observable.py`): a ViewModel
exposes observables, and a View subscribes to them. This keeps the ViewModels testable headless
(the entire behavior is exercised in `tests/presentation/` with no display) and keeps the Qt layer
a thin adapter over signals it does not own.

Dependency direction: this package may import `archivetrust.review` and `archivetrust.learning`;
nothing in the Trust Engine may import it (enforced by `tests/review/test_separation.py`).
"""
