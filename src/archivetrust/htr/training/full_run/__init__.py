"""Operator-controlled full-corpus Loghi training workflow.

Derives monitoring policy from the pilot's real observed behavior (never its raw epoch count),
resets to the pristine pinned base checkpoint (never continues from the pilot), and provides a
GUI + CLI to prepare, preflight, start, monitor, stop, and resume the run. See
docs/methods/loghi-full-corpus-training.md.
"""
