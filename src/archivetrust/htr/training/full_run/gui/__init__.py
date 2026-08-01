"""PySide6 GUI for the operator-controlled full-corpus training workflow. See `commands.py` for the
pure, Qt-free subprocess-command construction (`gui/app.py` never launches training in-process --
every action is a real `python -m archivetrust.htr.training.full_run <subcommand>` child process)."""
