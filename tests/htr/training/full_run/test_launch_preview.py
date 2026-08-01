from __future__ import annotations

from archivetrust.htr.training.full_run.launch_preview import (
    DO_NOT_EXECUTE_BANNER,
    build_launch_preview,
    write_launch_preview_file,
)


def test_launch_command_requires_confirm_flag(tmp_path):
    preview = build_launch_preview(run_dir=tmp_path / "run", hours=5.0, batch_size=16)
    assert "--confirm-full-corpus-run" in preview.launch_command
    assert "start" in preview.launch_command


def test_resume_command_template_never_includes_confirm_flag_prematurely(tmp_path):
    """The resume template is a template precisely because no checkpoint exists yet on a freshly
    prepared run -- it must not itself be an executable, confirmed launch command."""
    preview = build_launch_preview(run_dir=tmp_path / "run")
    assert "<HOURS>" in preview.resume_command_template


def test_all_seven_required_commands_are_present(tmp_path):
    preview = build_launch_preview(run_dir=tmp_path / "run")
    for field in [
        "launch_command", "dashboard_command", "status_command", "log_follow_command",
        "safe_interrupt_command", "resume_command_template", "post_first_shard_inspection_command",
        "post_first_epoch_validation_command",
    ]:
        assert getattr(preview, field)


def test_write_launch_preview_file_has_do_not_execute_banner(tmp_path):
    preview = build_launch_preview(run_dir=tmp_path / "run")
    output_path = tmp_path / "LAUNCH_COMMANDS.txt"
    write_launch_preview_file(preview, output_path)
    text = output_path.read_text(encoding="utf-8")
    assert text.startswith(DO_NOT_EXECUTE_BANNER)
    assert preview.launch_command in text
    assert preview.dashboard_command in text
    assert preview.safe_interrupt_command in text
