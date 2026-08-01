from __future__ import annotations

import json

from archivetrust.htr.training.incremental_jsonl_reader import read_new_lines


def test_missing_file_returns_empty(tmp_path):
    result = read_new_lines(tmp_path / "nope.jsonl")
    assert result.rows == ()
    assert result.new_offset == 0
    assert result.malformed_count == 0


def test_reads_all_lines_from_a_fresh_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    result = read_new_lines(path)
    assert [r["a"] for r in result.rows] == [1, 2]
    assert result.new_offset == path.stat().st_size


def test_second_call_reads_only_appended_bytes(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    first = read_new_lines(path)
    assert [r["a"] for r in first.rows] == [1]

    with path.open("a", encoding="utf-8") as f:
        f.write('{"a": 2}\n')

    second = read_new_lines(path, since_offset=first.new_offset)
    assert [r["a"] for r in second.rows] == [2]  # not [1, 2] -- never re-reads from the start


def test_a_line_still_being_written_without_a_trailing_newline_is_not_consumed(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}', encoding="utf-8")  # second line has no trailing newline
    result = read_new_lines(path)
    assert [r["a"] for r in result.rows] == [1]
    assert result.new_offset < path.stat().st_size  # the incomplete line's bytes were not consumed

    with path.open("a", encoding="utf-8") as f:
        f.write("\n")  # now complete

    second = read_new_lines(path, since_offset=result.new_offset)
    assert [r["a"] for r in second.rows] == [2]


def test_malformed_lines_are_skipped_and_counted_not_fatal(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\nNOT VALID JSON\n{"a": 2}\n', encoding="utf-8")
    result = read_new_lines(path)
    assert [r["a"] for r in result.rows] == [1, 2]
    assert result.malformed_count == 1


def test_blank_lines_are_silently_skipped_not_counted_as_malformed(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    result = read_new_lines(path)
    assert [r["a"] for r in result.rows] == [1, 2]
    assert result.malformed_count == 0


def test_empty_file_returns_empty(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")
    result = read_new_lines(path)
    assert result.rows == ()
    assert result.new_offset == 0


def test_offset_past_a_shrunk_file_restarts_from_the_beginning_rather_than_erroring(tmp_path):
    """A stale cursor pointing past the real end of a smaller/replaced file (e.g. a fresh run
    reusing a path) is not a malformed-data problem -- start over cleanly."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    result = read_new_lines(path, since_offset=10_000)
    assert [r["a"] for r in result.rows] == [1]


def test_offset_zero_reads_the_whole_file_every_time_when_never_advanced(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    result_a = read_new_lines(path, since_offset=0)
    result_b = read_new_lines(path, since_offset=0)
    assert result_a.rows == result_b.rows


def test_round_trips_real_dict_values(tmp_path):
    path = tmp_path / "events.jsonl"
    payload = {"epoch": 3, "gpu": {"utilization_pct": 42.5, "name": "RTX 3070"}, "nested": [1, 2, 3]}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    result = read_new_lines(path)
    assert result.rows[0] == payload
