"""Method overview ViewModel: real adapter metadata, capabilities, and README limitations."""

from __future__ import annotations

from archivetrust.presentation.htr_methods_viewmodel import (
    MethodOverviewViewModel,
    parse_known_limitations,
)
from archivetrust.providers.florence2_htr.adapter import Florence2Adapter
from archivetrust.providers.satrn.adapter import SatrnAdapter
from archivetrust.providers.transkribus.adapter import TranskribusAdapter
from tests.presentation._htr_fixtures import StubAdapter, build_fixture_corpus


def _real_adapters() -> tuple:
    """The three real adapters, constructed without touching a model or a network -- each one's
    constructor is inert; only `recognize()`/`validate_environment()` reach outward."""
    return (SatrnAdapter(), Florence2Adapter(), TranskribusAdapter())


def test_the_three_real_methods_report_their_own_identity_not_a_hardcoded_table() -> None:
    rows = MethodOverviewViewModel(_real_adapters()).method_rows()
    by_id = {row.method_id: row for row in rows}

    assert set(by_id) == {"satrn", "florence2_htr", "transkribus_swedish_lion_1"}
    assert by_id["satrn"].vendor == "Riksarkivet (Swedish National Archives)"
    # The pinned commit the SATRN adapter actually declares -- never "main"/"latest".
    assert by_id["satrn"].model_revision == "a40c7093232eaa47a83ce6469fc4abd033486bdc"
    assert by_id["transkribus_swedish_lion_1"].vendor == "READ-COOP (Transkribus)"


def test_adapter_version_comes_from_the_adapter_module_constant() -> None:
    from archivetrust.providers.satrn.adapter import ADAPTER_VERSION

    rows = MethodOverviewViewModel(_real_adapters()).method_rows()
    satrn = next(row for row in rows if row.method_id == "satrn")
    assert satrn.adapter_version == ADAPTER_VERSION


def test_every_capability_flag_is_listed_including_the_false_ones() -> None:
    """`MethodCapabilities` declares all six as required booleans precisely so an unsupported
    capability is an explicit False, never an omission. The surface must preserve that."""
    rows = MethodOverviewViewModel(_real_adapters()).method_rows()
    satrn = next(row for row in rows if row.method_id == "satrn")
    by_field = {cap.field: cap for cap in satrn.capabilities}

    assert set(by_field) == {
        "confidence_supported",
        "geometry_supported",
        "line_level_supported",
        "page_level_supported",
        "local_execution_supported",
        "external_upload_required",
    }
    assert by_field["line_level_supported"].supported is True
    assert by_field["page_level_supported"].supported is False  # present, and false
    assert by_field["page_level_supported"].label == "Page-level input"


def test_local_versus_external_is_derived_from_the_adapters_own_flags() -> None:
    rows = MethodOverviewViewModel(_real_adapters()).method_rows()
    by_id = {row.method_id: row for row in rows}

    assert by_id["satrn"].is_local is True
    assert by_id["satrn"].execution_location == "Local"
    # Transkribus runs no model in this process -- its adapter says so itself.
    assert by_id["transkribus_swedish_lion_1"].is_local is False
    assert by_id["transkribus_swedish_lion_1"].execution_location == "External (manual import)"


def test_known_limitations_are_the_real_bullets_from_each_adapter_readme() -> None:
    rows = MethodOverviewViewModel(_real_adapters()).method_rows()
    by_id = {row.method_id: row for row in rows}

    for method_id in ("satrn", "florence2_htr", "transkribus_swedish_lion_1"):
        assert by_id[method_id].known_limitations, method_id

    satrn_text = " ".join(by_id["satrn"].known_limitations)
    assert "Line-level only" in satrn_text
    florence_text = " ".join(by_id["florence2_htr"].known_limitations)
    assert "proxy" in florence_text
    transkribus_text = " ".join(by_id["transkribus_swedish_lion_1"].known_limitations)
    assert "No real Transkribus export was available" in transkribus_text


def test_wrapped_readme_bullets_are_rejoined_into_one_limitation_each() -> None:
    parsed = parse_known_limitations(
        "# Title\n"
        "\n"
        "## Known limitations\n"
        "\n"
        "- First limitation that wraps\n"
        "  onto a second line.\n"
        "- Second limitation.\n"
        "\n"
        "## Next section\n"
        "- not a limitation\n"
    )
    assert parsed == (
        "First limitation that wraps onto a second line.",
        "Second limitation.",
    )


def test_a_readme_without_the_section_yields_no_fabricated_limitation() -> None:
    assert parse_known_limitations("# Title\n\nSome prose, no limitations heading.\n") == ()


def test_environment_is_not_probed_unless_asked_and_none_means_not_probed() -> None:
    """`None` (not probed) and `False` (probed, unavailable) are different facts."""
    adapters = (
        StubAdapter(
            method_id="stub_ok",
            method_name="Stub OK",
            vendor="Test",
            model_revision="r1",
            local=True,
            external_upload=False,
        ),
        StubAdapter(
            method_id="stub_broken",
            method_name="Stub Broken",
            vendor="Test",
            model_revision="r2",
            local=True,
            external_upload=False,
            valid=False,
            healthy=False,
        ),
    )
    vm = MethodOverviewViewModel(adapters)

    unprobed = {row.method_id: row for row in vm.method_rows()}
    assert unprobed["stub_ok"].environment_valid is None
    assert unprobed["stub_ok"].healthy is None
    assert unprobed["stub_ok"].environment_messages == ()

    probed = {row.method_id: row for row in vm.method_rows(probe_environment=True)}
    assert probed["stub_ok"].environment_valid is True
    assert probed["stub_broken"].environment_valid is False
    assert probed["stub_broken"].environment_messages == ("model weights not found",)
    assert probed["stub_broken"].healthy is False
    assert probed["stub_broken"].health_message == "stub probe"


def test_run_history_reports_last_success_last_failure_and_its_reason() -> None:
    corpus = build_fixture_corpus()
    runs = corpus.store.method_runs()
    failure_reasons = {
        corpus.florence_run_line_1_failed: "CUDA out of memory while loading the fine-tuned checkpoint"
    }
    vm = MethodOverviewViewModel(
        _real_adapters(),
        method_runs=runs,
        run_devices={corpus.satrn_run_line_0: "cuda"},
        failure_reasons=failure_reasons,
    )
    by_id = {row.method_id: row for row in vm.method_rows()}

    assert by_id["satrn"].total_runs == 2
    assert by_id["satrn"].succeeded_runs == 2
    assert by_id["satrn"].failed_runs == 0
    assert by_id["satrn"].last_success_at == "2026-07-03T01:04:00Z"
    assert by_id["satrn"].last_failure_at is None
    assert by_id["satrn"].device == "cuda"

    assert by_id["florence2_htr"].failed_runs == 1
    assert by_id["florence2_htr"].last_failure_at == "2026-07-03T01:05:00Z"
    assert "CUDA out of memory" in by_id["florence2_htr"].last_failure_reason


def test_a_method_that_never_ran_reports_no_device_rather_than_a_default() -> None:
    rows = MethodOverviewViewModel(_real_adapters()).method_rows()
    assert all(row.device is None for row in rows)
    assert all(row.total_runs == 0 for row in rows)


def test_rows_are_returned_in_a_stable_alphabetical_order() -> None:
    rows = MethodOverviewViewModel(_real_adapters()).method_rows()
    assert [row.display_name for row in rows] == sorted(row.display_name for row in rows)
