"""Evidence-chain ViewModel: the `docs/htr-domain-design.md` §4 chain as walkable breadcrumbs."""

from __future__ import annotations

from archivetrust.htr.experiment.models import MethodRun
from archivetrust.presentation.htr_evidence_viewmodel import HtrEvidenceChainViewModel
from tests.presentation._htr_fixtures import build_fixture_corpus


def _kinds(chain) -> list[str]:
    return [link.kind for link in chain.links]


def test_canonical_span_walks_the_whole_chain_to_the_research_project() -> None:
    corpus = build_fixture_corpus()
    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_canonical_span(
        corpus.canonical_result_id, corpus.line_0_id
    )

    assert chain.complete is True
    assert chain.broken_at is None
    kinds = _kinds(chain)
    # Every hop §4 names, in order, with the four stage hops between canonical and method run.
    assert kinds[0] == "canonical_result"
    assert kinds[-1] == "project"
    for expected in (
        "result_stage",
        "method_run",
        "model_version",
        "input_crop",
        "text_line",
        "region",
        "page",
        "document",
        "collection",
        "dataset_version",
        "dataset",
        "project",
    ):
        assert expected in kinds, expected


def test_stage_hops_appear_newest_first_reviewed_before_raw() -> None:
    """The chain reads backwards from the canonical result, so the reviewed stage is nearer the top
    and the raw stage nearer the method run -- matching §4's arrow direction."""
    corpus = build_fixture_corpus()
    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_method_run(corpus.satrn_run_line_0)

    stage_labels = [link.label for link in chain.links if link.kind == "result_stage"]
    assert stage_labels == ["Human-reviewed", "Normalized", "Parsed", "Raw output"]


def test_a_run_with_no_reviewed_stage_simply_omits_that_hop() -> None:
    """A stage that never happened is not a dangling reference -- it is omitted, not marked broken."""
    corpus = build_fixture_corpus()
    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_method_run(corpus.florence_run_line_0)

    stage_labels = [link.label for link in chain.links if link.kind == "result_stage"]
    assert "Human-reviewed" not in stage_labels
    assert stage_labels == ["Normalized", "Parsed", "Raw output"]
    assert chain.complete is True


def test_every_link_carries_the_id_it_projects_so_a_view_can_navigate_onward() -> None:
    corpus = build_fixture_corpus()
    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_method_run(corpus.satrn_run_line_0)
    by_kind = {link.kind: link for link in chain.links}

    assert by_kind["method_run"].entity_id == corpus.satrn_run_line_0
    assert by_kind["input_crop"].entity_id == corpus.crop_a_id
    assert by_kind["text_line"].entity_id == corpus.line_0_id
    assert by_kind["region"].entity_id == corpus.region_id
    assert by_kind["page"].entity_id == corpus.page_id
    assert by_kind["project"].entity_id == corpus.project_id


def test_a_dangling_reference_is_reported_and_stops_the_walk() -> None:
    """§4's chain is id references, so a hop can genuinely dangle. It must be visible as a broken
    link rather than producing a shorter chain that looks complete."""
    corpus = build_fixture_corpus()
    orphan = MethodRun.create(
        experiment_run_id=corpus.experiment_run_id,
        method_id="satrn",
        evidence_id="evidence_orphan",
        outcome="succeeded",
        started_at="2026-07-03T03:00:00Z",
        input_crop_id="input_crop_never_registered",
    )
    corpus.store.register_method_run(orphan)

    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_method_run(orphan.method_run_id)
    assert chain.complete is False
    assert chain.broken_at == "input_crop_never_registered"
    broken = [link for link in chain.links if not link.resolved]
    assert broken and "not registered" in broken[-1].detail


def test_end_to_end_run_without_a_shared_crop_is_labeled_not_broken() -> None:
    """`MethodRun.input_crop_id=None` is a documented end-to-end mode, not a data error -- so the
    chain is incomplete with no `broken_at`, and says why."""
    corpus = build_fixture_corpus()
    end_to_end = MethodRun.create(
        experiment_run_id=corpus.experiment_run_id,
        method_id="florence2_htr",
        evidence_id="evidence_e2e",
        outcome="succeeded",
        started_at="2026-07-03T04:00:00Z",
    )
    corpus.store.register_method_run(end_to_end)

    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_method_run(end_to_end.method_run_id)
    assert chain.complete is False
    assert chain.broken_at is None
    crop_link = next(link for link in chain.links if link.kind == "input_crop")
    assert "own segmentation" in crop_link.detail


def test_unrecorded_model_version_is_an_unresolved_link_not_a_blank_label() -> None:
    corpus = build_fixture_corpus()
    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_method_run(
        corpus.florence_run_line_1_failed
    )
    model_link = next(link for link in chain.links if link.kind == "model_version")

    assert model_link.resolved is False
    assert model_link.label == "Model version not recorded"


def test_unknown_origins_return_a_single_unresolved_link_rather_than_raising() -> None:
    corpus = build_fixture_corpus()
    vm = HtrEvidenceChainViewModel(corpus.store)

    run_chain = vm.chain_for_method_run("nope")
    assert run_chain.complete is False
    assert run_chain.broken_at == "nope"
    assert len(run_chain.links) == 1

    canonical_chain = vm.chain_for_canonical_span("nope", corpus.line_0_id)
    assert canonical_chain.complete is False
    assert canonical_chain.broken_at == "nope"


def test_canonical_result_without_a_span_for_the_requested_line_stops_cleanly() -> None:
    corpus = build_fixture_corpus()
    chain = HtrEvidenceChainViewModel(corpus.store).chain_for_canonical_span(
        corpus.canonical_result_id, corpus.line_1_id
    )

    assert chain.complete is False
    assert chain.broken_at == corpus.line_1_id
    assert _kinds(chain) == ["canonical_result"]
