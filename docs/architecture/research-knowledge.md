# Research Knowledge: Domain-Relationship and Finding-Scoping Guidance

Status: Current
Governs: How `DomainRelationship`, `corpus_language`, and `method_primary_language_domain`
(`htr/experiment/models.py`) interact with the existing `ResearchScope`/`ResearchFinding` knowledge
layer (`docs/research-findings.md`, `docs/research-observations.md`) for the Lion-vs-Loghi research
phase and any future domain-transfer comparison.

## Why this document exists

`docs/research-findings.md` already specifies `ResearchScope` and `ResearchFinding` in full — this
document does not repeat that model. What it adds is specific to comparisons that cross a language or
training-domain boundary, which the retired three-method benchmark never did (SATRN, Florence-2, and
Transkribus Swedish Lion I were all evaluated on the same Swedish corpus).

## Domain relationship is a fact about a *run*, not a fact about a *method*

`DomainRelationship` (`IN_DOMAIN` / `CROSS_DOMAIN` / `MIXED_DOMAIN` / `UNKNOWN`) lives on
`ExperimentVersion`, scoped to one `(method, corpus)` pairing, never on `MethodMetadata` or
`MethodCapabilities`. The same method (`loghi`) is `in_domain` on the Dutch corpus and `cross_domain` on
the Swedish corpus in the same comparison — a single "domain" label on the method itself could not
express that. See `htr/screening/lion_loghi_experiment.py::build_lion_loghi_comparison` for where the
four cells' domain relationships are assigned, explicitly, never inferred from dataset file location
(the brief's own "do not infer domain solely from file location").

## A finding drawn from a cross-domain comparison must state the comparison it rests on

`ResearchScope`'s existing structural discipline (`experiment_version_id`/`experiment_run_id` required,
`sample_size` derived, `method_ids`/`model_version_ids` paired) already prevents an unscoped claim. For
a domain-transfer finding specifically, the scope's named `experiment_version_id` must resolve to an
`ExperimentVersion` whose `domain_relationship` is set — so "Loghi degrades on cross-domain Swedish
pages" is traceable to the exact cell (`loghi` × `dataset-rgb`, `cross_domain`) it was measured on, not
to Loghi in general.

## Transfer loss is not architecture quality

Per the brief: **domain-relationship status does not explain all transfer effects.** The Swedish and
Dutch corpora are only "Approximately matched" or worse on most dimensions
(`docs/experiments/lion-loghi-comparison/dataset-comparability.md`) — document genre is a **known
mismatch**. A finding that reports a cross-domain accuracy drop must carry that comparability gap as one
of its own `limitations`, exactly as `docs/research-findings.md`'s existing findings carry their N=1 and
non-comparable-confidence limitations. A future finding phrased as "Loghi's architecture handles domain
shift worse than Swedish Lion I's" without that limitation would overclaim what a genre-confounded,
N-small comparison can support.

## No finding is created by this integration pass

Nothing in this document, or in `htr/screening/lion_loghi_experiment.py`, creates a `ResearchFinding` or
a `ResearchObservation`. The four `ExperimentVersion`s this phase builds carry `domain_relationship`
classifications and no results — per the brief's explicit "do not fabricate experiment results before
the Swedish and Dutch datasets have been prepared and both systems have actually run."
