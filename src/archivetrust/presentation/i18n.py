"""Small presentation-layer localization helper.

Domain values keep stable English enum/string values for replay and telemetry; operator-facing
labels can be translated at the UI edge.
"""

from __future__ import annotations

import os
from string import Formatter

DEFAULT_LOCALE = "en"
PSEUDO_LOCALE = "pseudo"
ENV_LOCALE = "ARCHIVETRUST_LOCALE"

_CATALOG: dict[str, dict[str, str]] = {
    "en": {
        "review.accept_selected_source": "Accept selected source",
        "review.accept_source": "Accept {source}",
        "review.accept_hint.select": "Select a source to enable Accept.",
        "review.accept_hint.selected": "{source} selected - Accept will make it canonical.",
        "review.corrected_placeholder": "...or type a corrected reading",
        "review.correct": "Correct",
        "review.skip": "Skip",
        "review.mark_ambiguous": "Mark ambiguous",
        "review.different_things": "Different things",
        "review.different_things.tooltip": (
            "Both readings look valid, but they describe different parts or scopes of the document."
        ),
        "review.reject": "Reject",
        "review.reject.tooltip": "None of these readings belong in the archive at this location.",
        "review.request_further": "Request further review",
        "review.request_further.tooltip": "Beyond my authority or expertise to judge - route elsewhere.",
        "review.empty.current_value": "All uncertainties for this document have been reviewed.",
        "review.empty.scene": "No uncertainty under review.",
        "review.accessible.queue": "Documents awaiting review",
        "review.accessible.candidates": "Candidate readings",
        "review.accessible.corrected": "Corrected reading",
        "review.accessible.document": "Original archive document",
    },
    "da": {
        "review.accept_selected_source": "Godkend valgt kilde",
        "review.accept_source": "Godkend {source}",
        "review.accept_hint.select": "Vaelg en kilde for at aktivere Godkend.",
        "review.accept_hint.selected": "{source} valgt - Godkend goer den kanonisk.",
        "review.corrected_placeholder": "...eller skriv en rettet laesning",
        "review.correct": "Ret",
        "review.skip": "Spring over",
        "review.mark_ambiguous": "Marker tvetydig",
        "review.different_things": "Forskellige ting",
        "review.different_things.tooltip": (
            "Begge laesninger kan vaere gyldige, men de beskriver forskellige dele eller omfang."
        ),
        "review.reject": "Afvis",
        "review.reject.tooltip": "Ingen af disse laesninger hoerer til i arkivet paa dette sted.",
        "review.request_further": "Bed om yderligere review",
        "review.request_further.tooltip": "Uden for min bemyndigelse eller ekspertise - send videre.",
        "review.empty.current_value": "Alle usikkerheder for dette dokument er gennemgaaet.",
        "review.empty.scene": "Ingen usikkerhed under review.",
        "review.accessible.queue": "Dokumenter der afventer review",
        "review.accessible.candidates": "Kandidatlaesninger",
        "review.accessible.corrected": "Rettet laesning",
        "review.accessible.document": "Oprindeligt arkivdokument",
    },
    "sv": {
        "review.accept_selected_source": "Godkann vald kalla",
        "review.accept_source": "Godkann {source}",
        "review.accept_hint.select": "Valj en kalla for att aktivera Godkann.",
        "review.accept_hint.selected": "{source} vald - Godkann gor den kanonisk.",
        "review.corrected_placeholder": "...eller skriv en korrigerad lasning",
        "review.correct": "Korrigera",
        "review.skip": "Hoppa over",
        "review.mark_ambiguous": "Markera tvetydig",
        "review.different_things": "Olika saker",
        "review.different_things.tooltip": (
            "Bada lasningarna kan vara giltiga, men de beskriver olika delar eller omfattningar."
        ),
        "review.reject": "Avvisa",
        "review.reject.tooltip": "Ingen av dessa lasningar hor hemma i arkivet pa denna plats.",
        "review.request_further": "Begär vidare granskning",
        "review.request_further.tooltip": "Utanfor min behorighet eller expertis - skicka vidare.",
        "review.empty.current_value": "Alla osakerheter for detta dokument har granskats.",
        "review.empty.scene": "Ingen osakerhet granskas.",
        "review.accessible.queue": "Dokument som vantar pa granskning",
        "review.accessible.candidates": "Kandidatlasningar",
        "review.accessible.corrected": "Korrigerad lasning",
        "review.accessible.document": "Ursprungligt arkivdokument",
    },
}


def available_locales() -> tuple[str, ...]:
    return tuple(sorted((*_CATALOG.keys(), PSEUDO_LOCALE)))


def current_locale() -> str:
    return normalize_locale(os.getenv(ENV_LOCALE))


def normalize_locale(locale: str | None) -> str:
    if not locale:
        return DEFAULT_LOCALE
    if locale.lower() == PSEUDO_LOCALE:
        return PSEUDO_LOCALE
    normalized = locale.replace("_", "-").split("-", 1)[0].lower()
    return normalized if normalized in _CATALOG else DEFAULT_LOCALE


def tr(key: str, *, locale: str | None = None, **values: object) -> str:
    resolved = normalize_locale(locale) if locale is not None else current_locale()
    source_locale = DEFAULT_LOCALE if resolved == PSEUDO_LOCALE else resolved
    template = _CATALOG.get(source_locale, _CATALOG[DEFAULT_LOCALE]).get(
        key, _CATALOG[DEFAULT_LOCALE].get(key, key)
    )
    text = _format(template, values)
    return _pseudo(text) if resolved == PSEUDO_LOCALE else text


def _format(template: str, values: dict[str, object]) -> str:
    required = {field for _, field, _, _ in Formatter().parse(template) if field}
    filtered = {key: value for key, value in values.items() if key in required}
    return template.format(**filtered)


def _pseudo(text: str) -> str:
    expanded = "".join(f"{char}~" if char.isalpha() else char for char in text)
    return f"[!! {expanded} !!]"
