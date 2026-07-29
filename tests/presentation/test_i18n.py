from __future__ import annotations

from archivetrust.presentation.i18n import available_locales, normalize_locale, tr


def test_unknown_locale_falls_back_to_english() -> None:
    assert normalize_locale("fr-FR") == "en"
    assert tr("review.skip", locale="fr-FR") == "Skip"


def test_danish_and_swedish_catalogs_have_review_action_starters() -> None:
    assert tr("review.skip", locale="da") == "Spring over"
    assert tr("review.reject", locale="sv") == "Avvisa"


def test_pseudo_locale_expands_english_text_for_layout_testing() -> None:
    pseudo = tr("review.accept_selected_source", locale="pseudo")
    assert pseudo.startswith("[!! ")
    assert pseudo.endswith(" !!]")
    assert len(pseudo) > len("Accept selected source")


def test_available_locales_names_supported_catalogs() -> None:
    assert {"da", "en", "sv", "pseudo"}.issubset(set(available_locales()))
