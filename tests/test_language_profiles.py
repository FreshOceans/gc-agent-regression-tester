"""Unit tests for language profile normalization and selection."""

import pytest

from src.language_profiles import (
    get_language_profile,
    normalize_language_code,
    resolve_effective_language,
)


def test_normalize_language_code_accepts_aliases():
    assert normalize_language_code("fr_ca") == "fr-CA"
    assert normalize_language_code("FRENCH CANADIAN") == "fr-CA"
    assert normalize_language_code("es-es") == "es"
    assert normalize_language_code("english") == "en"


def test_normalize_language_code_rejects_unsupported():
    with pytest.raises(ValueError, match="Unsupported language"):
        normalize_language_code("de")


def test_resolve_effective_language_precedence():
    assert (
        resolve_effective_language(
            runtime_override="es",
            suite_language="fr-CA",
            config_language="en",
        )
        == "es"
    )
    assert (
        resolve_effective_language(
            runtime_override="",
            suite_language="fr",
            config_language="en",
        )
        == "fr"
    )
    assert (
        resolve_effective_language(
            runtime_override=None,
            suite_language=None,
            config_language="fr-CA",
        )
        == "fr-CA"
    )


def test_get_language_profile_contains_runtime_defaults():
    profile = get_language_profile("fr-CA")
    assert profile["default_speak_to_agent_follow_up"]
    assert profile["default_knowledge_closure_message"]
    assert "yes_tokens" in profile
    assert "no_tokens" in profile
