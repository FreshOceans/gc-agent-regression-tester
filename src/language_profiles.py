"""Language profiles and normalization helpers for multilingual execution."""

from __future__ import annotations

from typing import Any, Optional

SUPPORTED_LANGUAGE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("fr", "French"),
    ("fr-CA", "French (Canada)"),
    ("es", "Spanish"),
)

EVALUATION_RESULTS_LANGUAGE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("inherit", "Inherit Run & Transcript Language"),
    ("en", "English"),
    ("fr", "French"),
    ("fr-CA", "French (Canada)"),
    ("es", "Spanish"),
)

_CANONICAL_CODES = {code for code, _ in SUPPORTED_LANGUAGE_OPTIONS}

_ALIAS_TO_CANONICAL = {
    # English
    "en": "en",
    "english": "en",
    "en-us": "en",
    "en-ca": "en",
    "en-gb": "en",
    # French
    "fr": "fr",
    "french": "fr",
    "fr-fr": "fr",
    # French Canadian
    "fr-ca": "fr-CA",
    "frca": "fr-CA",
    "fr-canada": "fr-CA",
    "french-canadian": "fr-CA",
    "french canadian": "fr-CA",
    "canadian french": "fr-CA",
    # Spanish
    "es": "es",
    "spanish": "es",
    "es-es": "es",
    "es-mx": "es",
}

_DEFAULT_GREETING = (
    "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"
)

_LANGUAGE_PROFILES: dict[str, dict[str, Any]] = {
    "en": {
        "label": "English",
        "judge_language_instruction": "Use English for natural-language outputs.",
        "default_speak_to_agent_follow_up": "Yes, connect me to a live agent",
        "default_knowledge_closure_message": "no, thank you that is all",
        "default_flight_priority_yes": "yes",
        "default_flight_priority_no": "no",
        "default_vacation_flight_only": "flight only",
        "default_vacation_flight_and_hotel": "flight and hotel",
        "yes_tokens": {"yes", "y", "yeah", "yep", "affirmative"},
        "no_tokens": {"no", "n", "nope", "nah", "negative"},
        "vacation_flight_only_tokens": {"flight only", "flight"},
        "vacation_flight_and_hotel_tokens": {
            "flight and hotel",
            "flight & hotel",
            "hotel and flight",
            "hotel & flight",
        },
        "greeting_aliases": [_DEFAULT_GREETING],
        "transcript_customer_speaker_tokens": {
            "customer",
            "user",
            "traveler",
            "guest",
            "consumer",
            "client",
            "enduser",
            "end-user",
            "inbound",
            "visitor",
            "external",
            "externaluser",
            "customerparticipant",
            "webuser",
            "messengeruser",
        },
        "transcript_agent_speaker_tokens": {
            "agent",
            "assistant",
            "bot",
            "system",
            "outbound",
            "server",
            "ivr",
            "workflow",
            "architect",
            "flow",
            "acd",
            "queue",
            "script",
            "auto",
            "automation",
            "agentparticipant",
            "virtualassistant",
        },
        "transcript_ignored_exact_messages": {
            "conversation ended by user",
            "conversation ended",
            "presence events are not supported in this configuration",
            "typing",
            "typing...",
        },
        "transcript_ignored_prefixes": (
            "conversation_id:",
            "participant_id:",
            "detected_intent:",
            "conversationid:",
            "participantid:",
        ),
        "seeded_persona": (
            "A traveler contacting the WestJet Travel Agent for flight-related help. "
            "They explain their request clearly and provide needed details when asked."
        ),
        "seeded_goal_guideline": (
            "Help the traveler with a pricing-guidance request. The goal is achieved "
            "when the WestJet Travel Agent clearly explains it does not provide "
            "specific baggage fee or pricing details in chat, then directs the traveler "
            "to the WestJet website or app for current costs."
        ),
        "seeded_goal_intent_template": (
            "Help the traveler with a {intent_label} request. The goal is achieved when "
            "the WestJet Travel Agent provides a relevant, policy-aligned response and "
            "clear next steps for this request."
        ),
        "seeded_goal_general": (
            "Help the traveler with their request. The goal is achieved when the WestJet "
            "Travel Agent provides a relevant, policy-aligned response and clear next steps."
        ),
    },
    "fr": {
        "label": "French",
        "judge_language_instruction": "Use French for natural-language outputs.",
        "default_speak_to_agent_follow_up": "Oui, transferez-moi a un agent en direct",
        "default_knowledge_closure_message": "non, merci c'est tout",
        "default_flight_priority_yes": "oui",
        "default_flight_priority_no": "non",
        "default_vacation_flight_only": "vol seulement",
        "default_vacation_flight_and_hotel": "vol et hotel",
        "yes_tokens": {"oui", "ouais", "daccord", "d'accord", "affirmatif"},
        "no_tokens": {"non", "pas du tout", "negatif"},
        "vacation_flight_only_tokens": {
            "vol seulement",
            "seulement vol",
            "juste vol",
        },
        "vacation_flight_and_hotel_tokens": {
            "vol et hotel",
            "hotel et vol",
            "vol + hotel",
        },
        "greeting_aliases": [
            "Bonjour, je suis Ava, l'assistante virtuelle de WestJet. Comment puis-je vous aider aujourd'hui?",
            _DEFAULT_GREETING,
        ],
        "transcript_customer_speaker_tokens": {
            "client",
            "utilisateur",
            "voyageur",
            "visiteur",
            "inviter",
            "invite",
            "externe",
            "participantclient",
        },
        "transcript_agent_speaker_tokens": {
            "agent",
            "assistant",
            "conseiller",
            "systeme",
            "system",
            "robot",
            "automatisation",
            "flux",
            "assistantvirtuel",
        },
        "transcript_ignored_exact_messages": {
            "conversation terminee par l'utilisateur",
            "conversation terminee",
            "frappe",
            "frappe...",
        },
        "transcript_ignored_prefixes": (
            "conversation_id:",
            "participant_id:",
            "detected_intent:",
            "conversationid:",
            "participantid:",
            "intention_detectee:",
        ),
        "seeded_persona": (
            "Un voyageur qui contacte l'agent de voyage WestJet pour de l'aide liee aux vols. "
            "Il explique sa demande clairement et fournit les details requis au besoin."
        ),
        "seeded_goal_guideline": (
            "Aider le voyageur avec une demande sur les tarifs. Le but est atteint quand "
            "l'agent de voyage WestJet explique clairement qu'il ne fournit pas les details "
            "de prix ou de frais de bagages dans le clavardage, puis dirige le voyageur "
            "vers le site Web ou l'application WestJet pour les couts a jour."
        ),
        "seeded_goal_intent_template": (
            "Aider le voyageur avec une demande de type {intent_label}. Le but est atteint "
            "quand l'agent de voyage WestJet fournit une reponse pertinente, conforme aux "
            "politiques, et des prochaines etapes claires."
        ),
        "seeded_goal_general": (
            "Aider le voyageur avec sa demande. Le but est atteint quand l'agent de voyage "
            "WestJet fournit une reponse pertinente, conforme aux politiques, et des "
            "prochaines etapes claires."
        ),
    },
    "fr-CA": {
        "label": "French (Canada)",
        "judge_language_instruction": (
            "Use Canadian French (fr-CA) for natural-language outputs."
        ),
        "default_speak_to_agent_follow_up": "Oui, connectez-moi a un agent en direct",
        "default_knowledge_closure_message": "non merci, c'est tout",
        "default_flight_priority_yes": "oui",
        "default_flight_priority_no": "non",
        "default_vacation_flight_only": "vol seulement",
        "default_vacation_flight_and_hotel": "vol et hotel",
        "yes_tokens": {"oui", "ouais", "daccord", "d'accord", "affirmatif"},
        "no_tokens": {"non", "pas du tout", "negatif"},
        "vacation_flight_only_tokens": {
            "vol seulement",
            "seulement vol",
            "juste vol",
        },
        "vacation_flight_and_hotel_tokens": {
            "vol et hotel",
            "hotel et vol",
            "vol + hotel",
        },
        "greeting_aliases": [
            "Bonjour, je suis Ava, l'assistante virtuelle de WestJet. Comment puis-je vous aider aujourd'hui?",
            _DEFAULT_GREETING,
        ],
        "transcript_customer_speaker_tokens": {
            "client",
            "utilisateur",
            "voyageur",
            "visiteur",
            "invite",
            "externe",
            "participantclient",
        },
        "transcript_agent_speaker_tokens": {
            "agent",
            "assistant",
            "conseiller",
            "systeme",
            "system",
            "robot",
            "automatisation",
            "flux",
            "assistantvirtuel",
        },
        "transcript_ignored_exact_messages": {
            "conversation terminee par l'utilisateur",
            "conversation terminee",
            "frappe",
            "frappe...",
        },
        "transcript_ignored_prefixes": (
            "conversation_id:",
            "participant_id:",
            "detected_intent:",
            "conversationid:",
            "participantid:",
            "intention_detectee:",
        ),
        "seeded_persona": (
            "Un voyageur qui contacte l'agent de voyage WestJet pour de l'aide liee aux vols. "
            "Il explique sa demande clairement et fournit les details requis au besoin."
        ),
        "seeded_goal_guideline": (
            "Aider le voyageur avec une demande sur les tarifs. Le but est atteint quand "
            "l'agent de voyage WestJet explique clairement qu'il ne fournit pas les details "
            "de prix ou de frais de bagages dans le clavardage, puis dirige le voyageur "
            "vers le site Web ou l'application WestJet pour les couts a jour."
        ),
        "seeded_goal_intent_template": (
            "Aider le voyageur avec une demande de type {intent_label}. Le but est atteint "
            "quand l'agent de voyage WestJet fournit une reponse pertinente, conforme aux "
            "politiques, et des prochaines etapes claires."
        ),
        "seeded_goal_general": (
            "Aider le voyageur avec sa demande. Le but est atteint quand l'agent de voyage "
            "WestJet fournit une reponse pertinente, conforme aux politiques, et des "
            "prochaines etapes claires."
        ),
    },
    "es": {
        "label": "Spanish",
        "judge_language_instruction": "Use Spanish for natural-language outputs.",
        "default_speak_to_agent_follow_up": "Si, conectame con un agente en vivo",
        "default_knowledge_closure_message": "no, gracias, eso es todo",
        "default_flight_priority_yes": "si",
        "default_flight_priority_no": "no",
        "default_vacation_flight_only": "solo vuelo",
        "default_vacation_flight_and_hotel": "vuelo y hotel",
        "yes_tokens": {"si", "sí", "claro", "afirmativo"},
        "no_tokens": {"no", "negativo"},
        "vacation_flight_only_tokens": {
            "solo vuelo",
            "vuelo solamente",
            "solo el vuelo",
        },
        "vacation_flight_and_hotel_tokens": {
            "vuelo y hotel",
            "hotel y vuelo",
            "vuelo + hotel",
        },
        "greeting_aliases": [
            "Hola, soy Ava, la asistente virtual de WestJet. Como puedo ayudarte hoy?",
            _DEFAULT_GREETING,
        ],
        "transcript_customer_speaker_tokens": {
            "cliente",
            "usuario",
            "viajero",
            "visitante",
            "externo",
            "participantecliente",
        },
        "transcript_agent_speaker_tokens": {
            "agente",
            "asistente",
            "bot",
            "sistema",
            "automatizacion",
            "flujo",
            "asistentevirtual",
        },
        "transcript_ignored_exact_messages": {
            "conversacion finalizada por el usuario",
            "conversacion finalizada",
            "escribiendo",
            "escribiendo...",
        },
        "transcript_ignored_prefixes": (
            "conversation_id:",
            "participant_id:",
            "detected_intent:",
            "conversationid:",
            "participantid:",
            "intencion_detectada:",
        ),
        "seeded_persona": (
            "Un viajero que contacta al Agente de Viajes de WestJet para ayuda relacionada con vuelos. "
            "Explica su solicitud con claridad y entrega los datos necesarios cuando se los piden."
        ),
        "seeded_goal_guideline": (
            "Ayudar al viajero con una consulta de tarifas. El objetivo se cumple cuando el "
            "Agente de Viajes de WestJet explica claramente que no entrega precios ni costos "
            "especificos de equipaje por chat y redirige al viajero al sitio web o la app de "
            "WestJet para ver los costos actualizados."
        ),
        "seeded_goal_intent_template": (
            "Ayudar al viajero con una solicitud de tipo {intent_label}. El objetivo se cumple "
            "cuando el Agente de Viajes de WestJet entrega una respuesta relevante, alineada con "
            "las politicas, y pasos siguientes claros."
        ),
        "seeded_goal_general": (
            "Ayudar al viajero con su solicitud. El objetivo se cumple cuando el Agente de "
            "Viajes de WestJet entrega una respuesta relevante, alineada con las politicas, "
            "y pasos siguientes claros."
        ),
    },
}


def normalize_language_code(
    value: Optional[str],
    *,
    default: Optional[str] = "en",
    allow_none: bool = False,
) -> Optional[str]:
    """Normalize user-provided language code into canonical supported values."""
    raw = str(value or "").strip()
    if not raw:
        if allow_none:
            return None
        if default is None:
            raise ValueError("Language is required.")
        return normalize_language_code(default, default=None, allow_none=False)

    normalized = raw.replace("_", "-").strip()
    lowered = normalized.lower()
    canonical = _ALIAS_TO_CANONICAL.get(lowered)
    if canonical is None and lowered in _CANONICAL_CODES:
        canonical = lowered if lowered != "fr-ca" else "fr-CA"
    if canonical is None:
        supported = ", ".join(code for code, _ in SUPPORTED_LANGUAGE_OPTIONS)
        raise ValueError(
            f"Unsupported language '{value}'. Use one of: {supported}."
        )
    return canonical


def resolve_effective_language(
    *,
    runtime_override: Optional[str],
    suite_language: Optional[str],
    config_language: Optional[str],
    default: str = "en",
) -> str:
    """Resolve language precedence: runtime > suite > config > default."""
    for candidate in (runtime_override, suite_language, config_language, default):
        normalized = normalize_language_code(candidate, allow_none=True)
        if normalized:
            return normalized
    return "en"


def get_language_profile(language_code: Optional[str]) -> dict[str, Any]:
    """Return immutable language profile for the requested code."""
    canonical = normalize_language_code(language_code, default="en")
    return _LANGUAGE_PROFILES[canonical]


def get_language_label(language_code: Optional[str]) -> str:
    """Return display label for a language code."""
    profile = get_language_profile(language_code)
    return str(profile.get("label", "English"))


def normalize_evaluation_results_language(
    value: Optional[str],
    *,
    default: str = "inherit",
    allow_none: bool = False,
) -> Optional[str]:
    """Normalize evaluation/results language selector value.

    Supports explicit language codes and the special `inherit` mode.
    """
    raw = str(value or "").strip()
    if not raw:
        if allow_none:
            return None
        return default
    lowered = raw.lower()
    if lowered == "inherit":
        return "inherit"
    return normalize_language_code(raw, default="en")


def resolve_effective_evaluation_results_language(
    *,
    runtime_override: Optional[str],
    config_value: Optional[str],
    run_language: Optional[str],
) -> str:
    """Resolve evaluation/results language.

    Precedence:
    runtime override > config/env > inherit fallback to run language.
    """
    selected = normalize_evaluation_results_language(
        runtime_override,
        allow_none=True,
    )
    if selected is None:
        selected = normalize_evaluation_results_language(
            config_value,
            default="inherit",
        )
    if selected == "inherit":
        return normalize_language_code(run_language, default="en")
    return normalize_language_code(selected, default="en")
