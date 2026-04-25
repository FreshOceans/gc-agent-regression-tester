"""Gemma-powered test suite generation for operator supplied intents."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import ValidationError

from .config_loader import load_test_suite_from_string, print_test_suite
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .language_profiles import get_language_profile, normalize_language_code
from .models import TestScenario, TestSuite, normalize_gemma_single_model

SUPPORTED_SUITE_BUILDER_MODELS = ("gemma4:e4b", "gemma4:31b")
SUITE_BUILDER_DEFAULT_MODEL = "gemma4:e4b"
SUITE_BUILDER_DEFAULT_OUTPUT_DIR = Path("local_suites/generated")
SUITE_BUILDER_MODE_INTENT_TABLE = "intent_table"
SUITE_BUILDER_MODE_DESCRIPTION_ONLY = "description_only"
SUITE_BUILDER_GENERATION_MODES = (
    SUITE_BUILDER_MODE_INTENT_TABLE,
    SUITE_BUILDER_MODE_DESCRIPTION_ONLY,
)


@dataclass(frozen=True)
class SuiteBuilderIntentInput:
    """Operator-provided intent definition for suite generation."""

    id: str
    description: str
    examples: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SuiteBuilderRequest:
    """Validated request for a generated test suite."""

    suite_name: str
    model: str
    language: str
    scenario_count: int
    attempts: int
    user_turn_length: int
    include_language_selection: bool
    intents: list[SuiteBuilderIntentInput]


@dataclass(frozen=True)
class SuiteBuilderDescriptionRequest:
    """Validated request for description-to-intent-plan generation."""

    suite_name: str
    model: str
    language: str
    scenario_count: int
    attempts: int
    user_turn_length: int
    include_language_selection: bool
    suite_description: str
    inferred_intent_count: int


@dataclass(frozen=True)
class SuiteBuilderResult:
    """Generated suite with preview diagnostics."""

    suite: TestSuite
    suite_yaml: str
    warnings: list[str]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SuiteBuilderIntentPlanResult:
    """Editable inferred intent plan with diagnostics."""

    intents: list[SuiteBuilderIntentInput]
    warnings: list[str]
    diagnostics: dict[str, Any]


ChatCallable = Callable[[str, list[dict[str, str]]], str]


def _split_optional_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[\n,]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    return [item.strip() for item in raw_items if item and item.strip()]


def _coerce_int(value: Any, *, field_name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return parsed


def normalize_suite_builder_generation_mode(value: Any) -> str:
    """Normalize the Suite Builder generation mode with backward-compatible default."""

    normalized = str(value or "").strip().lower()
    if normalized in SUITE_BUILDER_GENERATION_MODES:
        return normalized
    return SUITE_BUILDER_MODE_INTENT_TABLE


def normalize_suite_builder_intents(raw_intents: list[dict[str, Any]]) -> list[SuiteBuilderIntentInput]:
    """Validate and normalize intent definitions."""

    normalized: list[SuiteBuilderIntentInput] = []
    seen: set[str] = set()
    for index, raw_intent in enumerate(raw_intents, start=1):
        intent_id = str(raw_intent.get("id") or raw_intent.get("intent") or "").strip()
        description = str(raw_intent.get("description") or "").strip()
        if not intent_id and not description:
            continue
        if not intent_id:
            raise ValueError(f"Intent row {index} is missing id.")
        if not description:
            raise ValueError(f"Intent '{intent_id}' is missing description.")
        key = intent_id.lower()
        if key in seen:
            raise ValueError(f"Duplicate intent id: {intent_id}")
        seen.add(key)
        normalized.append(
            SuiteBuilderIntentInput(
                id=intent_id,
                description=description,
                examples=_split_optional_list(raw_intent.get("examples")),
                avoid=_split_optional_list(raw_intent.get("avoid")),
            )
        )
    if not normalized:
        raise ValueError("At least one intent id and description is required.")
    return normalized


def parse_bulk_intents(text: str) -> list[dict[str, Any]]:
    """Parse bulk YAML/JSON intent input into raw intent dictionaries."""

    raw_text = str(text or "").strip()
    if not raw_text:
        return []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(raw_text)
    if parsed is None:
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get("intents"), list):
        parsed = parsed["intents"]
    elif isinstance(parsed, dict):
        parsed = [
            {"id": key, "description": value}
            for key, value in parsed.items()
            if isinstance(value, str)
        ]
    if not isinstance(parsed, list):
        raise ValueError("Bulk intent input must be a list or an object with an intents list.")
    output: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each bulk intent entry must be an object.")
        output.append(dict(item))
    return output


def build_suite_builder_request(
    *,
    suite_name: str,
    model: str,
    language: str,
    scenario_count: Any,
    attempts: Any,
    user_turn_length: Any,
    include_language_selection: bool,
    intents: list[dict[str, Any]],
) -> SuiteBuilderRequest:
    """Build a validated suite-builder request."""

    normalized_suite_name = str(suite_name or "").strip() or "Generated Test Suite"
    normalized_model = normalize_gemma_single_model(model or SUITE_BUILDER_DEFAULT_MODEL)
    normalized_language = normalize_language_code(language or "en", default="en")
    return SuiteBuilderRequest(
        suite_name=normalized_suite_name,
        model=normalized_model,
        language=normalized_language,
        scenario_count=_coerce_int(
            scenario_count,
            field_name="Scenario count",
            minimum=1,
            maximum=500,
        ),
        attempts=_coerce_int(
            attempts,
            field_name="Attempts per scenario",
            minimum=1,
            maximum=100,
        ),
        user_turn_length=_coerce_int(
            user_turn_length,
            field_name="User-turn length",
            minimum=1,
            maximum=10,
        ),
        include_language_selection=bool(include_language_selection),
        intents=normalize_suite_builder_intents(intents),
    )


def build_suite_builder_description_request(
    *,
    suite_name: str,
    model: str,
    language: str,
    scenario_count: Any,
    attempts: Any,
    user_turn_length: Any,
    include_language_selection: bool,
    suite_description: str,
    inferred_intent_count: Any,
) -> SuiteBuilderDescriptionRequest:
    """Build a validated description-only suite-builder request."""

    normalized_suite_name = str(suite_name or "").strip() or "Generated Test Suite"
    normalized_description = str(suite_description or "").strip()
    if not normalized_description:
        raise ValueError("Suite description is required for Description Only mode.")
    normalized_model = normalize_gemma_single_model(model or SUITE_BUILDER_DEFAULT_MODEL)
    normalized_language = normalize_language_code(language or "en", default="en")
    normalized_scenario_count = _coerce_int(
        scenario_count,
        field_name="Scenario count",
        minimum=1,
        maximum=500,
    )
    normalized_inferred_count = _coerce_int(
        inferred_intent_count,
        field_name="Inferred intent count",
        minimum=1,
        maximum=50,
    )
    if normalized_inferred_count > normalized_scenario_count:
        raise ValueError("Inferred intent count cannot exceed total scenarios.")
    return SuiteBuilderDescriptionRequest(
        suite_name=normalized_suite_name,
        model=normalized_model,
        language=normalized_language,
        scenario_count=normalized_scenario_count,
        attempts=_coerce_int(
            attempts,
            field_name="Attempts per scenario",
            minimum=1,
            maximum=100,
        ),
        user_turn_length=_coerce_int(
            user_turn_length,
            field_name="User-turn length",
            minimum=1,
            maximum=10,
        ),
        include_language_selection=bool(include_language_selection),
        suite_description=normalized_description,
        inferred_intent_count=normalized_inferred_count,
    )


def distribute_scenarios(total: int, intent_count: int) -> list[int]:
    """Evenly distribute total scenarios across intents, remainder in input order."""

    if intent_count < 1:
        raise ValueError("intent_count must be at least 1")
    base = total // intent_count
    remainder = total % intent_count
    return [base + (1 if index < remainder else 0) for index in range(intent_count)]


def _language_selection_message(language: str) -> str:
    normalized = normalize_language_code(language, default="en")
    if normalized == "es":
        return "espanol"
    if normalized in {"fr", "fr-CA"}:
        return "francais"
    return "english"


def _language_label(language: str) -> str:
    return str(get_language_profile(language).get("label") or language)


def _default_persona(language: str) -> str:
    profile = get_language_profile(language)
    return str(
        profile.get("seeded_persona")
        or "A customer contacting a virtual assistant. They explain their request naturally."
    )


def _goal_for_intent(intent: SuiteBuilderIntentInput) -> str:
    return (
        f"Help the customer with a {intent.id} request. The intent means: "
        f"{intent.description}. The expected detected intent is {intent.id}."
    )


def _coverage_for_index(index: int) -> str:
    styles = ["clear", "paraphrased", "vague", "typo/noisy", "edge-case"]
    return styles[index % len(styles)]


def _build_prompt(
    request: SuiteBuilderRequest,
    intent: SuiteBuilderIntentInput,
    count: int,
    *,
    retry_error: Optional[str] = None,
) -> list[dict[str, str]]:
    language_label = _language_label(request.language)
    examples = intent.examples or ["none provided"]
    avoid = intent.avoid or ["none provided"]
    retry_clause = f"\nPrevious output problem: {retry_error}\nFix it." if retry_error else ""
    system = (
        "Generate test-suite user utterances for a bot regression harness.\n"
        "Return exactly one JSON object and no markdown.\n"
        "JSON shape: {\"scenarios\":[{\"first_message\":\"...\",\"scripted_user_turns\":[\"...\"]}]}\n"
        "Rules:\n"
        f"- Generate exactly {count} scenarios.\n"
        f"- Each scenario must have exactly {request.user_turn_length} total user turns.\n"
        "- first_message is turn 1. scripted_user_turns contains the remaining turns.\n"
        "- For one-turn scenarios, scripted_user_turns must be an empty array.\n"
        f"- User messages must be in {language_label}.\n"
        "- Do not translate or change the intent id.\n"
        "- Keep messages realistic and concise.\n"
        "- Use balanced coverage: clear, paraphrased, vague, typo/noisy, and edge-case phrasing.\n"
        "- Avoid pricing claims, internal labels, JSON inside messages, and assistant/agent text.\n"
    )
    user = (
        f"Intent id: {intent.id}\n"
        f"Intent description: {intent.description}\n"
        f"Operator examples: {json.dumps(examples, ensure_ascii=False)}\n"
        f"Avoid or near misses: {json.dumps(avoid, ensure_ascii=False)}\n"
        f"Requested scenarios: {count}\n"
        f"Total user turns per scenario: {request.user_turn_length}\n"
        f"Coverage sequence to rotate: {[ _coverage_for_index(i) for i in range(count) ]}\n"
        f"{retry_clause}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize_inferred_intent_id(value: Any) -> str:
    """Convert model-proposed intent IDs to conservative English-style snake_case."""

    raw = str(value or "").strip().lower()
    ascii_text = (
        unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        raise ValueError("inferred intent id is blank after snake_case normalization")
    return normalized


def _build_intent_plan_prompt(
    request: SuiteBuilderDescriptionRequest,
    *,
    retry_error: Optional[str] = None,
) -> list[dict[str, str]]:
    language_label = _language_label(request.language)
    retry_clause = f"\nPrevious output problem: {retry_error}\nFix it." if retry_error else ""
    system = (
        "Infer a draft intent plan for a bot regression test-suite builder.\n"
        "Return exactly one JSON object and no markdown.\n"
        "JSON shape: {\"intents\":[{\"id\":\"english_snake_case\",\"description\":\"...\",\"examples\":[\"...\"],\"avoid\":[\"...\"]}]}\n"
        "Rules:\n"
        f"- Generate exactly {request.inferred_intent_count} intents.\n"
        "- Intent ids must be English snake_case and must not be translated.\n"
        "- Intent ids must be stable bot-style detected_intent values, not sentence fragments.\n"
        "- Descriptions must be concise English operator guidance.\n"
        f"- Examples should be realistic user utterances in {language_label}.\n"
        "- Avoid notes should capture near misses or topics that should not map to the intent.\n"
        "- Do not invent tool names, API fields, or internal labels.\n"
    )
    user = (
        f"Suite name: {request.suite_name}\n"
        f"Suite description:\n{request.suite_description}\n"
        f"Generation language for future utterances: {language_label}\n"
        f"Total scenarios planned: {request.scenario_count}\n"
        f"Attempts per scenario: {request.attempts}\n"
        f"User-turn length: {request.user_turn_length}\n"
        f"Include language-selection pre-step: {request.include_language_selection}\n"
        f"Requested inferred intent count: {request.inferred_intent_count}\n"
        f"{retry_clause}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_intent_plan_payload(
    response_text: str,
    client: JudgeLLMClient,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    extracted = client._extract_json(response_text)
    payload = json.loads(extracted)
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    intents = payload.get("intents")
    if not isinstance(intents, list):
        raise ValueError("model response must include intents list")
    if len(intents) != expected_count:
        raise ValueError(f"model returned {len(intents)} intents; expected {expected_count}")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(intents, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"inferred intent {index} must be an object")
        normalized_id = _normalize_inferred_intent_id(item.get("id"))
        description = str(item.get("description") or "").strip()
        if not description:
            raise ValueError(f"inferred intent '{normalized_id}' is missing description")
        output.append(
            {
                "id": normalized_id,
                "description": description,
                "examples": item.get("examples", []),
                "avoid": item.get("avoid", []),
            }
        )
    return output


def infer_intents_from_description(
    request: SuiteBuilderDescriptionRequest,
    *,
    ollama_base_url: str,
    timeout: int = 120,
    chat_callable: Optional[ChatCallable] = None,
) -> SuiteBuilderIntentPlanResult:
    """Infer editable intent rows from a plain-language suite description."""

    client = JudgeLLMClient(
        base_url=ollama_base_url,
        model=request.model,
        timeout=timeout,
    )
    diagnostics: dict[str, Any] = {
        "mode": SUITE_BUILDER_MODE_DESCRIPTION_ONLY,
        "model": request.model,
        "language": request.language,
        "scenario_count": request.scenario_count,
        "attempts": request.attempts,
        "user_turn_length": request.user_turn_length,
        "include_language_selection": request.include_language_selection,
        "inferred_intent_count": request.inferred_intent_count,
        "retry_used": False,
    }
    last_error: Optional[str] = None
    for attempt in range(2):
        messages = _build_intent_plan_prompt(
            request,
            retry_error=last_error if attempt > 0 else None,
        )
        try:
            response_text = (
                chat_callable(request.model, messages)
                if chat_callable is not None
                else client._call_chat(messages, operation="generate_user_message")
            )
            raw_intents = _parse_intent_plan_payload(
                response_text,
                client,
                expected_count=request.inferred_intent_count,
            )
            intents = normalize_suite_builder_intents(raw_intents)
            diagnostics["retry_used"] = attempt > 0
            diagnostics["intent_ids"] = [intent.id for intent in intents]
            warnings = [
                "Review inferred intent IDs against actual bot detected_intent values before running."
            ]
            return SuiteBuilderIntentPlanResult(
                intents=intents,
                warnings=warnings,
                diagnostics=diagnostics,
            )
        except (JudgeLLMError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            diagnostics["retry_used"] = attempt == 0

    raise ValueError(
        "Could not infer a valid intent plan from the suite description. "
        f"Last model error: {last_error}"
    )


def _parse_generation_payload(response_text: str, client: JudgeLLMClient) -> list[dict[str, Any]]:
    extracted = client._extract_json(response_text)  # Reuse the repo's tolerant JSON extraction.
    payload = json.loads(extracted)
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("model response must include scenarios list")
    return [item for item in scenarios if isinstance(item, dict)]


def _scenario_from_model_item(
    *,
    item: dict[str, Any],
    intent: SuiteBuilderIntentInput,
    request: SuiteBuilderRequest,
    scenario_number: int,
) -> TestScenario:
    first_message = str(item.get("first_message") or "").strip()
    if not first_message:
        raise ValueError("scenario is missing first_message")
    raw_turns = item.get("scripted_user_turns") or []
    if not isinstance(raw_turns, list):
        raise ValueError("scripted_user_turns must be a list")
    scripted_turns = [str(turn).strip() for turn in raw_turns if str(turn or "").strip()]
    expected_scripted_count = request.user_turn_length - 1
    if len(scripted_turns) != expected_scripted_count:
        raise ValueError(
            f"expected {expected_scripted_count} scripted turns, got {len(scripted_turns)}"
        )

    scenario_data: dict[str, Any] = {
        "name": f"{intent.id} - Generated {scenario_number:02d}",
        "persona": _default_persona(request.language),
        "goal": _goal_for_intent(intent),
        "first_message": first_message,
        "attempts": request.attempts,
    }
    if request.include_language_selection:
        scenario_data["language_selection_message"] = _language_selection_message(
            request.language
        )
    if request.user_turn_length == 1:
        scenario_data["expected_intent"] = intent.id
    else:
        scenario_data["scripted_user_turns"] = scripted_turns
        scenario_data["scripted_final_expected_intent"] = intent.id
    return TestScenario.model_validate(scenario_data)


def _fallback_model_item(
    intent: SuiteBuilderIntentInput,
    request: SuiteBuilderRequest,
    index: int,
) -> dict[str, Any]:
    seed = intent.examples[index % len(intent.examples)] if intent.examples else intent.description
    first_message = f"I need help with this: {seed}"
    turns = [
        f"More detail for {intent.id}: {intent.description}"
        for _ in range(max(0, request.user_turn_length - 1))
    ]
    return {"first_message": first_message, "scripted_user_turns": turns}


def generate_suite_with_gemma(
    request: SuiteBuilderRequest,
    *,
    ollama_base_url: str,
    timeout: int = 120,
    chat_callable: Optional[ChatCallable] = None,
) -> SuiteBuilderResult:
    """Generate, validate, and serialize a suite from a validated request."""

    counts = distribute_scenarios(request.scenario_count, len(request.intents))
    client = JudgeLLMClient(
        base_url=ollama_base_url,
        model=request.model,
        timeout=timeout,
    )
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {
        "model": request.model,
        "language": request.language,
        "scenario_count_requested": request.scenario_count,
        "attempts": request.attempts,
        "user_turn_length": request.user_turn_length,
        "include_language_selection": request.include_language_selection,
        "intent_batches": [],
    }
    generated_scenarios: list[TestScenario] = []
    scenario_number = 1

    for intent, count in zip(request.intents, counts):
        if count <= 0:
            continue
        batch_diag: dict[str, Any] = {
            "intent_id": intent.id,
            "requested": count,
            "generated": 0,
            "retry_used": False,
            "fallback_used": 0,
        }
        items: list[dict[str, Any]] = []
        last_error: Optional[str] = None
        for attempt in range(2):
            messages = _build_prompt(
                request,
                intent,
                count,
                retry_error=last_error if attempt > 0 else None,
            )
            try:
                response_text = (
                    chat_callable(request.model, messages)
                    if chat_callable is not None
                    else client._call_chat(messages, operation="generate_user_message")
                )
                items = _parse_generation_payload(response_text, client)
                if len(items) >= count:
                    break
                last_error = f"model returned {len(items)} scenarios; expected {count}"
            except (JudgeLLMError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
            batch_diag["retry_used"] = attempt == 0
        if last_error and len(items) < count:
            warnings.append(
                f"Intent '{intent.id}' generated short or malformed output; fallback filled missing scenarios. Last error: {last_error}"
            )

        accepted = 0
        item_index = 0
        while accepted < count:
            if item_index < len(items):
                item = items[item_index]
                item_index += 1
            else:
                item = _fallback_model_item(intent, request, accepted)
                batch_diag["fallback_used"] += 1
            try:
                scenario = _scenario_from_model_item(
                    item=item,
                    intent=intent,
                    request=request,
                    scenario_number=scenario_number,
                )
            except (ValidationError, ValueError) as exc:
                warnings.append(
                    f"Intent '{intent.id}' scenario {accepted + 1} was invalid and replaced with fallback: {exc}"
                )
                item = _fallback_model_item(intent, request, accepted)
                batch_diag["fallback_used"] += 1
                scenario = _scenario_from_model_item(
                    item=item,
                    intent=intent,
                    request=request,
                    scenario_number=scenario_number,
                )
            generated_scenarios.append(scenario)
            scenario_number += 1
            accepted += 1
        batch_diag["generated"] = accepted
        diagnostics["intent_batches"].append(batch_diag)

    suite = TestSuite(
        name=request.suite_name,
        language=request.language,
        scenarios=generated_scenarios,
    )
    suite_yaml = print_test_suite(suite, format="yaml")
    # Validate the exact serialized payload operators will preview/save.
    suite = load_test_suite_from_string(suite_yaml, "yaml")
    suite_yaml = print_test_suite(suite, format="yaml")
    diagnostics["scenario_count_generated"] = len(suite.scenarios)
    return SuiteBuilderResult(
        suite=suite,
        suite_yaml=suite_yaml,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def safe_suite_filename(suite_name: str) -> str:
    """Convert a suite name to a safe local YAML filename stem."""

    normalized = str(suite_name or "generated_suite").strip().lower().replace(" ", "_")
    safe = "".join(ch for ch in normalized if ch.isalnum() or ch in {"_", "-"})[:80]
    return safe or "generated_suite"


def save_generated_suite_yaml(
    suite_yaml: str,
    *,
    output_dir: str | Path = SUITE_BUILDER_DEFAULT_OUTPUT_DIR,
) -> Path:
    """Validate and save generated suite YAML without overwriting existing files."""

    suite = load_test_suite_from_string(suite_yaml, "yaml")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = safe_suite_filename(suite.name)
    destination = root / f"{stem}.yaml"
    suffix = 2
    while destination.exists():
        destination = root / f"{stem}_{suffix}.yaml"
        suffix += 1
    destination.write_text(print_test_suite(suite, format="yaml"), encoding="utf-8")
    return destination
