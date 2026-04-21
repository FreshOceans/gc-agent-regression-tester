"""Pydantic data models for the GC Agent Regression Tester."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .journey_mode import normalize_category_strategy, normalize_harness_mode
from .judging_options import (
    normalize_explanation_mode,
    normalize_judging_strictness,
    normalize_objective_profile,
)
from .language_profiles import (
    normalize_evaluation_results_language,
    normalize_language_code,
)

ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS = "client_credentials"
ANALYTICS_AUTH_MODE_MANUAL_BEARER = "manual_bearer"
_ANALYTICS_AUTH_MODES = {
    ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
    ANALYTICS_AUTH_MODE_MANUAL_BEARER,
}


def normalize_analytics_auth_mode(
    value: Optional[str],
    *,
    default: str = ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
) -> str:
    """Normalize AJR auth mode values."""
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        raw = default
    if raw not in _ANALYTICS_AUTH_MODES:
        raise ValueError(
            "analytics auth mode must be one of: client_credentials, manual_bearer"
        )
    return raw


# --- Test Suite and Scenarios ---


class ToolRuleExpression(BaseModel):
    """Boolean expression tree for tool validation rules."""

    tool: Optional[str] = None
    min_count: int = 1
    status_in: Optional[list[str]] = None
    all: Optional[list["ToolRuleExpression"]] = None
    any: Optional[list["ToolRuleExpression"]] = None
    not_rule: Optional["ToolRuleExpression"] = Field(
        default=None,
        alias="not",
        serialization_alias="not",
    )
    in_order: Optional[list["ToolRuleExpression"]] = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("tool")
    @classmethod
    def normalize_tool_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None

    @field_validator("min_count")
    @classmethod
    def min_count_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_count must be at least 1")
        return value

    @field_validator("status_in", mode="before")
    @classmethod
    def parse_status_in(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item).strip() for item in value]
        else:
            raise ValueError("status_in must be a list or comma-separated string")
        normalized = [item.lower() for item in values if item]
        return normalized or None

    @model_validator(mode="after")
    def validate_expression_shape(self):
        branches = [
            self.tool is not None,
            self.all is not None,
            self.any is not None,
            self.not_rule is not None,
            self.in_order is not None,
        ]
        active = sum(1 for enabled in branches if enabled)
        if active != 1:
            raise ValueError(
                "Tool rule expression must contain exactly one of: "
                "tool, all, any, not, in_order"
            )

        if self.all is not None and len(self.all) == 0:
            raise ValueError("all must include at least one nested rule")
        if self.any is not None and len(self.any) == 0:
            raise ValueError("any must include at least one nested rule")
        if self.in_order is not None and len(self.in_order) == 0:
            raise ValueError("in_order must include at least one nested rule")

        if self.tool is None:
            # min_count/status_in are leaf-only constraints.
            if self.min_count != 1:
                raise ValueError("min_count can only be used with a tool leaf rule")
            if self.status_in:
                raise ValueError("status_in can only be used with a tool leaf rule")

        return self


class ToolValidationConfig(BaseModel):
    """Scenario-level tool validation configuration."""

    loose_rule: ToolRuleExpression
    strict_rule: Optional[ToolRuleExpression] = None

    model_config = ConfigDict(extra="forbid")


class PrimaryCategoryConfig(BaseModel):
    """Configurable primary category definition for journey regression."""

    name: str
    keywords: list[str] = Field(default_factory=list)
    rubric: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("primary category name cannot be blank")
        return normalized

    @field_validator("keywords", mode="before")
    @classmethod
    def parse_keywords(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",")]
            return [item for item in values if item]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("keywords must be a list or comma-separated string")

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, value: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for token in value:
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped


class JourneyValidationConfig(BaseModel):
    """Scenario-level full-journey validation controls."""

    require_containment: bool = True
    require_fulfillment: bool = True
    path_rubric: Optional[str] = None
    category_rubric_override: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TestScenario(BaseModel):
    """A single test case consisting of a persona, goal, and attempt count."""

    name: str
    persona: str
    goal: str
    first_message: Optional[str] = None  # If set, used as the first user message instead of LLM-generated
    language_selection_message: Optional[str] = None  # Optional pre-step message sent after greeting and before first_message
    expected_intent: Optional[str] = None  # If set, compare detected intent string against this value
    intent_follow_up_user_message: Optional[str] = None  # Optional deterministic follow-up user reply for intent flows
    attempts: Optional[int] = None  # Uses default from config if omitted
    tool_validation: Optional[ToolValidationConfig] = None
    journey_category: Optional[str] = None
    journey_validation: Optional[JourneyValidationConfig] = None

    @field_validator("attempts")
    @classmethod
    def attempts_must_be_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("attempts must be a positive integer")
        return v

    @field_validator("intent_follow_up_user_message")
    @classmethod
    def intent_follow_up_user_message_must_not_be_blank(cls, v):
        if v is None:
            return v
        normalized = v.strip()
        if not normalized:
            raise ValueError("intent_follow_up_user_message must not be blank")
        return normalized

    @field_validator("language_selection_message")
    @classmethod
    def language_selection_message_must_not_be_blank(cls, v):
        if v is None:
            return v
        normalized = v.strip()
        if not normalized:
            raise ValueError("language_selection_message must not be blank")
        return normalized

    @field_validator("journey_category")
    @classmethod
    def normalize_journey_category(cls, v):
        if v is None:
            return None
        normalized = v.strip().lower().replace(" ", "_")
        return normalized or None


class TestSuite(BaseModel):
    """A collection of test scenarios that defines the full regression test."""

    name: str
    language: Optional[str] = None
    harness_mode: Optional[str] = None
    primary_categories: Optional[list[PrimaryCategoryConfig]] = None
    scenarios: list[TestScenario] = Field(min_length=1)

    @field_validator("language")
    @classmethod
    def normalize_suite_language(cls, value: Optional[str]) -> Optional[str]:
        return normalize_language_code(value, allow_none=True)

    @field_validator("harness_mode")
    @classmethod
    def normalize_suite_harness_mode(cls, value: Optional[str]) -> Optional[str]:
        return normalize_harness_mode(value, allow_none=True)


# --- Configuration ---


class AppConfig(BaseModel):
    """Application configuration with defaults."""

    # Genesys Cloud
    gc_region: Optional[str] = None
    gc_deployment_id: Optional[str] = None
    gc_client_id: Optional[str] = None
    gc_client_secret: Optional[str] = None
    intent_attribute_name: str = "detected_intent"
    debug_capture_frames: bool = False
    debug_capture_frame_limit: int = 8
    judge_warmup_enabled: bool = True
    step_skip_timeout_seconds: int = 90
    knowledge_mode_timeout_seconds: int = 120
    history_dir: str = ".gc_tester_history"
    history_max_runs: int = 50
    history_full_json_runs: int = 20
    history_gzip_runs: int = 20
    transcript_import_enabled: bool = False
    transcript_import_time: str = "02:00"
    transcript_import_timezone: Optional[str] = None
    transcript_import_max_ids: int = 50
    transcript_import_filter_json: str = "{}"
    transcript_import_dir: str = ".gc_tester_history/transcript_imports"
    transcript_url_allowlist: list[str] = Field(
        default_factory=lambda: [
            "pure.cloud",
            "mypurecloud.com",
        ]
    )
    transcript_url_timeout_seconds: int = 30
    transcript_url_max_bytes: int = 5_000_000
    tool_attribute_keys: list[str] = Field(
        default_factory=lambda: ["rth_tool_events", "tool_events"]
    )
    tool_marker_prefixes: list[str] = Field(default_factory=lambda: ["tool_event:"])
    harness_mode: str = "standard"
    journey_category_strategy: str = "rules_first"
    journey_primary_categories_json: str = ""
    journey_primary_categories_file: Optional[str] = None
    judging_mechanics_enabled: bool = False
    judging_objective_profile: str = "blended"
    judging_strictness: str = "balanced"
    judging_tolerance: float = 0.0
    judging_containment_weight: float = 0.35
    judging_fulfillment_weight: float = 0.45
    judging_path_weight: float = 0.20
    judging_explanation_mode: str = "standard"
    journey_dashboard_enabled: bool = False
    journey_taxonomy_overrides_json: str = ""
    journey_taxonomy_overrides_file: Optional[str] = None
    analytics_journey_enabled: bool = False
    analytics_journey_auth_mode: str = ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS
    analytics_journey_details_page_size_cap: int = 100
    analytics_journey_default_page_size: int = 50
    analytics_journey_default_max_conversations: int = 150
    analytics_journey_policy_map_json: str = ""
    analytics_journey_policy_map_file: Optional[str] = None
    analytics_journey_judge_model: Optional[str] = None
    analytics_journey_default_language_filter: Optional[str] = None
    analytics_journey_artifact_dir: str = ".gc_tester_history/analytics_journey"
    attempt_parallel_enabled: bool = True
    max_parallel_attempt_workers: int = 2
    adaptive_attempt_pacing_enabled: bool = True
    web_auth_enabled: bool = False
    web_auth_username: Optional[str] = None
    web_auth_password: Optional[str] = None
    web_session_idle_minutes: int = 30
    language: str = "en"
    evaluation_results_language: str = "inherit"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: Optional[str] = None

    # Defaults
    default_attempts: int = 5
    max_turns: int = 10
    min_attempt_interval_seconds: float = 5.0
    response_timeout: int = 90  # seconds
    success_threshold: float = 0.8  # 80%
    expected_greeting: str = "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"

    @field_validator("min_attempt_interval_seconds")
    @classmethod
    def min_attempt_interval_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("min_attempt_interval_seconds must be non-negative")
        return float(v)

    @field_validator("debug_capture_frame_limit")
    @classmethod
    def debug_capture_frame_limit_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("debug_capture_frame_limit must be at least 1")
        return v

    @field_validator("step_skip_timeout_seconds")
    @classmethod
    def step_skip_timeout_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("step_skip_timeout_seconds must be at least 1")
        return v

    @field_validator("knowledge_mode_timeout_seconds")
    @classmethod
    def knowledge_mode_timeout_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("knowledge_mode_timeout_seconds must be at least 1")
        return v

    @field_validator("history_max_runs")
    @classmethod
    def history_max_runs_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("history_max_runs must be at least 1")
        return v

    @field_validator("history_full_json_runs", "history_gzip_runs")
    @classmethod
    def history_compaction_windows_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("history compaction window values must be non-negative")
        return v

    @field_validator("transcript_import_max_ids")
    @classmethod
    def transcript_import_max_ids_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("transcript_import_max_ids must be at least 1")
        return v

    @field_validator(
        "analytics_journey_default_page_size",
        "analytics_journey_default_max_conversations",
        "analytics_journey_details_page_size_cap",
    )
    @classmethod
    def analytics_journey_limits_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("analytics journey limits must be at least 1")
        return v

    @field_validator("analytics_journey_auth_mode")
    @classmethod
    def normalize_analytics_journey_auth_mode(cls, value: str) -> str:
        return normalize_analytics_auth_mode(value)

    @field_validator("max_parallel_attempt_workers")
    @classmethod
    def max_parallel_attempt_workers_range(cls, v):
        parsed = int(v)
        if parsed < 1:
            parsed = 1
        if parsed > 3:
            parsed = 3
        return parsed

    @field_validator("web_session_idle_minutes")
    @classmethod
    def web_session_idle_minutes_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("web_session_idle_minutes must be at least 1")
        return v

    @field_validator("transcript_url_timeout_seconds")
    @classmethod
    def transcript_url_timeout_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("transcript_url_timeout_seconds must be at least 1")
        return v

    @field_validator("transcript_url_max_bytes")
    @classmethod
    def transcript_url_max_bytes_must_be_positive(cls, v):
        if v < 1024:
            raise ValueError("transcript_url_max_bytes must be at least 1024")
        return v

    @field_validator("transcript_import_time")
    @classmethod
    def transcript_import_time_must_be_hhmm(cls, v):
        raw = (v or "").strip()
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError("transcript_import_time must use HH:MM format")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError as e:
            raise ValueError("transcript_import_time must use HH:MM format") from e
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError("transcript_import_time must be a valid 24-hour time")
        return f"{hour:02d}:{minute:02d}"

    @field_validator(
        "tool_attribute_keys",
        "tool_marker_prefixes",
        "transcript_url_allowlist",
        mode="before",
    )
    @classmethod
    def parse_list_like_config(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("value must be a list or comma-separated string")

    @field_validator(
        "tool_attribute_keys",
        "tool_marker_prefixes",
        "transcript_url_allowlist",
    )
    @classmethod
    def normalize_list_like_config(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value if item and item.strip()]
        if not normalized:
            raise ValueError("configuration list must include at least one value")
        return normalized

    @field_validator("language")
    @classmethod
    def normalize_app_language(cls, value: str) -> str:
        return normalize_language_code(value, default="en")

    @field_validator("analytics_journey_default_language_filter")
    @classmethod
    def normalize_analytics_language_filter(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        normalized = normalize_language_code(value, allow_none=True)
        return str(normalized) if normalized else None

    @field_validator("evaluation_results_language")
    @classmethod
    def normalize_app_evaluation_results_language(cls, value: str) -> str:
        return str(
            normalize_evaluation_results_language(
                value,
                default="inherit",
            )
        )

    @field_validator("harness_mode")
    @classmethod
    def normalize_harness_mode_config(cls, value: str) -> str:
        return normalize_harness_mode(value)

    @field_validator("journey_category_strategy")
    @classmethod
    def normalize_journey_category_strategy(cls, value: str) -> str:
        return normalize_category_strategy(value)

    @field_validator("judging_objective_profile")
    @classmethod
    def normalize_judging_profile(cls, value: str) -> str:
        return normalize_objective_profile(value, default="blended")

    @field_validator("judging_strictness")
    @classmethod
    def normalize_judging_strictness_value(cls, value: str) -> str:
        return normalize_judging_strictness(value, default="balanced")

    @field_validator("judging_explanation_mode")
    @classmethod
    def normalize_judging_explanation_mode_value(cls, value: str) -> str:
        return normalize_explanation_mode(value, default="standard")

    @field_validator(
        "judging_tolerance",
        "judging_containment_weight",
        "judging_fulfillment_weight",
        "judging_path_weight",
    )
    @classmethod
    def judging_numeric_values_must_be_non_negative(cls, value: float) -> float:
        parsed = float(value)
        if parsed < 0:
            raise ValueError("judging numeric values must be non-negative")
        return parsed


# --- Conversation and Messages ---


class MessageRole(str, Enum):
    """Role of a message sender in a conversation."""

    AGENT = "agent"
    USER = "user"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


class ContinueDecision(BaseModel):
    """Decision on whether to continue the conversation."""

    should_continue: bool
    goal_achieved: Optional[bool] = None  # Set when should_continue is False
    explanation: Optional[str] = None


class GoalEvaluation(BaseModel):
    """Evaluation of whether the goal was achieved."""

    success: bool
    explanation: str


# --- Results ---


class TimeoutDiagnostics(BaseModel):
    """Structured timeout telemetry captured for debugging failed/skipped attempts."""

    timeout_class: str
    step_name: Optional[str] = None
    step_timeout_seconds: Optional[float] = None
    configured_timeout_seconds: Optional[float] = None
    step_skip_timeout_seconds: Optional[float] = None
    greeting_wait_base_seconds: Optional[float] = None
    greeting_wait_buffer_seconds: Optional[float] = None
    greeting_wait_timeout_seconds: Optional[float] = None
    expected_greeting_configured: bool = False
    language_pre_step_active: Optional[bool] = None
    elapsed_attempt_seconds: Optional[float] = None
    conversation_total_messages: int = 0
    conversation_user_messages: int = 0
    conversation_agent_messages: int = 0
    greeting_detected: Optional[bool] = None
    conversation_id: Optional[str] = None
    participant_id: Optional[str] = None
    conversation_id_candidates: list[str] = Field(default_factory=list)
    attempt_parallel_enabled: Optional[bool] = None
    max_parallel_attempt_workers: Optional[int] = None
    min_attempt_interval_seconds: Optional[float] = None

    @field_validator("timeout_class")
    @classmethod
    def normalize_timeout_class(cls, value: str) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("timeout_class must not be blank")
        return normalized

    @field_validator("conversation_id_candidates", mode="before")
    @classmethod
    def normalize_conversation_id_candidates(cls, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            deduped: list[str] = []
            seen: set[str] = set()
            for item in value:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                deduped.append(text)
            return deduped
        return []


class FailureDiagnostics(BaseModel):
    """Structured non-timeout failure telemetry captured for debugging."""

    failure_class: str
    gate_step: Optional[str] = None
    matched_pattern_id: Optional[str] = None
    terminal_message_excerpt: Optional[str] = None
    elapsed_attempt_seconds: Optional[float] = None
    conversation_total_messages: int = 0
    conversation_user_messages: int = 0
    conversation_agent_messages: int = 0
    conversation_id: Optional[str] = None
    participant_id: Optional[str] = None
    conversation_id_candidates: list[str] = Field(default_factory=list)
    attempt_parallel_enabled: Optional[bool] = None
    max_parallel_attempt_workers: Optional[int] = None
    min_attempt_interval_seconds: Optional[float] = None

    @field_validator("failure_class")
    @classmethod
    def normalize_failure_class(cls, value: str) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("failure_class must not be blank")
        return normalized

    @field_validator("terminal_message_excerpt", mode="before")
    @classmethod
    def normalize_terminal_message_excerpt(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = " ".join(text.split())
        return text[:240]

    @field_validator("conversation_id_candidates", mode="before")
    @classmethod
    def normalize_conversation_id_candidates(cls, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            deduped: list[str] = []
            seen: set[str] = set()
            for item in value:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                deduped.append(text)
            return deduped
        return []


class AttemptResult(BaseModel):
    """Result of a single conversation attempt."""

    attempt_number: int
    success: bool
    conversation: list[Message]
    explanation: str
    error: Optional[str] = None  # Set if attempt failed due to error
    timed_out: bool = False
    skipped: bool = False
    detected_intent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    turn_durations_seconds: list[float] = Field(default_factory=list)
    step_log: list[dict] = Field(default_factory=list)
    debug_frames: list[dict] = Field(default_factory=list)
    timeout_diagnostics: Optional["TimeoutDiagnostics"] = None
    failure_diagnostics: Optional["FailureDiagnostics"] = None
    tool_events: list["ToolEvent"] = Field(default_factory=list)
    tool_validation_result: Optional["ToolValidationResult"] = None
    journey_validation_result: Optional["JourneyValidationResult"] = None
    judging_mechanics_result: Optional["JudgingMechanicsResult"] = None
    analytics_journey_result: Optional["AnalyticsJourneyResult"] = None
    journey_taxonomy_label: Optional[str] = None


class ToolEvent(BaseModel):
    """Normalized tool event captured from participant attributes or response markers."""

    name: str
    status: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: str
    raw_payload: Optional[dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def normalize_event_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("tool event name cannot be empty")
        return normalized

    @field_validator("status", mode="before")
    @classmethod
    def normalize_event_status(cls, value):
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    @field_validator("source")
    @classmethod
    def normalize_event_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("tool event source cannot be empty")
        return normalized


class ToolValidationResult(BaseModel):
    """Outcome details for loose and strict tool validation checks."""

    loose_pass: bool
    strict_pass: Optional[bool] = None
    missing_signal: bool = False
    loose_fail_reasons: list[str] = Field(default_factory=list)
    strict_fail_reasons: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)
    order_violations: list[str] = Field(default_factory=list)
    matched_tools: list[str] = Field(default_factory=list)


class JourneyValidationResult(BaseModel):
    """Outcome details for full-journey validation checks."""

    category_match: Optional[bool] = None
    fulfilled: bool = False
    path_correct: bool = False
    contained: Optional[bool] = None
    expected_category: Optional[str] = None
    actual_category: Optional[str] = None
    containment_source: str = "unknown"
    confidence: Optional[float] = None
    explanation: str = ""
    failure_reasons: list[str] = Field(default_factory=list)

    @field_validator("containment_source")
    @classmethod
    def normalize_containment_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        return normalized or "unknown"


class JudgingMechanicsResult(BaseModel):
    """Per-attempt scoring metadata for configurable judging mechanics."""

    enabled: bool = False
    objective_profile: str = "blended"
    strictness: str = "balanced"
    tolerance: float = 0.0
    threshold: float = 0.0
    score: float = 0.0
    passed_threshold: bool = True
    hard_gate_passed: bool = True
    final_gate_passed: bool = True
    explanation_mode: str = "standard"
    criteria: dict[str, float] = Field(default_factory=dict)


class AnalyticsJourneyResult(BaseModel):
    """Per-attempt analytics journey gate breakdown."""

    conversation_id: Optional[str] = None
    category: Optional[str] = None
    classification_source: str = "unknown"
    classification_confidence: Optional[float] = None
    policy_key: Optional[str] = None
    expected_auth_behavior: str = "optional"
    observed_auth: Optional[bool] = None
    auth_gate: Optional[bool] = None
    expected_transfer_behavior: str = "optional"
    observed_transfer: Optional[bool] = None
    transfer_gate: Optional[bool] = None
    category_gate: Optional[bool] = None
    journey_quality_gate: Optional[bool] = None
    enrichment_used: bool = False
    skipped_reason: Optional[str] = None
    evidence_notes: list[str] = Field(default_factory=list)

    @field_validator("classification_source", "expected_auth_behavior", "expected_transfer_behavior")
    @classmethod
    def normalize_gate_value(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        return normalized or "unknown"


class JourneyTaxonomyRollup(BaseModel):
    """Fixed-label journey taxonomy rollup row."""

    label: str
    count: int
    rate: float
    delta: Optional[int] = None


class AdaptivePacingAdjustment(BaseModel):
    """One adaptive pacing interval adjustment during a run."""

    attempt_window_end: int
    window_size: int
    signal_count: int
    signal_rate: float
    from_interval_seconds: float
    to_interval_seconds: float
    reason: str

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("reason must not be blank")
        return normalized


class AnalyticsRunDiagnosticsRequest(BaseModel):
    """Sanitized request snapshot for analytics journey runs."""

    bot_flow_id: str
    interval: str
    page_size: int
    max_conversations: int
    auth_mode: str = ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS
    divisions_count: int = 0
    language_filter: Optional[str] = None
    extra_query_param_keys: list[str] = Field(default_factory=list)


class AnalyticsRunDiagnosticsSummary(BaseModel):
    """Aggregated counters and timings for one analytics journey run."""

    pages_fetched: int = 0
    rows_scanned: int = 0
    unique_conversations: int = 0
    evaluated: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    retry_count: int = 0
    http_429_count: int = 0
    http_5xx_count: int = 0
    fetch_duration_seconds: float = 0.0
    evaluation_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0


class AnalyticsRunDiagnosticsTimelineEntry(BaseModel):
    """One timestamped stage event for an analytics journey run."""

    timestamp: datetime
    elapsed_seconds: float = 0.0
    stage: str
    message: str
    page_number: Optional[int] = None
    conversation_id: Optional[str] = None
    duration_ms: Optional[float] = None
    details: Optional[dict[str, Any]] = None

    @field_validator("stage")
    @classmethod
    def normalize_stage(cls, value: str) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("stage must not be blank")
        return normalized


class AnalyticsRunDiagnostics(BaseModel):
    """Run-level observability payload for analytics journey execution."""

    request: AnalyticsRunDiagnosticsRequest
    summary: AnalyticsRunDiagnosticsSummary
    timeline: list[AnalyticsRunDiagnosticsTimelineEntry] = Field(default_factory=list)
    dropped_timeline_entries: int = 0


class ScenarioResult(BaseModel):
    """Result of running all attempts for a single test scenario."""

    scenario_name: str
    expected_intent: Optional[str] = None
    attempts: int
    successes: int
    failures: int
    timeouts: int = 0
    skipped: int = 0
    success_rate: float
    is_regression: bool
    tool_validated_attempts: int = 0
    tool_loose_passes: int = 0
    tool_strict_passes: int = 0
    tool_missing_signal_count: int = 0
    tool_order_mismatch_count: int = 0
    tool_loose_pass_rate: float = 0.0
    tool_strict_pass_rate: float = 0.0
    journey_validated_attempts: int = 0
    journey_passes: int = 0
    journey_contained_passes: int = 0
    journey_fulfillment_passes: int = 0
    journey_path_passes: int = 0
    journey_category_match_passes: int = 0
    judging_scored_attempts: int = 0
    judging_threshold_passes: int = 0
    judging_threshold_failures: int = 0
    judging_average_score: float = 0.0
    analytics_evaluated_attempts: int = 0
    analytics_gate_passes: int = 0
    analytics_skipped_unknown: int = 0
    attempt_results: list[AttemptResult]


class TestReport(BaseModel):
    """Aggregated output of running an entire test suite."""

    suite_name: str
    timestamp: datetime
    duration_seconds: float
    scenario_results: list[ScenarioResult]
    overall_attempts: int
    overall_successes: int
    overall_failures: int
    overall_timeouts: int = 0
    overall_skipped: int = 0
    overall_success_rate: float
    overall_tool_validated_attempts: int = 0
    overall_tool_loose_passes: int = 0
    overall_tool_strict_passes: int = 0
    overall_tool_missing_signal_count: int = 0
    overall_tool_order_mismatch_count: int = 0
    overall_tool_loose_pass_rate: float = 0.0
    overall_tool_strict_pass_rate: float = 0.0
    overall_journey_validated_attempts: int = 0
    overall_journey_passes: int = 0
    overall_journey_contained_passes: int = 0
    overall_journey_fulfillment_passes: int = 0
    overall_journey_path_passes: int = 0
    overall_journey_category_match_passes: int = 0
    overall_judging_scored_attempts: int = 0
    overall_judging_threshold_passes: int = 0
    overall_judging_threshold_failures: int = 0
    overall_judging_average_score: float = 0.0
    overall_analytics_evaluated_attempts: int = 0
    overall_analytics_gate_passes: int = 0
    overall_analytics_skipped_unknown: int = 0
    analytics_run_diagnostics: Optional[AnalyticsRunDiagnostics] = None
    journey_taxonomy_rollups: list[JourneyTaxonomyRollup] = Field(default_factory=list)
    adaptive_attempt_pacing_enabled: bool = False
    adaptive_attempt_pacing_base_interval_seconds: Optional[float] = None
    adaptive_attempt_pacing_final_interval_seconds: Optional[float] = None
    adaptive_attempt_pacing_adjustment_count: int = 0
    adaptive_attempt_pacing_adjustments: list[AdaptivePacingAdjustment] = Field(
        default_factory=list
    )
    stopped_by_user: bool = False
    stop_mode: Optional[str] = None
    stop_requested_at: Optional[datetime] = None
    stop_finalized_at: Optional[datetime] = None
    force_finalized: bool = False
    has_regressions: bool
    regression_threshold: float


# --- Progress Events ---


class ProgressEventType(str, Enum):
    """Types of progress events emitted during test execution."""

    SUITE_STARTED = "suite_started"
    SCENARIO_STARTED = "scenario_started"
    ATTEMPT_STARTED = "attempt_started"
    ATTEMPT_STATUS = "attempt_status"
    ATTEMPT_COMPLETED = "attempt_completed"
    SCENARIO_COMPLETED = "scenario_completed"
    SUITE_COMPLETED = "suite_completed"


class ProgressEvent(BaseModel):
    """A progress event emitted during test execution."""

    event_type: ProgressEventType
    emitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    suite_name: Optional[str] = None
    scenario_name: Optional[str] = None
    expected_intent: Optional[str] = None
    attempt_number: Optional[int] = None
    success: Optional[bool] = None
    success_rate: Optional[float] = None
    message: str
    duration_seconds: Optional[float] = None
    attempt_result: Optional[AttemptResult] = None  # Full attempt data for live results
    planned_attempts: Optional[int] = None
    completed_attempts: Optional[int] = None
