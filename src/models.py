"""Pydantic data models for the GC Agent Regression Tester."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class TestScenario(BaseModel):
    """A single test case consisting of a persona, goal, and attempt count."""

    name: str
    persona: str
    goal: str
    first_message: Optional[str] = None  # If set, used as the first user message instead of LLM-generated
    expected_intent: Optional[str] = None  # If set, compare detected intent string against this value
    intent_follow_up_user_message: Optional[str] = None  # Optional deterministic follow-up user reply for intent flows
    attempts: Optional[int] = None  # Uses default from config if omitted
    tool_validation: Optional[ToolValidationConfig] = None

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


class TestSuite(BaseModel):
    """A collection of test scenarios that defines the full regression test."""

    name: str
    scenarios: list[TestScenario] = Field(min_length=1)


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
    tool_attribute_keys: list[str] = Field(
        default_factory=lambda: ["rth_tool_events", "tool_events"]
    )
    tool_marker_prefixes: list[str] = Field(default_factory=lambda: ["tool_event:"])

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: Optional[str] = None

    # Defaults
    default_attempts: int = 5
    max_turns: int = 10
    min_attempt_interval_seconds: int = 15
    response_timeout: int = 90  # seconds
    success_threshold: float = 0.8  # 80%
    expected_greeting: str = "Hi, I'm Ava, WestJet's virtual assistant. How may I help you today?"

    @field_validator("min_attempt_interval_seconds")
    @classmethod
    def min_attempt_interval_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("min_attempt_interval_seconds must be non-negative")
        return v

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

    @field_validator("tool_attribute_keys", "tool_marker_prefixes", mode="before")
    @classmethod
    def parse_list_like_config(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("value must be a list or comma-separated string")

    @field_validator("tool_attribute_keys", "tool_marker_prefixes")
    @classmethod
    def normalize_list_like_config(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value if item and item.strip()]
        if not normalized:
            raise ValueError("configuration list must include at least one value")
        return normalized


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
    tool_events: list["ToolEvent"] = Field(default_factory=list)
    tool_validation_result: Optional["ToolValidationResult"] = None


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


class ScenarioResult(BaseModel):
    """Result of running all attempts for a single test scenario."""

    scenario_name: str
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
    attempt_number: Optional[int] = None
    success: Optional[bool] = None
    success_rate: Optional[float] = None
    message: str
    duration_seconds: Optional[float] = None
    attempt_result: Optional[AttemptResult] = None  # Full attempt data for live results
    planned_attempts: Optional[int] = None
    completed_attempts: Optional[int] = None
