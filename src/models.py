"""Pydantic data models for the GC Agent Regression Tester."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --- Test Suite and Scenarios ---


class TestScenario(BaseModel):
    """A single test case consisting of a persona, goal, and attempt count."""

    name: str
    persona: str
    goal: str
    first_message: Optional[str] = None  # If set, used as the first user message instead of LLM-generated
    expected_intent: Optional[str] = None  # If set, compare detected intent string against this value
    attempts: Optional[int] = None  # Uses default from config if omitted

    @field_validator("attempts")
    @classmethod
    def attempts_must_be_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("attempts must be a positive integer")
        return v


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

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: Optional[str] = None

    # Defaults
    default_attempts: int = 5
    max_turns: int = 10
    min_attempt_interval_seconds: int = 30
    response_timeout: int = 30  # seconds
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
    detected_intent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    turn_durations_seconds: list[float] = Field(default_factory=list)
    step_log: list[dict] = Field(default_factory=list)
    debug_frames: list[dict] = Field(default_factory=list)


class ScenarioResult(BaseModel):
    """Result of running all attempts for a single test scenario."""

    scenario_name: str
    attempts: int
    successes: int
    failures: int
    timeouts: int = 0
    success_rate: float
    is_regression: bool
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
    overall_success_rate: float
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
