"""Judge execution routing for Gemma-optimized single and dual modes."""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import (
    JUDGE_EXECUTION_MODE_DUAL_STRICT_FALLBACK,
    JUDGE_EXECUTION_MODE_SINGLE,
    AppConfig,
    GoalEvaluation,
    JourneyValidationResult,
    JudgeDiagnosticEntry,
    normalize_gemma_single_model,
    normalize_judge_execution_mode,
)

_DUAL_PRIMARY_MODEL = "gemma4:e4b"
_DUAL_FALLBACK_MODEL = "gemma4:31b"
_FALLBACK_CONFIDENCE_THRESHOLD = 0.70


@dataclass(frozen=True)
class JudgeExecutionSettings:
    """Resolved judge execution settings for one run surface."""

    mode: str
    single_model: str
    custom_model_override: Optional[str] = None


def resolve_judge_execution_settings(
    config: AppConfig,
    *,
    analytics: bool = False,
) -> JudgeExecutionSettings:
    """Resolve run-surface judge execution settings from app config."""

    if analytics:
        mode = normalize_judge_execution_mode(config.analytics_judge_execution_mode)
        single_model = normalize_gemma_single_model(config.analytics_judge_single_model)
        custom_model_override = (
            str(config.analytics_journey_judge_model or "").strip() or None
        )
    else:
        mode = normalize_judge_execution_mode(config.judge_execution_mode)
        single_model = normalize_gemma_single_model(config.judge_single_model)
        custom_model_override = str(config.ollama_model or "").strip() or None

    return JudgeExecutionSettings(
        mode=mode,
        single_model=single_model,
        custom_model_override=custom_model_override,
    )


def resolve_effective_judge_model_name(
    config: AppConfig,
    *,
    analytics: bool = False,
) -> str:
    """Resolve the primary model name that will be used for a run surface."""

    settings = resolve_judge_execution_settings(config, analytics=analytics)
    if settings.mode == JUDGE_EXECUTION_MODE_DUAL_STRICT_FALLBACK:
        return _DUAL_PRIMARY_MODEL
    return settings.custom_model_override or settings.single_model


def build_judge_execution_client(
    config: AppConfig,
    *,
    analytics: bool = False,
) -> "JudgeExecutionClient":
    """Create a judge execution client for one run surface."""

    settings = resolve_judge_execution_settings(config, analytics=analytics)
    return JudgeExecutionClient(
        base_url=config.ollama_base_url,
        settings=settings,
        timeout=config.response_timeout,
    )


class JudgeExecutionClient:
    """Routes judge calls through Gemma-optimized single or dual execution."""

    def __init__(
        self,
        *,
        base_url: str,
        settings: JudgeExecutionSettings,
        timeout: int,
    ):
        self.base_url = str(base_url or "").rstrip("/")
        self.settings = settings
        self.timeout = int(timeout)
        self.primary_client = JudgeLLMClient(
            base_url=self.base_url,
            model=self._primary_model_name(),
            timeout=self.timeout,
        )
        self.fallback_client: Optional[JudgeLLMClient] = None
        if self.settings.mode == JUDGE_EXECUTION_MODE_DUAL_STRICT_FALLBACK:
            self.fallback_client = JudgeLLMClient(
                base_url=self.base_url,
                model=_DUAL_FALLBACK_MODEL,
                timeout=self.timeout,
            )
        self._attempt_diagnostics: ContextVar[list[JudgeDiagnosticEntry]] = ContextVar(
            "judge_execution_attempt_diagnostics",
            default=[],
        )
        self._pending_status_messages: ContextVar[list[str]] = ContextVar(
            "judge_execution_pending_status_messages",
            default=[],
        )

    @property
    def model(self) -> str:
        """Expose the primary/effective model name for compatibility."""
        return self.primary_client.model

    def _primary_model_name(self) -> str:
        if self.settings.mode == JUDGE_EXECUTION_MODE_DUAL_STRICT_FALLBACK:
            return _DUAL_PRIMARY_MODEL
        return self.settings.custom_model_override or self.settings.single_model

    def reset_attempt_diagnostics(self) -> None:
        """Reset per-attempt routing diagnostics."""
        self._attempt_diagnostics.set([])
        self._pending_status_messages.set([])

    def consume_attempt_diagnostics(self) -> list[JudgeDiagnosticEntry]:
        """Return and clear per-attempt routing diagnostics."""
        current = list(self._attempt_diagnostics.get([]))
        self._attempt_diagnostics.set([])
        return current

    def consume_pending_status_messages(self) -> list[str]:
        """Return and clear pending live status messages."""
        current = list(self._pending_status_messages.get([]))
        self._pending_status_messages.set([])
        return current

    def verify_connection(self) -> None:
        """Verify Ollama connectivity for all configured judge models."""
        self.primary_client.verify_connection()
        if self.fallback_client is not None:
            self.fallback_client.verify_connection()

    def warm_up(
        self,
        prompt: Optional[str] = None,
        language_code: str = "en",
    ) -> str:
        return self._execute_single_operation(
            "warm_up",
            lambda client: client.warm_up(prompt=prompt, language_code=language_code),
        )

    def generate_user_message(
        self,
        persona: str,
        goal: str,
        conversation_history: list[Any],
        language_code: str = "en",
    ) -> str:
        return self._execute_single_operation(
            "generate_user_message",
            lambda client: client.generate_user_message(
                persona=persona,
                goal=goal,
                conversation_history=conversation_history,
                language_code=language_code,
            ),
        )

    def should_continue(
        self,
        persona: str,
        goal: str,
        conversation_history: list[Any],
        language_code: str = "en",
    ):
        return self._execute_single_operation(
            "should_continue",
            lambda client: client.should_continue(
                persona=persona,
                goal=goal,
                conversation_history=conversation_history,
                language_code=language_code,
            ),
        )

    def evaluate_goal(
        self,
        persona: str,
        goal: str,
        conversation_history: list[Any],
        language_code: str = "en",
    ) -> GoalEvaluation:
        return self._execute_eval_operation(
            "evaluate_goal",
            lambda client: client.evaluate_goal(
                persona=persona,
                goal=goal,
                conversation_history=conversation_history,
                language_code=language_code,
            ),
            validator=self._validate_goal_evaluation_result,
            confidence_getter=None,
        )

    def classify_primary_category(
        self,
        *,
        first_message: str,
        categories: list[dict],
        language_code: str = "en",
    ) -> dict:
        return self._execute_eval_operation(
            "classify_primary_category",
            lambda client: client.classify_primary_category(
                first_message=first_message,
                categories=categories,
                language_code=language_code,
            ),
            validator=self._validate_primary_category_result,
            confidence_getter=self._extract_confidence,
        )

    def infer_containment(
        self,
        *,
        conversation_history: list[Any],
        language_code: str = "en",
    ) -> dict:
        return self._execute_eval_operation(
            "infer_containment",
            lambda client: client.infer_containment(
                conversation_history=conversation_history,
                language_code=language_code,
            ),
            validator=self._validate_containment_result,
            confidence_getter=self._extract_confidence,
        )

    def evaluate_journey(
        self,
        *,
        persona: str,
        goal: str,
        expected_category: Optional[str],
        path_rubric: Optional[str],
        category_rubric: Optional[str],
        conversation_history: list[Any],
        language_code: str = "en",
        known_contained: Optional[bool] = None,
    ) -> JourneyValidationResult:
        return self._execute_eval_operation(
            "evaluate_journey",
            lambda client: client.evaluate_journey(
                persona=persona,
                goal=goal,
                expected_category=expected_category,
                path_rubric=path_rubric,
                category_rubric=category_rubric,
                conversation_history=conversation_history,
                language_code=language_code,
                known_contained=known_contained,
            ),
            validator=lambda result: self._validate_journey_result(
                result,
                expected_category=expected_category,
            ),
            confidence_getter=self._extract_confidence,
        )

    def extract_conversation_id(
        self,
        conversation_history: list[Any],
        language_code: str = "en",
    ) -> Optional[str]:
        return self._execute_single_operation(
            "extract_conversation_id",
            lambda client: client.extract_conversation_id(
                conversation_history=conversation_history,
                language_code=language_code,
            ),
        )

    def _execute_single_operation(
        self,
        operation_name: str,
        invoke: Callable[[JudgeLLMClient], Any],
    ) -> Any:
        started_at = time.monotonic()
        try:
            return invoke(self.primary_client)
        finally:
            duration_ms = (time.monotonic() - started_at) * 1000.0
            self._record_diagnostic(
                JudgeDiagnosticEntry(
                    operation_name=operation_name,
                    mode=self.settings.mode,
                    primary_model=self.primary_client.model,
                    fallback_model=self.fallback_client.model if self.fallback_client else None,
                    fallback_used=False,
                    duration_ms=duration_ms,
                )
            )

    def _execute_eval_operation(
        self,
        operation_name: str,
        invoke: Callable[[JudgeLLMClient], Any],
        *,
        validator: Callable[[Any], Optional[str]],
        confidence_getter: Optional[Callable[[Any], Optional[float]]],
    ) -> Any:
        started_at = time.monotonic()
        fallback_used = False
        fallback_reason: Optional[str] = None
        primary_confidence: Optional[float] = None
        fallback_confidence: Optional[float] = None
        try:
            try:
                result = invoke(self.primary_client)
            except JudgeLLMError:
                if self.fallback_client is None:
                    raise
                fallback_used = True
                fallback_reason = "primary_request_failure"
                self._queue_status_message(operation_name, fallback_reason)
                result = self._invoke_fallback(operation_name, invoke)
                fallback_confidence = (
                    confidence_getter(result) if confidence_getter is not None else None
                )
                invalid_fallback_reason = validator(result)
                if invalid_fallback_reason is not None:
                    raise JudgeLLMError(
                        f"{operation_name} fallback result unusable: {invalid_fallback_reason}"
                    )
                return result

            primary_confidence = (
                confidence_getter(result) if confidence_getter is not None else None
            )
            invalid_primary_reason = validator(result)
            if invalid_primary_reason is None or self.fallback_client is None:
                return result

            fallback_used = True
            fallback_reason = invalid_primary_reason
            self._queue_status_message(operation_name, fallback_reason)
            fallback_result = self._invoke_fallback(operation_name, invoke)
            fallback_confidence = (
                confidence_getter(fallback_result)
                if confidence_getter is not None
                else None
            )
            invalid_fallback_reason = validator(fallback_result)
            if invalid_fallback_reason is not None:
                raise JudgeLLMError(
                    f"{operation_name} fallback result unusable: {invalid_fallback_reason}"
                )
            return fallback_result
        finally:
            duration_ms = (time.monotonic() - started_at) * 1000.0
            self._record_diagnostic(
                JudgeDiagnosticEntry(
                    operation_name=operation_name,
                    mode=self.settings.mode,
                    primary_model=self.primary_client.model,
                    fallback_model=self.fallback_client.model if self.fallback_client else None,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                    primary_confidence=primary_confidence,
                    fallback_confidence=fallback_confidence,
                    duration_ms=duration_ms,
                )
            )

    def _invoke_fallback(
        self,
        operation_name: str,
        invoke: Callable[[JudgeLLMClient], Any],
    ) -> Any:
        if self.fallback_client is None:
            raise JudgeLLMError(
                f"{operation_name} fallback requested, but no fallback model is configured"
            )
        try:
            return invoke(self.fallback_client)
        except JudgeLLMError as exc:
            raise JudgeLLMError(
                f"{operation_name} fallback failed on model '{self.fallback_client.model}': {exc}"
            ) from exc

    def _record_diagnostic(self, entry: JudgeDiagnosticEntry) -> None:
        current = list(self._attempt_diagnostics.get([]))
        current.append(entry)
        self._attempt_diagnostics.set(current)

    def _queue_status_message(self, operation_name: str, fallback_reason: str) -> None:
        if self.fallback_client is None:
            return
        current = list(self._pending_status_messages.get([]))
        current.append(
            (
                f"Judge fallback triggered for {operation_name}: "
                f"{fallback_reason} on {self.primary_client.model}; "
                f"escalating to {self.fallback_client.model}."
            )
        )
        self._pending_status_messages.set(current)

    def _extract_confidence(self, result: Any) -> Optional[float]:
        raw_value = None
        if hasattr(result, "confidence"):
            raw_value = getattr(result, "confidence")
        elif isinstance(result, dict):
            raw_value = result.get("confidence")
        try:
            return float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            return None

    def _validate_goal_evaluation_result(self, result: Any) -> Optional[str]:
        if not isinstance(result, GoalEvaluation):
            return "schema_invalid"
        if not isinstance(result.success, bool):
            return "required_field_missing"
        if not str(result.explanation or "").strip():
            return "required_field_missing"
        return None

    def _validate_primary_category_result(self, result: Any) -> Optional[str]:
        if not isinstance(result, dict):
            return "schema_invalid"
        category = str(result.get("category") or "").strip().lower()
        if not category or category == "unknown":
            return "unknown_result"
        confidence = self._extract_confidence(result)
        if confidence is None:
            return "confidence_missing"
        if confidence < _FALLBACK_CONFIDENCE_THRESHOLD:
            return "low_confidence"
        return None

    def _validate_containment_result(self, result: Any) -> Optional[str]:
        if not isinstance(result, dict):
            return "schema_invalid"
        if not isinstance(result.get("contained"), bool):
            return "unknown_result"
        confidence = self._extract_confidence(result)
        if confidence is None:
            return "confidence_missing"
        if confidence < _FALLBACK_CONFIDENCE_THRESHOLD:
            return "low_confidence"
        return None

    def _validate_journey_result(
        self,
        result: Any,
        *,
        expected_category: Optional[str],
    ) -> Optional[str]:
        if not isinstance(result, JourneyValidationResult):
            return "schema_invalid"
        if result.contained is None:
            return "unknown_result"
        if expected_category and result.category_match is None:
            return "required_field_missing"
        confidence = self._extract_confidence(result)
        if confidence is None:
            return "confidence_missing"
        if confidence < _FALLBACK_CONFIDENCE_THRESHOLD:
            return "low_confidence"
        return None
