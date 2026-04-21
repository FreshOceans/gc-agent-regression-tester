"""Conversation Runner for executing single test attempts."""

import asyncio
import contextlib
import inspect
import json
import random
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from .genesys_conversations_client import (
    GenesysConversationsClient,
    GenesysConversationsError,
)
from .journey_mode import HARNESS_JOURNEY, normalize_harness_mode
from .journey_regression import (
    infer_containment_from_payload_metadata,
    normalize_category_name,
)
from .judging_mechanics import (
    format_mechanics_summary,
    resolve_judging_mechanics_config,
    score_goal_evaluation,
    score_journey_evaluation,
)
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .language_profiles import get_language_profile, normalize_language_code
from .models import (
    AttemptResult,
    GoalEvaluation,
    JourneyValidationResult,
    JudgingMechanicsResult,
    Message,
    MessageRole,
    TimeoutDiagnostics,
    TestScenario,
    ToolEvent,
    ToolValidationResult,
)
from .tool_validation import (
    dedupe_tool_events,
    evaluate_tool_validation,
    parse_tool_events_from_attribute_map,
    parse_tool_events_from_markers,
)
from .web_messaging_client import WebMessagingClient, WebMessagingError


class StepTimeoutError(Exception):
    """Raised when an attempt step exceeds the configured step timeout."""

    def __init__(self, step_name: str, timeout_seconds: float):
        self.step_name = step_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Step '{step_name}' exceeded step timeout after {timeout_seconds:.0f}s"
        )


class StopRequestedError(Exception):
    """Raised when a user stop request should interrupt the current attempt."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        super().__init__(f"Stop requested while running step '{step_name}'")


class GreetingGateTimeoutError(TimeoutError):
    """Raised when expected greeting is not observed before main user utterance."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            "Expected greeting was not received before sending first scenario user message"
        )


class ConversationRunner:
    """Manages a single conversation attempt between the Judge LLM and a Genesys Cloud agent.

    Creates a new WebMessagingClient per attempt for test isolation,
    drives the conversation loop via the Judge LLM, and evaluates the goal.
    """

    def __init__(self, judge: JudgeLLMClient, web_msg_config: dict, max_turns: int = 20):
        """Initialize with judge client and web messaging configuration.

        Args:
            judge: The JudgeLLMClient instance for generating messages and evaluating goals.
            web_msg_config: Dict with keys: region, deployment_id, timeout.
            max_turns: Maximum number of user-agent message pairs before stopping.
        """
        self.judge = judge
        self.web_msg_config = web_msg_config
        self.max_turns = max_turns
        self._active_status_callback: Optional[Callable[[str], None]] = None
        self._active_step_log: Optional[list[dict]] = None
        self._active_timeout_context: Optional[dict] = None

    def _emit_attempt_status(self, message: str) -> None:
        entry = {
            "timestamp": self._now_utc().isoformat(),
            "message": message,
        }
        if self._active_step_log is not None:
            self._active_step_log.append(entry)
        callback = self._active_status_callback
        if callback is None:
            return
        try:
            callback(message)
        except Exception:
            # Status telemetry must not impact test execution.
            pass

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _step_timeout_seconds(self) -> float:
        raw_value = self.web_msg_config.get("step_skip_timeout_seconds", 90)
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            parsed = 90.0
        return max(1.0, parsed)

    def _record_step_timeout_window(self, *, step_name: str, timeout_seconds: float) -> None:
        context = self._active_timeout_context
        if context is None:
            return
        context["last_step_name"] = step_name
        context["last_step_timeout_seconds"] = timeout_seconds

    def _set_greeting_timeout_context(
        self,
        *,
        expected_greeting_configured: bool,
        language_pre_step_active: bool,
        base_wait_timeout_seconds: float,
        wait_buffer_seconds: float,
        greeting_wait_timeout_seconds: float,
    ) -> None:
        context = self._active_timeout_context
        if context is None:
            return
        context["greeting"] = {
            "expected_greeting_configured": bool(expected_greeting_configured),
            "language_pre_step_active": bool(language_pre_step_active),
            "greeting_wait_base_seconds": float(base_wait_timeout_seconds),
            "greeting_wait_buffer_seconds": float(wait_buffer_seconds),
            "greeting_wait_timeout_seconds": float(greeting_wait_timeout_seconds),
        }

    def _classify_timeout_class(
        self,
        *,
        error: Exception,
        last_step_name: Optional[str],
    ) -> str:
        message = str(error or "").lower()
        step = str(last_step_name or "").lower()
        if "greeting" in step:
            return "greeting_gate"
        if "judge llm" in step or "ollama" in message:
            return "judge_timeout"
        if "welcome" in step:
            return "welcome_timeout"
        if "agent response" in step:
            return "response_timeout"
        return "timeout"

    def _build_timeout_diagnostics(
        self,
        *,
        timeout_class: str,
        attempt_start_monotonic: float,
        conversation: list[Message],
        client: WebMessagingClient,
        step_name: Optional[str] = None,
        step_timeout_seconds: Optional[float] = None,
    ) -> TimeoutDiagnostics:
        context = self._active_timeout_context or {}
        greeting_context = context.get("greeting") if isinstance(context.get("greeting"), dict) else {}
        last_step_name = str(context.get("last_step_name") or "").strip() or None
        last_step_timeout = context.get("last_step_timeout_seconds")
        effective_step_name = step_name or last_step_name
        effective_step_timeout = (
            step_timeout_seconds
            if step_timeout_seconds is not None
            else (
                float(last_step_timeout)
                if isinstance(last_step_timeout, (int, float))
                else None
            )
        )

        agent_messages = [
            msg
            for msg in conversation
            if msg.role == MessageRole.AGENT
        ]
        user_messages = [
            msg
            for msg in conversation
            if msg.role == MessageRole.USER
        ]
        greeting_detected = any(self._is_expected_greeting(msg.content) for msg in agent_messages)

        conversation_id = getattr(client, "conversation_id", None)
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            conversation_id = None
        participant_id = getattr(client, "participant_id", None)
        if not isinstance(participant_id, str) or not participant_id.strip():
            participant_id = None

        candidate_ids: list[str] = []
        get_candidates = getattr(client, "get_conversation_id_candidates", None)
        if callable(get_candidates):
            try:
                raw_candidates = get_candidates()
                if inspect.isawaitable(raw_candidates):
                    close_fn = getattr(raw_candidates, "close", None)
                    if callable(close_fn):
                        close_fn()
                    raw_candidates = None
                if isinstance(raw_candidates, list):
                    candidate_ids = [str(item).strip() for item in raw_candidates if str(item).strip()]
            except Exception:
                candidate_ids = []

        return TimeoutDiagnostics(
            timeout_class=timeout_class,
            step_name=effective_step_name,
            step_timeout_seconds=effective_step_timeout,
            configured_timeout_seconds=float(self.web_msg_config.get("timeout", 30)),
            step_skip_timeout_seconds=self._step_timeout_seconds(),
            greeting_wait_base_seconds=(
                float(greeting_context.get("greeting_wait_base_seconds"))
                if greeting_context.get("greeting_wait_base_seconds") is not None
                else None
            ),
            greeting_wait_buffer_seconds=(
                float(greeting_context.get("greeting_wait_buffer_seconds"))
                if greeting_context.get("greeting_wait_buffer_seconds") is not None
                else None
            ),
            greeting_wait_timeout_seconds=(
                float(greeting_context.get("greeting_wait_timeout_seconds"))
                if greeting_context.get("greeting_wait_timeout_seconds") is not None
                else None
            ),
            expected_greeting_configured=bool(
                greeting_context.get("expected_greeting_configured", False)
            ),
            language_pre_step_active=(
                bool(greeting_context["language_pre_step_active"])
                if "language_pre_step_active" in greeting_context
                else None
            ),
            elapsed_attempt_seconds=max(0.0, time.monotonic() - attempt_start_monotonic),
            conversation_total_messages=len(conversation),
            conversation_user_messages=len(user_messages),
            conversation_agent_messages=len(agent_messages),
            greeting_detected=greeting_detected,
            conversation_id=conversation_id,
            participant_id=participant_id,
            conversation_id_candidates=candidate_ids,
            attempt_parallel_enabled=bool(self.web_msg_config.get("attempt_parallel_enabled", False)),
            max_parallel_attempt_workers=(
                int(self.web_msg_config.get("max_parallel_attempt_workers"))
                if self.web_msg_config.get("max_parallel_attempt_workers") is not None
                else None
            ),
            min_attempt_interval_seconds=(
                float(self.web_msg_config.get("min_attempt_interval_seconds"))
                if self.web_msg_config.get("min_attempt_interval_seconds") is not None
                else None
            ),
        )

    def _is_stop_requested(self) -> bool:
        stop_event = self.web_msg_config.get("stop_event")
        return bool(
            stop_event is not None
            and hasattr(stop_event, "is_set")
            and stop_event.is_set()
        )

    async def _await_step(
        self,
        step_name: str,
        awaitable,
        timeout_override_seconds: Optional[float] = None,
    ):
        timeout_seconds = (
            max(0.1, float(timeout_override_seconds))
            if timeout_override_seconds is not None
            else self._step_timeout_seconds()
        )
        self._record_step_timeout_window(
            step_name=step_name,
            timeout_seconds=timeout_seconds,
        )
        deadline = time.monotonic() + timeout_seconds
        task = asyncio.create_task(awaitable)
        try:
            while True:
                if self._is_stop_requested():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    raise StopRequestedError(step_name)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    raise StepTimeoutError(step_name, timeout_seconds)

                poll_timeout = min(1.0, remaining)
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout=poll_timeout)
                except asyncio.TimeoutError:
                    # Distinguish "poll expired" from "task ended with TimeoutError".
                    if task.done():
                        return await task
                    continue
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _run_judge_step(self, step_name: str, func, *args, **kwargs):
        return await self._await_step(
            step_name,
            asyncio.to_thread(func, *args, **kwargs),
        )

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        normalized = normalized.replace("’", "'")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _conversation_language_code(self) -> str:
        raw = self.web_msg_config.get("language")
        return normalize_language_code(raw, default="en")

    def _evaluation_results_language_code(self) -> str:
        raw = self.web_msg_config.get("evaluation_results_language")
        if raw:
            return normalize_language_code(raw, default="en")
        return self._conversation_language_code()

    def _language_profile(self) -> dict:
        return get_language_profile(self._conversation_language_code())

    def _harness_mode(self) -> str:
        raw = self.web_msg_config.get("harness_mode")
        return normalize_harness_mode(raw)

    def _is_journey_mode(self) -> bool:
        return self._harness_mode() == HARNESS_JOURNEY

    def _judging_mechanics_config(self) -> dict:
        raw = self.web_msg_config.get("judging_mechanics")
        return resolve_judging_mechanics_config(raw if isinstance(raw, dict) else {})

    def _is_judging_mechanics_enabled(self) -> bool:
        return bool(self._judging_mechanics_config().get("enabled", False))

    def _apply_goal_judging_mechanics(
        self,
        *,
        evaluation: GoalEvaluation,
        hard_gate_success: bool,
        explanation: str,
    ) -> tuple[bool, str, Optional[JudgingMechanicsResult]]:
        config = self._judging_mechanics_config()
        if not config.get("enabled"):
            return hard_gate_success, explanation, None

        mechanics_payload = score_goal_evaluation(
            evaluation=evaluation,
            config=config,
            hard_gate_passed=hard_gate_success,
        )
        mechanics_result = JudgingMechanicsResult.model_validate(mechanics_payload)
        note = format_mechanics_summary(mechanics_payload)
        combined_explanation = explanation
        if note:
            combined_explanation = f"{explanation}\n\n{note}" if explanation else note
        return mechanics_result.final_gate_passed, combined_explanation, mechanics_result

    def _apply_journey_judging_mechanics(
        self,
        *,
        journey_result: JourneyValidationResult,
        hard_gate_success: bool,
        explanation: str,
    ) -> tuple[bool, str, Optional[JudgingMechanicsResult], Optional[str]]:
        config = self._judging_mechanics_config()
        if not config.get("enabled"):
            return hard_gate_success, explanation, None, None

        mechanics_payload = score_journey_evaluation(
            journey_result=journey_result,
            config=config,
            hard_gate_passed=hard_gate_success,
        )
        mechanics_result = JudgingMechanicsResult.model_validate(mechanics_payload)
        note = format_mechanics_summary(mechanics_payload)
        combined_explanation = explanation
        if note:
            combined_explanation = f"{explanation}\n\n{note}" if explanation else note

        error_code = None
        if hard_gate_success and not mechanics_result.passed_threshold:
            if "judging_score_below_threshold" not in journey_result.failure_reasons:
                journey_result.failure_reasons.append("judging_score_below_threshold")
            error_code = "journey_judging_score_below_threshold"

        return (
            mechanics_result.final_gate_passed,
            combined_explanation,
            mechanics_result,
            error_code,
        )

    def _category_rubric(self, category_name: Optional[str]) -> Optional[str]:
        normalized = normalize_category_name(category_name)
        if not normalized:
            return None
        categories = self.web_msg_config.get("primary_categories")
        if not isinstance(categories, list):
            return None
        for category in categories:
            if not isinstance(category, dict):
                continue
            name = normalize_category_name(category.get("name"))
            if name != normalized:
                continue
            rubric = str(category.get("rubric") or "").strip()
            if rubric:
                return rubric
        return None

    def _is_expected_greeting(self, message: str) -> bool:
        expected = (self.web_msg_config.get("expected_greeting") or "").strip()
        # Preserve legacy behavior: only enforce greeting matching when an
        # explicit expected greeting is configured for the run.
        if not expected:
            return True
        profile = self._language_profile()
        candidates: list[str] = [expected]
        for alias in profile.get("greeting_aliases", []):
            alias_text = str(alias or "").strip()
            if alias_text and alias_text not in candidates:
                candidates.append(alias_text)
        message_norm = self._normalize_text(message)
        for candidate in candidates:
            expected_norm = self._normalize_text(candidate)
            if not expected_norm:
                continue
            if expected_norm == message_norm or expected_norm in message_norm:
                return True
        required_tokens = profile.get("greeting_heuristic_required_tokens", [])
        any_tokens = profile.get("greeting_heuristic_any_tokens", [])
        normalized_required = [
            self._normalize_text(str(token or ""))
            for token in required_tokens
            if str(token or "").strip()
        ]
        normalized_any = [
            self._normalize_text(str(token or ""))
            for token in any_tokens
            if str(token or "").strip()
        ]
        if normalized_required and not all(
            token in message_norm for token in normalized_required
        ):
            return False
        if normalized_any and any(token in message_norm for token in normalized_any):
            return True
        return False

    def _is_presence_unsupported_message(self, message: str) -> bool:
        return "presence events are not supported" in self._normalize_text(message)

    def _extract_intent_from_text(self, text: str) -> Optional[str]:
        """Extract intent from test-mode agent text (e.g., INTENT=flight_cancel)."""
        stripped = text.strip()
        if not stripped:
            return None

        # JSON format, e.g. {"intent": "flight_cancel"} or {"detected_intent": "flight_cancel"}
        try:
            data = json.loads(stripped)
            intent_value = None
            if isinstance(data, dict):
                for key in ("intent", "detected_intent"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        intent_value = value
                        break
            if isinstance(intent_value, str) and intent_value.strip():
                return intent_value.strip().lower()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Plain text format, e.g. INTENT=flight_cancel, intent: flight_cancel,
        # detected_intent: flight_cancel
        match = re.search(
            r"\b(?:detected[_\-\s]?intent|intent)\b\s*[:=]\s*['\"]?([a-zA-Z0-9_\-./]+)",
            stripped,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().lower()

        return None

    def _find_detected_intent(
        self,
        conversation: list[Message],
        *,
        start_index: int = 0,
    ) -> Optional[str]:
        """Find most recent detected intent in agent messages after a start index."""
        clipped = conversation[max(0, int(start_index)) :]
        for msg in reversed(clipped):
            if msg.role != MessageRole.AGENT:
                continue
            detected = self._extract_intent_from_text(msg.content)
            if detected:
                return detected
        return None

    async def _send_language_selection_pre_step(
        self,
        *,
        scenario: TestScenario,
        client: WebMessagingClient,
        conversation: list[Message],
        turn_durations_seconds: list[float],
    ) -> Optional[str]:
        """Send optional language-selection message before scenario first message."""
        selection_message = str(scenario.language_selection_message or "").strip()
        if not selection_message:
            return None

        self._emit_attempt_status(
            f"Sending language selection message: {selection_message}"
        )
        user_sent_monotonic = time.monotonic()
        conversation.append(
            Message(
                role=MessageRole.USER,
                content=selection_message,
                timestamp=self._now_utc(),
            )
        )
        await self._await_step(
            "Sending language selection message",
            client.send_message(selection_message),
        )
        self._emit_attempt_status(
            "Waiting for agent response after language selection message"
        )
        try:
            response = await self._await_step(
                "Waiting for agent response after language selection message",
                client.receive_response(),
            )
        except TimeoutError:
            return None

        turn_durations_seconds.append(time.monotonic() - user_sent_monotonic)
        conversation.append(
            Message(
                role=MessageRole.AGENT,
                content=response,
                timestamp=self._now_utc(),
            )
        )
        return response

    async def _ensure_expected_greeting_before_main_utterance(
        self,
        *,
        client: WebMessagingClient,
        conversation: list[Message],
        language_pre_step_active: bool = False,
    ) -> None:
        """Block main scenario utterance until expected greeting is observed."""
        expected = str(self.web_msg_config.get("expected_greeting") or "").strip()
        if not expected:
            return

        max_agent_messages_before_first_user = 5
        configured_timeout = float(self.web_msg_config.get("timeout", 30))
        base_wait_timeout = min(
            self.web_msg_config.get("timeout", 30),
            self.web_msg_config.get("greeting_wait_timeout_seconds", 8),
        )
        wait_buffer_seconds = 0.0
        if language_pre_step_active:
            try:
                wait_buffer_seconds = max(
                    0.0,
                    float(
                        self.web_msg_config.get(
                            "localized_greeting_wait_buffer_seconds",
                            5.0,
                        )
                    ),
                )
            except (TypeError, ValueError):
                wait_buffer_seconds = 5.0
        greeting_wait_timeout = min(
            configured_timeout,
            base_wait_timeout + wait_buffer_seconds,
        )
        self._set_greeting_timeout_context(
            expected_greeting_configured=bool(expected),
            language_pre_step_active=language_pre_step_active,
            base_wait_timeout_seconds=base_wait_timeout,
            wait_buffer_seconds=wait_buffer_seconds,
            greeting_wait_timeout_seconds=greeting_wait_timeout,
        )
        deadline = time.monotonic() + greeting_wait_timeout

        while True:
            if any(
                msg.role == MessageRole.AGENT and self._is_expected_greeting(msg.content)
                for msg in conversation
            ):
                self._emit_attempt_status(
                    "Expected greeting detected before sending first user message"
                )
                return

            if (
                len([m for m in conversation if m.role == MessageRole.AGENT])
                >= max_agent_messages_before_first_user
            ):
                raise GreetingGateTimeoutError(greeting_wait_timeout)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GreetingGateTimeoutError(greeting_wait_timeout)

            self._emit_attempt_status(
                "Waiting for expected greeting before sending first user message"
            )
            try:
                agent_text = await self._await_step(
                    "Waiting for expected greeting before sending first user message",
                    client.receive_response(),
                    timeout_override_seconds=remaining,
                )
            except (TimeoutError, StepTimeoutError):
                raise GreetingGateTimeoutError(greeting_wait_timeout)

            conversation.append(
                Message(
                    role=MessageRole.AGENT,
                    content=agent_text,
                    timestamp=self._now_utc(),
                )
            )

    def _has_intent_api_fallback_config(self) -> bool:
        return bool(
            (self.web_msg_config.get("region") or "").strip()
            and (self.web_msg_config.get("gc_client_id") or "").strip()
            and (self.web_msg_config.get("gc_client_secret") or "").strip()
        )

    def _should_use_goal_evaluation_for_knowledge(
        self, expected_intent: str
    ) -> bool:
        """Return True when this expected intent should use goal-evaluation mode.

        Knowledge questions are validated by answer quality (goal evaluation) rather than
        intent string matching, because many agents do not emit intent markers for those.
        """
        normalized_expected = (expected_intent or "").strip().lower()
        if not normalized_expected:
            return False
        configured_values = self.web_msg_config.get(
            "knowledge_evaluation_intents",
            ["knowledge", "pets", "baggage"],
        )
        if not isinstance(configured_values, list):
            configured_values = ["knowledge", "pets", "baggage"]
        normalized_values = {
            str(value).strip().lower()
            for value in configured_values
            if str(value).strip()
        }
        return (
            normalized_expected in normalized_values
            or normalized_expected.startswith("knowledge")
        )

    def _extract_labeled_ids_from_text(
        self, text: str, label_pattern: str
    ) -> list[str]:
        """Extract UUIDs from explicit labeled fields (deterministic, no inference)."""
        if not text:
            return []

        # Matches:
        # - conversation_id: <uuid>
        # - "conversationId":"<uuid>"
        # - conversation-id = '<uuid>'
        pattern = re.compile(
            rf"(?i)(?:['\"])?(?:{label_pattern})(?:['\"])?\s*[:=]\s*['\"]?([0-9a-fA-F-]{{36}})"
        )
        results: list[str] = []
        for match in pattern.findall(text):
            normalized = match.strip().lower()
            if self._is_valid_conversation_id(normalized) and normalized not in results:
                results.append(normalized)
        return results

    def _extract_ids_from_transcript(
        self, conversation: list[Message]
    ) -> tuple[list[str], list[str]]:
        """Extract conversation/participant IDs explicitly surfaced in transcript text."""
        conversation_ids: list[str] = []
        participant_ids: list[str] = []

        for msg in conversation:
            text = msg.content or ""
            for cid in self._extract_labeled_ids_from_text(
                text, r"conversation[_\-\s]?id|conversationId"
            ):
                if cid not in conversation_ids:
                    conversation_ids.append(cid)
            for pid in self._extract_labeled_ids_from_text(
                text, r"participant[_\-\s]?id|participantId"
            ):
                if pid not in participant_ids:
                    participant_ids.append(pid)

        return conversation_ids, participant_ids

    def _resolve_conversation_ids_for_fallback(
        self,
        client: WebMessagingClient,
        conversation: list[Message],
    ) -> tuple[list[str], Optional[str]]:
        candidate_ids: list[str] = []

        if client.conversation_id:
            candidate_ids.append(client.conversation_id)

        client_candidates: list[str] = []
        try:
            raw_candidates = client.get_conversation_id_candidates()
            if inspect.isawaitable(raw_candidates):
                close_fn = getattr(raw_candidates, "close", None)
                if callable(close_fn):
                    close_fn()
                raw_candidates = []
            if isinstance(raw_candidates, list):
                client_candidates = [
                    c for c in raw_candidates if isinstance(c, str) and c.strip()
                ]
        except Exception:
            client_candidates = []

        for candidate in client_candidates:
            if candidate not in candidate_ids:
                candidate_ids.append(candidate)

        transcript_conversation_ids, _ = self._extract_ids_from_transcript(conversation)
        for candidate in transcript_conversation_ids:
            if candidate not in candidate_ids:
                candidate_ids.append(candidate)

        valid_candidate_ids = [
            cid for cid in candidate_ids if self._is_valid_conversation_id(cid)
        ]
        if valid_candidate_ids:
            return valid_candidate_ids, None
        if candidate_ids:
            return [], (
                "Web Messaging returned candidate IDs, but none matched a valid "
                "conversation-id format."
            )

        return [], (
            "Web Messaging did not provide a conversation ID and "
            "no explicit conversation_id was found in transcript text."
        )

    def _resolve_participant_id_for_fallback(
        self, client: WebMessagingClient, conversation: list[Message]
    ) -> Optional[str]:
        """Resolve participant ID from pulled payloads or explicit transcript labels."""
        if isinstance(client.participant_id, str):
            normalized = client.participant_id.strip().lower()
            if self._is_valid_conversation_id(normalized):
                return normalized

        _, transcript_participant_ids = self._extract_ids_from_transcript(conversation)
        if transcript_participant_ids:
            return transcript_participant_ids[0]
        return None

    def _is_valid_conversation_id(self, value: str) -> bool:
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError, TypeError):
            return False
        return str(parsed) == value.lower()

    async def _collect_follow_up_agent_messages_for_intent(
        self, client: WebMessagingClient
    ) -> list[str]:
        """Collect additional agent messages sent right after first response in intent mode.

        Some flows emit multiple outbound messages for a single user turn (e.g. first
        conversation_id, then detected_intent). This allows intent extraction/fallback
        to inspect those follow-up messages before ending the single-turn scenario.
        """
        max_messages_raw = self.web_msg_config.get("intent_follow_up_max_messages", 3)
        window_seconds_raw = self.web_msg_config.get("intent_follow_up_window_seconds", 8)
        try:
            max_messages = max(0, int(max_messages_raw))
        except (TypeError, ValueError):
            max_messages = 3
        try:
            window_seconds = max(0.0, float(window_seconds_raw))
        except (TypeError, ValueError):
            window_seconds = 8.0

        if max_messages == 0 or window_seconds == 0:
            return []

        self._emit_attempt_status(
            "Collecting follow-up agent messages for intent detection"
        )
        messages: list[str] = []
        deadline = time.monotonic() + window_seconds
        while len(messages) < max_messages:
            if self._is_stop_requested():
                raise StopRequestedError(
                    "Collecting follow-up agent messages for intent detection"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                follow_up = await self._await_step(
                    "Collecting follow-up agent messages for intent detection",
                    client.receive_response(),
                    timeout_override_seconds=remaining,
                )
            except (TimeoutError, StepTimeoutError):
                break
            messages.append(follow_up)
        return messages

    async def _send_knowledge_closure_turn(
        self,
        client: WebMessagingClient,
        conversation: list[Message],
        turn_durations_seconds: list[float],
    ) -> Optional[str]:
        """Send a closing user message to trigger final AVA output in knowledge flows."""
        profile = self._language_profile()
        closure_message = str(
            self.web_msg_config.get("knowledge_closure_message")
            or profile.get("default_knowledge_closure_message")
            or "no, thank you that is all"
        ).strip()
        if not closure_message:
            return None

        self._emit_attempt_status(
            f"Sending knowledge closure message: {closure_message}"
        )
        user_sent_monotonic = time.monotonic()
        conversation.append(
            Message(
                role=MessageRole.USER,
                content=closure_message,
                timestamp=self._now_utc(),
            )
        )
        await self._await_step(
            "Sending knowledge closure message",
            client.send_message(closure_message),
        )
        self._emit_attempt_status(
            "Waiting for agent response after knowledge closure message"
        )
        try:
            closure_response = await self._await_step(
                "Waiting for agent response after knowledge closure message",
                client.receive_response(),
            )
        except TimeoutError:
            return None

        turn_durations_seconds.append(time.monotonic() - user_sent_monotonic)
        conversation.append(
            Message(
                role=MessageRole.AGENT,
                content=closure_response,
                timestamp=self._now_utc(),
            )
        )
        return closure_response

    async def _get_intent_from_api_fallback(
        self,
        client: WebMessagingClient,
        conversation: list[Message],
    ) -> tuple[Optional[str], Optional[str]]:
        """Try Conversations API participant-attribute fallback for intent."""
        self._emit_attempt_status(
            "Querying Conversations API fallback for detected intent"
        )
        if not self._has_intent_api_fallback_config():
            return None, None

        conversation_ids, conversation_id_error = self._resolve_conversation_ids_for_fallback(
            client=client,
            conversation=conversation,
        )
        if not conversation_ids:
            return None, conversation_id_error

        retries_raw = self.web_msg_config.get("intent_attribute_retries", 3)
        retry_delay_raw = self.web_msg_config.get(
            "intent_attribute_retry_delay_seconds", 1.0
        )
        try:
            retries = max(1, int(retries_raw))
        except (TypeError, ValueError):
            retries = 3
        try:
            retry_delay_seconds = max(0.0, float(retry_delay_raw))
        except (TypeError, ValueError):
            retry_delay_seconds = 1.0

        intent_attribute_name = (
            self.web_msg_config.get("intent_attribute_name") or "detected_intent"
        )
        conversations_client = GenesysConversationsClient(
            region=self.web_msg_config["region"],
            client_id=self.web_msg_config["gc_client_id"],
            client_secret=self.web_msg_config["gc_client_secret"],
            timeout=self.web_msg_config.get("timeout", 30),
        )
        participant_id = self._resolve_participant_id_for_fallback(
            client=client, conversation=conversation
        )
        last_error: Optional[str] = None
        for conversation_id in conversation_ids:
            try:
                detected_intent = await self._await_step(
                    "Querying participant attribute via Conversations API",
                    asyncio.to_thread(
                        conversations_client.get_participant_attribute,
                        conversation_id=conversation_id,
                        attribute_name=intent_attribute_name,
                        participant_id=participant_id,
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                    ),
                )
            except GenesysConversationsError as e:
                last_error = (
                    f"Conversation ID '{conversation_id}' failed: {e}"
                )
                continue

            if detected_intent is not None:
                return detected_intent, None

            last_error = (
                f"Conversation ID '{conversation_id}' did not include attribute "
                f"'{intent_attribute_name}'."
            )

        return None, (
            last_error
            or f"Conversations API did not include participant attribute '{intent_attribute_name}'."
        )

    async def _resolve_journey_containment(
        self,
        *,
        client: WebMessagingClient,
        conversation: list[Message],
    ) -> tuple[Optional[bool], str, Optional[str]]:
        """Resolve containment, preferring metadata and falling back to LLM inference."""
        metadata_errors: list[str] = []

        if self._has_intent_api_fallback_config():
            conversation_ids, conversation_id_error = self._resolve_conversation_ids_for_fallback(
                client=client,
                conversation=conversation,
            )
            if not conversation_ids and conversation_id_error:
                metadata_errors.append(conversation_id_error)
            if conversation_ids:
                conversations_client = GenesysConversationsClient(
                    region=self.web_msg_config["region"],
                    client_id=self.web_msg_config["gc_client_id"],
                    client_secret=self.web_msg_config["gc_client_secret"],
                    timeout=self.web_msg_config.get("timeout", 30),
                )
                for conversation_id in conversation_ids:
                    try:
                        payload = await self._await_step(
                            "Fetching conversation metadata for journey containment",
                            asyncio.to_thread(
                                conversations_client.get_conversation_payload,
                                conversation_id,
                            ),
                        )
                    except (GenesysConversationsError, StepTimeoutError) as e:
                        metadata_errors.append(
                            f"Conversation metadata lookup failed for '{conversation_id}': {e}"
                        )
                        continue
                    contained = infer_containment_from_payload_metadata(payload)
                    if contained is not None:
                        return contained, "metadata", None
                if conversation_ids:
                    metadata_errors.append(
                        "Conversation metadata did not provide a deterministic containment signal."
                    )
        else:
            metadata_errors.append(
                "GC client credentials are not configured; metadata containment lookup skipped."
            )

        try:
            inference = await self._run_judge_step(
                "Inferring containment with Judge LLM",
                self.judge.infer_containment,
                conversation_history=conversation,
                language_code=self._evaluation_results_language_code(),
            )
            contained = inference.get("contained")
            if isinstance(contained, bool):
                return contained, "llm_fallback", None
            explanation = str(inference.get("explanation") or "").strip()
            fallback_note = (
                "Containment inference was inconclusive."
                + (f" {explanation}" if explanation else "")
            )
            return None, "llm_inconclusive", fallback_note
        except JudgeLLMError as e:
            metadata_note = "; ".join(metadata_errors).strip()
            reason = f"Containment inference failed: {e}"
            if metadata_note:
                reason = f"{metadata_note}; {reason}"
            return None, "llm_error", reason

    async def _evaluate_journey_attempt(
        self,
        *,
        scenario: TestScenario,
        conversation: list[Message],
        client: WebMessagingClient,
    ) -> tuple[bool, str, JourneyValidationResult, Optional[str]]:
        """Evaluate attempt outcome for journey-mode scenarios."""
        validation_config = scenario.journey_validation
        require_containment = (
            validation_config.require_containment
            if validation_config is not None
            else True
        )
        require_fulfillment = (
            validation_config.require_fulfillment
            if validation_config is not None
            else True
        )
        expected_category = normalize_category_name(scenario.journey_category)
        path_rubric = (
            (validation_config.path_rubric or "").strip()
            if validation_config is not None
            else ""
        )
        category_rubric_override = (
            (validation_config.category_rubric_override or "").strip()
            if validation_config is not None
            else ""
        )
        if not path_rubric:
            path_rubric = self._category_rubric(expected_category) or ""
        if not category_rubric_override:
            category_rubric_override = self._category_rubric(expected_category) or ""

        self._emit_attempt_status(
            (
                "Resolving journey category requirement: "
                + (expected_category or "none")
            )
        )
        self._emit_attempt_status("Resolving journey containment signal")
        contained, containment_source, containment_note = await self._resolve_journey_containment(
            client=client,
            conversation=conversation,
        )
        if contained is True:
            self._emit_attempt_status(
                f"Journey containment resolved: contained=true ({containment_source})"
            )
        elif contained is False:
            self._emit_attempt_status(
                f"Journey containment resolved: contained=false ({containment_source})"
            )
        else:
            self._emit_attempt_status(
                f"Journey containment unresolved ({containment_source})"
            )

        self._emit_attempt_status("Evaluating journey outcome with Judge LLM")
        journey_result = await self._run_judge_step(
            "Evaluating journey outcome with Judge LLM",
            self.judge.evaluate_journey,
            persona=scenario.persona,
            goal=scenario.goal,
            expected_category=expected_category,
            path_rubric=path_rubric or None,
            category_rubric=category_rubric_override or None,
            conversation_history=conversation,
            language_code=self._evaluation_results_language_code(),
            known_contained=contained if containment_source == "metadata" else None,
        )
        journey_result.expected_category = expected_category
        journey_result.containment_source = containment_source
        if contained is not None:
            # Preserve metadata/LLM fallback signal as source of truth for gating.
            journey_result.contained = contained
        if (
            expected_category
            and journey_result.category_match is None
            and journey_result.actual_category
        ):
            journey_result.category_match = (
                normalize_category_name(journey_result.actual_category)
                == expected_category
            )

        failure_reasons: list[str] = []
        if require_containment:
            if journey_result.contained is False:
                failure_reasons.append("escalated_to_human")
            elif journey_result.contained is None:
                failure_reasons.append("containment_unknown")
        if require_fulfillment:
            if not journey_result.fulfilled:
                failure_reasons.append("fulfillment_failed")
            if not journey_result.path_correct:
                failure_reasons.append("path_mismatch")
        if expected_category and journey_result.category_match is not True:
            failure_reasons.append("category_mismatch")

        journey_result.failure_reasons = sorted(set(failure_reasons))
        success = len(journey_result.failure_reasons) == 0

        summary_lines = [
            journey_result.explanation or "Journey evaluation completed.",
            "",
            "Journey Validation Summary:",
            f"- Category Match: {journey_result.category_match}",
            f"- Fulfilled: {journey_result.fulfilled}",
            f"- Path Correct: {journey_result.path_correct}",
            f"- Contained: {journey_result.contained} ({containment_source})",
        ]
        if containment_note:
            summary_lines.append(f"- Containment Note: {containment_note}")
        if journey_result.failure_reasons:
            summary_lines.append(
                "- Failure Reasons: " + ", ".join(journey_result.failure_reasons)
            )

        error_code = (
            f"journey_{journey_result.failure_reasons[0]}"
            if journey_result.failure_reasons
            else None
        )
        return success, "\n".join(summary_lines), journey_result, error_code

    def _configured_tool_attribute_keys(self) -> list[str]:
        raw = self.web_msg_config.get("tool_attribute_keys", ["rth_tool_events", "tool_events"])
        if isinstance(raw, str):
            values = [item.strip().lower() for item in raw.split(",") if item.strip()]
            return values or ["rth_tool_events", "tool_events"]
        if isinstance(raw, list):
            values = [str(item).strip().lower() for item in raw if str(item).strip()]
            return values or ["rth_tool_events", "tool_events"]
        return ["rth_tool_events", "tool_events"]

    def _configured_tool_marker_prefixes(self) -> list[str]:
        raw = self.web_msg_config.get("tool_marker_prefixes", ["tool_event:"])
        if isinstance(raw, str):
            values = [item.strip().lower() for item in raw.split(",") if item.strip()]
            return values or ["tool_event:"]
        if isinstance(raw, list):
            values = [str(item).strip().lower() for item in raw if str(item).strip()]
            return values or ["tool_event:"]
        return ["tool_event:"]

    async def _collect_tool_events(
        self,
        client: WebMessagingClient,
        conversation: list[Message],
    ) -> tuple[list[ToolEvent], list[str]]:
        """Collect tool events from participant attributes and response markers."""
        events: list[ToolEvent] = []
        collection_notes: list[str] = []
        attribute_keys = self._configured_tool_attribute_keys()
        marker_prefixes = self._configured_tool_marker_prefixes()

        if self._has_intent_api_fallback_config():
            conversation_ids, conversation_id_error = self._resolve_conversation_ids_for_fallback(
                client=client,
                conversation=conversation,
            )
            participant_id = self._resolve_participant_id_for_fallback(
                client=client,
                conversation=conversation,
            )
            if not conversation_ids:
                if conversation_id_error:
                    collection_notes.append(conversation_id_error)
            else:
                retries_raw = self.web_msg_config.get("tool_attribute_retries", 2)
                retry_delay_raw = self.web_msg_config.get(
                    "tool_attribute_retry_delay_seconds", 0.5
                )
                try:
                    retries = max(1, int(retries_raw))
                except (TypeError, ValueError):
                    retries = 2
                try:
                    retry_delay_seconds = max(0.0, float(retry_delay_raw))
                except (TypeError, ValueError):
                    retry_delay_seconds = 0.5

                conversations_client = GenesysConversationsClient(
                    region=self.web_msg_config["region"],
                    client_id=self.web_msg_config["gc_client_id"],
                    client_secret=self.web_msg_config["gc_client_secret"],
                    timeout=self.web_msg_config.get("timeout", 30),
                )

                found_primary_events = False
                for conversation_id in conversation_ids:
                    try:
                        self._emit_attempt_status(
                            (
                                "Collecting tool events from participant attributes "
                                f"(conversation {conversation_id})"
                            )
                        )
                        attributes = await self._await_step(
                            "Collecting tool events from participant attributes",
                            asyncio.to_thread(
                                conversations_client.get_participant_attributes,
                                conversation_id=conversation_id,
                                participant_id=participant_id,
                                retries=retries,
                                retry_delay_seconds=retry_delay_seconds,
                            ),
                        )
                    except StepTimeoutError as e:
                        collection_notes.append(
                            (
                                f"Tool attribute lookup timed out for "
                                f"conversation '{conversation_id}': {e}"
                            )
                        )
                        continue
                    except GenesysConversationsError as e:
                        collection_notes.append(
                            f"Conversation ID '{conversation_id}' attribute lookup failed: {e}"
                        )
                        continue

                    parsed_events = parse_tool_events_from_attribute_map(
                        attributes,
                        attribute_keys=attribute_keys,
                        source="participant_attribute",
                    )
                    if parsed_events:
                        events.extend(parsed_events)
                        found_primary_events = True
                        break

                if not found_primary_events:
                    collection_notes.append(
                        (
                            "No tool events found in participant attributes for keys: "
                            + ", ".join(attribute_keys)
                        )
                    )
        else:
            collection_notes.append(
                "GC client credentials are not configured; participant-attribute tool lookup skipped."
            )

        marker_events = parse_tool_events_from_markers(
            conversation,
            marker_prefixes=marker_prefixes,
        )
        if marker_events:
            self._emit_attempt_status(
                f"Parsed {len(marker_events)} tool event marker(s) from transcript responses"
            )
            events.extend(marker_events)

        deduped_events = dedupe_tool_events(events)
        if events and len(deduped_events) != len(events):
            self._emit_attempt_status(
                f"Deduplicated tool events from {len(events)} to {len(deduped_events)}"
            )
        return deduped_events, collection_notes

    def _tool_validation_note(
        self,
        validation_result: ToolValidationResult,
        collection_notes: list[str],
    ) -> str:
        """Render a concise validation summary for explanation payloads."""
        lines = [
            "Tool Validation Summary:",
            (
                f"- Loose Rule: {'PASS' if validation_result.loose_pass else 'FAIL'}"
            ),
        ]
        if validation_result.strict_pass is not None:
            lines.append(
                f"- Strict Rule: {'PASS' if validation_result.strict_pass else 'FAIL'}"
            )
        if validation_result.missing_tools:
            lines.append(
                f"- Missing Tools: {', '.join(sorted(validation_result.missing_tools))}"
            )
        if validation_result.order_violations:
            lines.append(
                "- Order Violations: " + "; ".join(validation_result.order_violations)
            )
        if validation_result.loose_fail_reasons:
            lines.append(
                "- Loose Fail Reasons: " + "; ".join(validation_result.loose_fail_reasons)
            )
        if validation_result.strict_fail_reasons:
            lines.append(
                "- Strict Fail Reasons: " + "; ".join(validation_result.strict_fail_reasons)
            )
        if collection_notes:
            lines.append("- Collection Notes: " + "; ".join(collection_notes))
        return "\n".join(lines)

    async def _apply_tool_validation(
        self,
        attempt_result: AttemptResult,
        scenario: TestScenario,
        client: WebMessagingClient,
        conversation: list[Message],
    ) -> AttemptResult:
        """Attach tool evidence and apply loose/strict tool validation outcomes."""
        if scenario.tool_validation is None:
            return attempt_result

        self._emit_attempt_status("Collecting tool execution evidence")
        collection_notes: list[str] = []
        tool_events: list[ToolEvent] = []
        try:
            tool_events, collection_notes = await self._collect_tool_events(
                client=client,
                conversation=conversation,
            )
            self._emit_attempt_status("Evaluating loose/strict tool validation rules")
            validation_result = evaluate_tool_validation(
                scenario.tool_validation,
                tool_events,
            )
        except Exception as e:
            collection_notes.append(f"Tool validation pipeline error: {e}")
            validation_result = ToolValidationResult(
                loose_pass=False,
                strict_pass=False if scenario.tool_validation.strict_rule else None,
                missing_signal=True,
                loose_fail_reasons=[
                    "Tool validation pipeline failed before rule evaluation."
                ],
                strict_fail_reasons=[],
                missing_tools=[],
                order_violations=[],
                matched_tools=[],
            )
        attempt_result.tool_validation_result = validation_result
        attempt_result.tool_events = tool_events

        if validation_result.loose_pass:
            self._emit_attempt_status("Tool validation loose rule passed")
        else:
            self._emit_attempt_status("Tool validation loose rule failed")
        if validation_result.strict_pass is not None:
            self._emit_attempt_status(
                (
                    "Tool validation strict rule "
                    + ("passed" if validation_result.strict_pass else "failed")
                )
            )

        note = self._tool_validation_note(validation_result, collection_notes)
        if attempt_result.explanation:
            attempt_result.explanation = f"{attempt_result.explanation}\n\n{note}"
        else:
            attempt_result.explanation = note

        if attempt_result.timed_out or attempt_result.skipped:
            # Timeout/skipped status always takes precedence.
            return attempt_result

        if not validation_result.loose_pass:
            attempt_result.success = False
            if validation_result.missing_signal:
                attempt_result.error = "missing_tool_signal"
            elif not attempt_result.error:
                attempt_result.error = "tool_validation_failed"

        return attempt_result

    def _intent_result_explanation(
        self,
        expected_intent: str,
        detected_intent: str,
        *,
        from_api_fallback: bool,
    ) -> str:
        if from_api_fallback:
            source = (
                self.web_msg_config.get("intent_attribute_name")
                or "detected_intent"
            )
            if detected_intent == expected_intent:
                return (
                    f"Intent matched expected value '{expected_intent}' via Conversations "
                    f"API participant attribute '{source}'."
                )
            return (
                f"Intent mismatch: expected '{expected_intent}' but got '{detected_intent}' "
                f"from Conversations API participant attribute '{source}'."
            )

        if detected_intent == expected_intent:
            return f"Intent matched expected value '{expected_intent}'."
        return (
            f"Intent mismatch: expected '{expected_intent}' but got '{detected_intent}'."
        )

    def _default_follow_up_answer_for_intent(
        self,
        expected_intent: Optional[str],
    ) -> Optional[str]:
        """Return default simulated follow-up answer for known intent branches."""
        profile = self._language_profile()
        normalized = (expected_intent or "").strip().lower()
        if normalized == "flight_priority_change":
            return random.choice(
                [
                    str(profile.get("default_flight_priority_yes") or "yes"),
                    str(profile.get("default_flight_priority_no") or "no"),
                ]
            )
        if normalized == "speak_to_agent":
            return str(
                profile.get("default_speak_to_agent_follow_up")
                or "Yes, connect me to a live agent"
            )
        if normalized == "vacation_inquiry_flight_only":
            return str(profile.get("default_vacation_flight_only") or "flight only")
        if normalized == "vacation_flight_and_hotel":
            return str(
                profile.get("default_vacation_flight_and_hotel")
                or "flight and hotel"
            )
        if normalized == "vacation_inquiry":
            return random.choice(
                [
                    str(profile.get("default_vacation_flight_only") or "flight only"),
                    str(
                        profile.get("default_vacation_flight_and_hotel")
                        or "flight and hotel"
                    ),
                ]
            )
        return None

    def _answer_matches_tokens(
        self,
        normalized_answer: str,
        tokens: set[str],
    ) -> bool:
        if not normalized_answer:
            return False
        for token in tokens:
            normalized_token = self._normalize_text(token)
            if not normalized_token:
                continue
            if normalized_answer == normalized_token:
                return True
            if re.search(rf"\b{re.escape(normalized_token)}\b", normalized_answer):
                return True
        return False

    def _resolve_follow_up_answer_for_intent(
        self,
        scenario: TestScenario,
        expected_intent: Optional[str],
    ) -> Optional[str]:
        """Resolve simulated follow-up user answer with scenario override precedence."""
        override = (scenario.intent_follow_up_user_message or "").strip()
        if override:
            return override
        return self._default_follow_up_answer_for_intent(expected_intent)

    def _resolve_expected_intent_after_follow_up(
        self,
        expected_intent: Optional[str],
        follow_up_answer: Optional[str],
    ) -> Optional[str]:
        """Resolve dynamic expected intent variants based on follow-up user answers."""
        profile = self._language_profile()
        english_profile = get_language_profile("en")
        normalized_intent = (expected_intent or "").strip().lower()
        normalized_answer = self._normalize_text(follow_up_answer or "")

        if normalized_intent == "flight_priority_change":
            yes_tokens = set(profile.get("yes_tokens", set())) | set(
                english_profile.get("yes_tokens", set())
            )
            no_tokens = set(profile.get("no_tokens", set())) | set(
                english_profile.get("no_tokens", set())
            )
            if self._answer_matches_tokens(normalized_answer, yes_tokens):
                return "flight_change_priority_within_72_hours"
            if self._answer_matches_tokens(normalized_answer, no_tokens):
                return "flight_change_later_than_72_hours"
            return normalized_intent

        if normalized_intent == "vacation_inquiry":
            flight_only_tokens = set(
                profile.get("vacation_flight_only_tokens", set())
            ) | set(english_profile.get("vacation_flight_only_tokens", set()))
            flight_hotel_tokens = set(
                profile.get("vacation_flight_and_hotel_tokens", set())
            ) | set(
                english_profile.get("vacation_flight_and_hotel_tokens", set())
            )
            if self._answer_matches_tokens(normalized_answer, flight_hotel_tokens):
                return "vacation_flight_and_hotel"
            if self._answer_matches_tokens(normalized_answer, flight_only_tokens):
                return "vacation_inquiry_flight_only"
            return normalized_intent

        return normalized_intent

    async def run_attempt(
        self,
        scenario: TestScenario,
        attempt_number: int,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> AttemptResult:
        """Execute a single conversation attempt for a scenario.

        Creates a new WebMessagingClient (test isolation), connects, waits for
        the agent welcome message, drives the conversation via the Judge LLM,
        evaluates the goal, and returns the result.

        Args:
            scenario: The test scenario to execute.
            attempt_number: The attempt number (1-based).

        Returns:
            AttemptResult with conversation history, success/failure, and explanation.
        """
        conversation: list[Message] = []
        turn_durations_seconds: list[float] = []
        detected_intent: Optional[str] = None
        intent_fallback_error: Optional[str] = None
        intent_detected_via_api_fallback = False
        knowledge_closure_attempted = False
        intent_follow_up_user_answer_sent = False
        journey_mode = self._is_journey_mode()
        expected_intent = (
            scenario.expected_intent.strip().lower()
            if scenario.expected_intent
            else None
        )
        step_log: list[dict] = []
        self._active_status_callback = status_callback
        self._active_step_log = step_log
        self._active_timeout_context = {
            "last_step_name": None,
            "last_step_timeout_seconds": None,
            "greeting": {},
        }
        if expected_intent and self._should_use_goal_evaluation_for_knowledge(expected_intent):
            self._emit_attempt_status(
                (
                    f"Knowledge mode detected for expected_intent '{expected_intent}'. "
                    "Using LLM goal evaluation instead of strict intent matching."
                )
            )
            expected_intent = None
        if journey_mode and expected_intent:
            self._emit_attempt_status(
                (
                    "Journey mode is active; ignoring scenario expected_intent "
                    "and using full-journey validation."
                )
            )
            expected_intent = None
        started_at = self._now_utc()
        attempt_start_monotonic = time.monotonic()
        client = WebMessagingClient(
            region=self.web_msg_config["region"],
            deployment_id=self.web_msg_config["deployment_id"],
            timeout=self.web_msg_config.get("timeout", 30),
            origin=self.web_msg_config.get("origin", "https://apps.mypurecloud.com"),
            debug_capture_frames=self.web_msg_config.get("debug_capture_frames", False),
            debug_capture_frame_limit=self.web_msg_config.get("debug_capture_frame_limit", 8),
        )

        def build_attempt_result(
            *,
            success: bool,
            explanation: str,
            error: Optional[str] = None,
            timed_out: bool = False,
            skipped: bool = False,
            timeout_diagnostics: Optional[TimeoutDiagnostics] = None,
        ) -> AttemptResult:
            debug_frames: list[dict] = []
            try:
                get_debug_frames = getattr(client, "get_debug_frames", None)
                if callable(get_debug_frames):
                    raw_frames = get_debug_frames()
                    if inspect.isawaitable(raw_frames):
                        # Avoid leaking un-awaited coroutine objects from AsyncMock in tests.
                        close_fn = getattr(raw_frames, "close", None)
                        if callable(close_fn):
                            close_fn()
                        raw_frames = None
                    if isinstance(raw_frames, list):
                        debug_frames = [f for f in raw_frames if isinstance(f, dict)]
            except Exception:
                debug_frames = []
            return AttemptResult(
                attempt_number=attempt_number,
                success=success,
                conversation=conversation,
                explanation=explanation,
                error=error,
                timed_out=timed_out,
                skipped=skipped,
                detected_intent=detected_intent,
                started_at=started_at,
                completed_at=self._now_utc(),
                duration_seconds=time.monotonic() - attempt_start_monotonic,
                turn_durations_seconds=turn_durations_seconds,
                step_log=step_log,
                debug_frames=debug_frames,
                timeout_diagnostics=timeout_diagnostics,
            )

        async def finalize_attempt_result(
            *,
            success: bool,
            explanation: str,
            error: Optional[str] = None,
            timed_out: bool = False,
            skipped: bool = False,
            timeout_diagnostics: Optional[TimeoutDiagnostics] = None,
            journey_validation_result: Optional[JourneyValidationResult] = None,
            judging_mechanics_result: Optional[JudgingMechanicsResult] = None,
        ) -> AttemptResult:
            raw_result = build_attempt_result(
                success=success,
                explanation=explanation,
                error=error,
                timed_out=timed_out,
                skipped=skipped,
                timeout_diagnostics=timeout_diagnostics,
            )
            raw_result.journey_validation_result = journey_validation_result
            raw_result.judging_mechanics_result = judging_mechanics_result
            return await self._apply_tool_validation(
                raw_result,
                scenario=scenario,
                client=client,
                conversation=conversation,
            )

        try:
            if self._is_stop_requested():
                raise StopRequestedError("Attempt initialization")

            # Connect, send join event, and wait for welcome message
            self._emit_attempt_status("Connecting to Web Messaging")
            await self._await_step("Connecting to Web Messaging", client.connect())
            self._emit_attempt_status("Sending join event")
            await self._await_step("Sending join event", client.send_join())
            self._emit_attempt_status("Waiting for welcome message")
            welcome_text = await self._await_step(
                "Waiting for welcome message",
                client.wait_for_welcome(),
            )

            # Add welcome message to conversation history
            conversation.append(
                Message(
                    role=MessageRole.AGENT,
                    content=welcome_text,
                    timestamp=self._now_utc(),
                )
            )

            # Some deployments reject presence events and only emit the greeting
            # after receiving any user text. Send a neutral bootstrap message so
            # the scenario's test utterance can still be sent after greeting.
            if (
                not self._is_expected_greeting(conversation[-1].content)
                and self._is_presence_unsupported_message(conversation[-1].content)
            ):
                bootstrap_message = self.web_msg_config.get(
                    "greeting_bootstrap_message", "Hi"
                )
                conversation.append(
                    Message(
                        role=MessageRole.USER,
                        content=bootstrap_message,
                        timestamp=self._now_utc(),
                    )
                )
                await self._await_step(
                    "Sending bootstrap message for greeting",
                    client.send_message(bootstrap_message),
                )
                bootstrap_response = await self._await_step(
                    "Waiting for bootstrap agent response",
                    client.receive_response(),
                )
                conversation.append(
                    Message(
                        role=MessageRole.AGENT,
                        content=bootstrap_response,
                        timestamp=self._now_utc(),
                    )
                )
                bootstrap_detected = self._extract_intent_from_text(bootstrap_response)
                if bootstrap_detected is not None:
                    detected_intent = bootstrap_detected
                    intent_detected_via_api_fallback = False

            language_pre_step_active = bool(
                str(scenario.language_selection_message or "").strip()
            )
            await self._send_language_selection_pre_step(
                scenario=scenario,
                client=client,
                conversation=conversation,
                turn_durations_seconds=turn_durations_seconds,
            )
            await self._ensure_expected_greeting_before_main_utterance(
                client=client,
                conversation=conversation,
                language_pre_step_active=language_pre_step_active,
            )
            # Intent assertions should begin only after pre-steps and greeting gate.
            intent_detection_start_index = len(conversation)

            # Conversation loop — keep going until goal achieved or max turns
            turn_count = 0
            first_turn = True
            early_success = False
            while turn_count < self.max_turns:
                # Intent-classification scenarios are single-input by design.
                if expected_intent is not None and not first_turn:
                    break

                # On first turn, use first_message if provided
                if first_turn and scenario.first_message:
                    user_message = scenario.first_message
                    first_turn = False
                else:
                    first_turn = False
                    # Generate next user message
                    self._emit_attempt_status("Generating user message with Judge LLM")
                    user_message = await self._run_judge_step(
                        "Generating user message with Judge LLM",
                        self.judge.generate_user_message,
                        persona=scenario.persona,
                        goal=scenario.goal,
                        conversation_history=conversation,
                        language_code=self._conversation_language_code(),
                    )
                    self._emit_attempt_status("Judge LLM generated next user message")
                user_sent_monotonic = time.monotonic()
                conversation.append(
                    Message(
                        role=MessageRole.USER,
                        content=user_message,
                        timestamp=self._now_utc(),
                    )
                )

                # Send to agent and receive response
                await self._await_step(
                    "Sending user message to agent",
                    client.send_message(user_message),
                )
                try:
                    self._emit_attempt_status("Waiting for agent response")
                    agent_response = await self._await_step(
                        "Waiting for agent response",
                        client.receive_response(),
                    )
                except TimeoutError as timeout_error:
                    if expected_intent is not None:
                        fallback_intent, fallback_error = await self._get_intent_from_api_fallback(
                            client=client,
                            conversation=conversation,
                        )
                        if fallback_intent is not None:
                            detected_intent = fallback_intent
                            intent_detected_via_api_fallback = True
                            matched = detected_intent == expected_intent
                            return await finalize_attempt_result(
                                success=matched,
                                explanation=self._intent_result_explanation(
                                    expected_intent,
                                    detected_intent,
                                    from_api_fallback=True,
                                ),
                            )
                        intent_fallback_error = fallback_error
                        fallback_hint = ""
                        if intent_fallback_error:
                            fallback_hint = (
                                f" Conversations API fallback result: {intent_fallback_error}"
                            )
                        elif not self._has_intent_api_fallback_config():
                            fallback_hint = (
                                " Configure GC_CLIENT_ID and GC_CLIENT_SECRET "
                                "to enable Conversations API fallback."
                            )
                        return await finalize_attempt_result(
                            success=False,
                            explanation=(
                                f"Expected intent '{expected_intent}' was not found because "
                                "the bot did not return a test-mode intent response."
                                f"{fallback_hint}"
                            ),
                            error=str(timeout_error),
                        )
                    raise
                turn_durations_seconds.append(time.monotonic() - user_sent_monotonic)
                conversation.append(
                    Message(
                        role=MessageRole.AGENT,
                        content=agent_response,
                        timestamp=self._now_utc(),
                    )
                )

                response_detected_intent = self._extract_intent_from_text(agent_response)
                if response_detected_intent is not None:
                    detected_intent = response_detected_intent
                    intent_detected_via_api_fallback = False

                if expected_intent is not None:
                    if response_detected_intent is not None:
                        matched = response_detected_intent == expected_intent
                        return await finalize_attempt_result(
                            success=matched,
                            explanation=self._intent_result_explanation(
                                expected_intent,
                                response_detected_intent,
                                from_api_fallback=False,
                            ),
                        )

                    follow_up_responses = await self._collect_follow_up_agent_messages_for_intent(client)
                    for follow_up_response in follow_up_responses:
                        conversation.append(
                            Message(
                                role=MessageRole.AGENT,
                                content=follow_up_response,
                                timestamp=self._now_utc(),
                            )
                        )
                        follow_up_detected_intent = self._extract_intent_from_text(
                            follow_up_response
                        )
                        if follow_up_detected_intent is not None:
                            detected_intent = follow_up_detected_intent
                            intent_detected_via_api_fallback = False
                            matched = follow_up_detected_intent == expected_intent
                            return await finalize_attempt_result(
                                success=matched,
                                explanation=self._intent_result_explanation(
                                    expected_intent,
                                    follow_up_detected_intent,
                                    from_api_fallback=False,
                                ),
                            )

                    if not intent_follow_up_user_answer_sent:
                        follow_up_answer = self._resolve_follow_up_answer_for_intent(
                            scenario,
                            expected_intent,
                        )
                        if follow_up_answer:
                            resolved_expected_intent = (
                                self._resolve_expected_intent_after_follow_up(
                                    expected_intent,
                                    follow_up_answer,
                                )
                            )
                            if (
                                resolved_expected_intent
                                and resolved_expected_intent != expected_intent
                            ):
                                self._emit_attempt_status(
                                    (
                                        "Expected intent updated after follow-up answer "
                                        f"'{follow_up_answer}': {resolved_expected_intent}"
                                    )
                                )
                                expected_intent = resolved_expected_intent
                            intent_follow_up_user_answer_sent = True
                            self._emit_attempt_status(
                                (
                                    "Sending simulated follow-up user answer "
                                    f"for intent '{expected_intent}': {follow_up_answer}"
                                )
                            )
                            conversation.append(
                                Message(
                                    role=MessageRole.USER,
                                    content=follow_up_answer,
                                    timestamp=self._now_utc(),
                                )
                            )
                            await self._await_step(
                                "Sending simulated follow-up user answer",
                                client.send_message(follow_up_answer),
                            )
                            follow_up_agent_response: Optional[str] = None
                            try:
                                self._emit_attempt_status(
                                    "Waiting for agent response to simulated follow-up answer"
                                )
                                follow_up_agent_response = await self._await_step(
                                    "Waiting for agent response to simulated follow-up answer",
                                    client.receive_response(),
                                )
                            except (TimeoutError, StepTimeoutError):
                                # Some flows emit the final intent only in later
                                # follow-up messages. Continue collection below.
                                follow_up_agent_response = None
                            if follow_up_agent_response is not None:
                                conversation.append(
                                    Message(
                                        role=MessageRole.AGENT,
                                        content=follow_up_agent_response,
                                        timestamp=self._now_utc(),
                                    )
                                )
                                follow_up_agent_intent = self._extract_intent_from_text(
                                    follow_up_agent_response
                                )
                                if follow_up_agent_intent is not None:
                                    detected_intent = follow_up_agent_intent
                                    intent_detected_via_api_fallback = False
                                    matched = follow_up_agent_intent == expected_intent
                                    return await finalize_attempt_result(
                                        success=matched,
                                        explanation=self._intent_result_explanation(
                                            expected_intent,
                                            follow_up_agent_intent,
                                            from_api_fallback=False,
                                        ),
                                    )

                            follow_up_responses = (
                                await self._collect_follow_up_agent_messages_for_intent(
                                    client
                                )
                            )
                            for follow_up_response in follow_up_responses:
                                conversation.append(
                                    Message(
                                        role=MessageRole.AGENT,
                                        content=follow_up_response,
                                        timestamp=self._now_utc(),
                                    )
                                )
                                follow_up_detected_intent = self._extract_intent_from_text(
                                    follow_up_response
                                )
                                if follow_up_detected_intent is not None:
                                    detected_intent = follow_up_detected_intent
                                    intent_detected_via_api_fallback = False
                                    matched = follow_up_detected_intent == expected_intent
                                    return await finalize_attempt_result(
                                        success=matched,
                                        explanation=self._intent_result_explanation(
                                            expected_intent,
                                            follow_up_detected_intent,
                                            from_api_fallback=False,
                                        ),
                                    )

                    if not knowledge_closure_attempted:
                        knowledge_closure_attempted = True
                        closure_response = await self._send_knowledge_closure_turn(
                            client=client,
                            conversation=conversation,
                            turn_durations_seconds=turn_durations_seconds,
                        )
                        if closure_response is not None:
                            closure_detected_intent = self._extract_intent_from_text(
                                closure_response
                            )
                            if closure_detected_intent is not None:
                                detected_intent = closure_detected_intent
                                intent_detected_via_api_fallback = False
                                matched = closure_detected_intent == expected_intent
                                return await finalize_attempt_result(
                                    success=matched,
                                    explanation=self._intent_result_explanation(
                                        expected_intent,
                                        closure_detected_intent,
                                        from_api_fallback=False,
                                    ),
                                )

                            closure_follow_ups = await self._collect_follow_up_agent_messages_for_intent(
                                client
                            )
                            for closure_follow_up in closure_follow_ups:
                                conversation.append(
                                    Message(
                                        role=MessageRole.AGENT,
                                        content=closure_follow_up,
                                        timestamp=self._now_utc(),
                                    )
                                )
                                closure_follow_up_intent = self._extract_intent_from_text(
                                    closure_follow_up
                                )
                                if closure_follow_up_intent is not None:
                                    detected_intent = closure_follow_up_intent
                                    intent_detected_via_api_fallback = False
                                    matched = closure_follow_up_intent == expected_intent
                                    return await finalize_attempt_result(
                                        success=matched,
                                        explanation=self._intent_result_explanation(
                                            expected_intent,
                                            closure_follow_up_intent,
                                            from_api_fallback=False,
                                        ),
                                    )

                    fallback_intent, fallback_error = await self._get_intent_from_api_fallback(
                        client=client,
                        conversation=conversation,
                    )
                    if fallback_intent is not None:
                        detected_intent = fallback_intent
                        intent_detected_via_api_fallback = True
                        matched = detected_intent == expected_intent
                        return await finalize_attempt_result(
                            success=matched,
                            explanation=self._intent_result_explanation(
                                expected_intent,
                                detected_intent,
                                from_api_fallback=True,
                            ),
                        )
                    if fallback_error is not None:
                        intent_fallback_error = fallback_error
                    continue

                turn_count += 1

                # Check if goal achieved — only stop early on success
                try:
                    self._emit_attempt_status(
                        "Evaluating goal with Judge LLM (mid-conversation)"
                    )
                    evaluation = await self._run_judge_step(
                        "Evaluating goal with Judge LLM (mid-conversation)",
                        self.judge.evaluate_goal,
                        persona=scenario.persona,
                        goal=scenario.goal,
                        conversation_history=conversation,
                        language_code=self._evaluation_results_language_code(),
                    )
                    if evaluation.success:
                        self._emit_attempt_status(
                            "Judge LLM marked goal as achieved"
                        )
                        early_success = True
                        break
                    self._emit_attempt_status(
                        "Judge LLM marked goal as not yet achieved"
                    )
                except JudgeLLMError:
                    self._emit_attempt_status(
                        "Judge LLM mid-conversation evaluation failed; continuing"
                    )
                    pass  # If evaluation fails mid-conversation, keep going

            if expected_intent is not None:
                detected_intent = self._find_detected_intent(
                    conversation,
                    start_index=intent_detection_start_index,
                )
                if detected_intent is not None:
                    intent_detected_via_api_fallback = False
                if detected_intent is None:
                    fallback_intent, fallback_error = await self._get_intent_from_api_fallback(
                        client=client,
                        conversation=conversation,
                    )
                    if fallback_intent is not None:
                        detected_intent = fallback_intent
                        intent_detected_via_api_fallback = True
                    elif fallback_error is not None:
                        intent_fallback_error = fallback_error

                if detected_intent is None:
                    if intent_fallback_error:
                        explanation = (
                            f"Expected intent '{expected_intent}' was not found in agent responses, "
                            f"and Conversations API fallback did not resolve it. "
                            f"Details: {intent_fallback_error}"
                        )
                    elif self._has_intent_api_fallback_config():
                        explanation = (
                            f"Expected intent '{expected_intent}' was not found in agent responses "
                            f"or Conversations API participant attributes."
                        )
                    else:
                        explanation = (
                            f"Expected intent '{expected_intent}' was not found in agent responses. "
                            f"Add a test-mode bot reply like 'INTENT={expected_intent}', or set "
                            "GC_CLIENT_ID and GC_CLIENT_SECRET to enable Conversations API fallback."
                        )
                    success = False
                else:
                    success = detected_intent == expected_intent
                    explanation = self._intent_result_explanation(
                        expected_intent,
                        detected_intent,
                        from_api_fallback=intent_detected_via_api_fallback,
                    )

                return await finalize_attempt_result(
                    success=success,
                    explanation=explanation,
                )

            if journey_mode:
                journey_success, journey_explanation, journey_validation_result, journey_error = (
                    await self._evaluate_journey_attempt(
                        scenario=scenario,
                        conversation=conversation,
                        client=client,
                    )
                )
                self._emit_attempt_status(
                    "Journey evaluation completed"
                )
                (
                    journey_success,
                    journey_explanation,
                    judging_mechanics_result,
                    mechanics_error,
                ) = self._apply_journey_judging_mechanics(
                    journey_result=journey_validation_result,
                    hard_gate_success=journey_success,
                    explanation=journey_explanation,
                )
                if self._is_judging_mechanics_enabled():
                    self._emit_attempt_status(
                        "Applied judging mechanics scoring to journey evaluation"
                    )
                return await finalize_attempt_result(
                    success=journey_success,
                    explanation=journey_explanation,
                    error=mechanics_error or journey_error,
                    journey_validation_result=journey_validation_result,
                    judging_mechanics_result=judging_mechanics_result,
                )

            # Final evaluation
            if early_success:
                success, explanation, judging_mechanics_result = (
                    self._apply_goal_judging_mechanics(
                        evaluation=evaluation,
                        hard_gate_success=True,
                        explanation=evaluation.explanation,
                    )
                )
                if self._is_judging_mechanics_enabled():
                    self._emit_attempt_status(
                        "Applied judging mechanics scoring to goal evaluation"
                    )
                return await finalize_attempt_result(
                    success=success,
                    explanation=explanation,
                    judging_mechanics_result=judging_mechanics_result,
                )

            # Reached max turns — do final evaluation
            self._emit_attempt_status("Running final goal evaluation with Judge LLM")
            evaluation = await self._run_judge_step(
                "Running final goal evaluation with Judge LLM",
                self.judge.evaluate_goal,
                persona=scenario.persona,
                goal=scenario.goal,
                conversation_history=conversation,
                language_code=self._evaluation_results_language_code(),
            )
            self._emit_attempt_status("Final Judge LLM evaluation completed")
            success, explanation, judging_mechanics_result = (
                self._apply_goal_judging_mechanics(
                    evaluation=evaluation,
                    hard_gate_success=evaluation.success,
                    explanation=evaluation.explanation,
                )
            )
            if self._is_judging_mechanics_enabled():
                self._emit_attempt_status(
                    "Applied judging mechanics scoring to goal evaluation"
                )

            return await finalize_attempt_result(
                success=success,
                explanation=explanation,
                judging_mechanics_result=judging_mechanics_result,
            )

        except StepTimeoutError as e:
            self._emit_attempt_status(f"Step timeout triggered skip: {e}")
            timeout_diagnostics = self._build_timeout_diagnostics(
                timeout_class="step_timeout",
                attempt_start_monotonic=attempt_start_monotonic,
                conversation=conversation,
                client=client,
                step_name=e.step_name,
                step_timeout_seconds=e.timeout_seconds,
            )
            return await finalize_attempt_result(
                success=False,
                explanation=(
                    "Attempt skipped because a step exceeded the time limit "
                    f"({e.step_name}, {e.timeout_seconds:.0f}s)."
                ),
                error=str(e),
                skipped=True,
                timeout_diagnostics=timeout_diagnostics,
            )
        except StopRequestedError as e:
            self._emit_attempt_status(f"Attempt interrupted by stop request: {e}")
            return await finalize_attempt_result(
                success=False,
                explanation="Attempt stopped by user request",
                error=str(e),
                skipped=True,
            )
        except GreetingGateTimeoutError as e:
            timeout_diagnostics = self._build_timeout_diagnostics(
                timeout_class="greeting_gate",
                attempt_start_monotonic=attempt_start_monotonic,
                conversation=conversation,
                client=client,
                step_name="Waiting for expected greeting before sending first user message",
                step_timeout_seconds=e.timeout_seconds,
            )
            return await finalize_attempt_result(
                success=False,
                explanation=(
                    "Attempt timed out waiting for expected greeting before the first "
                    "scenario user message"
                ),
                error=f"{e} ({e.timeout_seconds:.1f}s)",
                timed_out=True,
                timeout_diagnostics=timeout_diagnostics,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            timeout_diagnostics = self._build_timeout_diagnostics(
                timeout_class=self._classify_timeout_class(
                    error=e,
                    last_step_name=(
                        str((self._active_timeout_context or {}).get("last_step_name") or "")
                        or None
                    ),
                ),
                attempt_start_monotonic=attempt_start_monotonic,
                conversation=conversation,
                client=client,
            )
            return await finalize_attempt_result(
                success=False,
                explanation="Attempt failed due to timeout",
                error=str(e),
                timed_out=True,
                timeout_diagnostics=timeout_diagnostics,
            )
        except WebMessagingError as e:
            return await finalize_attempt_result(
                success=False,
                explanation="Attempt failed due to web messaging error",
                error=str(e),
            )
        except JudgeLLMError as e:
            self._emit_attempt_status(f"Judge LLM error: {e}")
            if "timed out" in str(e).lower():
                timeout_diagnostics = self._build_timeout_diagnostics(
                    timeout_class="judge_timeout",
                    attempt_start_monotonic=attempt_start_monotonic,
                    conversation=conversation,
                    client=client,
                )
                return await finalize_attempt_result(
                    success=False,
                    explanation="Attempt failed due to timeout",
                    error=str(e),
                    timed_out=True,
                    timeout_diagnostics=timeout_diagnostics,
                )
            return await finalize_attempt_result(
                success=False,
                explanation="Attempt failed due to Judge LLM error",
                error=str(e),
            )
        finally:
            self._active_status_callback = None
            self._active_step_log = None
            self._active_timeout_context = None
            await client.disconnect()
