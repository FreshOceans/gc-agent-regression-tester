"""Conversation Runner for executing single test attempts."""

import asyncio
import inspect
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from .genesys_conversations_client import (
    GenesysConversationsClient,
    GenesysConversationsError,
)
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import (
    AttemptResult,
    Message,
    MessageRole,
    TestScenario,
)
from .web_messaging_client import WebMessagingClient, WebMessagingError


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

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        normalized = normalized.replace("’", "'")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _is_expected_greeting(self, message: str) -> bool:
        expected = (self.web_msg_config.get("expected_greeting") or "").strip()
        if not expected:
            return True
        expected_norm = self._normalize_text(expected)
        message_norm = self._normalize_text(message)
        return expected_norm == message_norm or expected_norm in message_norm

    def _is_presence_unsupported_message(self, message: str) -> bool:
        return "presence events are not supported" in self._normalize_text(message)

    def _extract_intent_from_text(self, text: str) -> Optional[str]:
        """Extract intent from test-mode agent text (e.g., INTENT=flight_cancel)."""
        stripped = text.strip()
        if not stripped:
            return None

        # JSON format, e.g. {"intent": "flight_cancel"}
        try:
            data = json.loads(stripped)
            intent_value = data.get("intent") if isinstance(data, dict) else None
            if isinstance(intent_value, str) and intent_value.strip():
                return intent_value.strip().lower()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Plain text format, e.g. INTENT=flight_cancel or intent: flight_cancel
        match = re.search(
            r"\bintent\b\s*[:=]\s*['\"]?([a-zA-Z0-9_\-./]+)",
            stripped,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().lower()

        return None

    def _find_detected_intent(self, conversation: list[Message]) -> Optional[str]:
        """Find most recent detected intent in agent messages."""
        for msg in reversed(conversation):
            if msg.role != MessageRole.AGENT:
                continue
            detected = self._extract_intent_from_text(msg.content)
            if detected:
                return detected
        return None

    def _has_intent_api_fallback_config(self) -> bool:
        return bool(
            (self.web_msg_config.get("region") or "").strip()
            and (self.web_msg_config.get("gc_client_id") or "").strip()
            and (self.web_msg_config.get("gc_client_secret") or "").strip()
        )

    def _should_judge_capture_conversation_id(self, scenario: TestScenario) -> bool:
        if scenario.judge_capture_conversation_id is not None:
            return bool(scenario.judge_capture_conversation_id)
        return bool(self.web_msg_config.get("judge_capture_conversation_id", True))

    def _resolve_conversation_ids_for_fallback(
        self,
        client: WebMessagingClient,
        scenario: TestScenario,
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

        if not self._should_judge_capture_conversation_id(scenario):
            return [], (
                "Web Messaging did not provide a conversation ID and "
                "judge conversation-id capture is disabled."
            )

        try:
            inferred_conversation_id = self.judge.extract_conversation_id(
                conversation_history=conversation
            )
        except JudgeLLMError as e:
            return [], (
                "Web Messaging did not provide a conversation ID, and the judge "
                f"failed to infer one from transcript text: {e}"
            )

        if inferred_conversation_id:
            if not self._is_valid_conversation_id(inferred_conversation_id):
                return [], (
                    "Judge inferred a value, but it does not match a valid "
                    "conversation-id format."
                )
            return [inferred_conversation_id], None

        return [], (
            "Web Messaging did not provide a conversation ID, and the judge could not "
            "infer one from transcript text."
        )

    def _is_valid_conversation_id(self, value: str) -> bool:
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError, TypeError):
            return False
        return str(parsed) == value.lower()

    async def _get_intent_from_api_fallback(
        self,
        client: WebMessagingClient,
        scenario: TestScenario,
        conversation: list[Message],
    ) -> tuple[Optional[str], Optional[str]]:
        """Try Conversations API participant-attribute fallback for intent."""
        if not self._has_intent_api_fallback_config():
            return None, None

        conversation_ids, conversation_id_error = self._resolve_conversation_ids_for_fallback(
            client=client,
            scenario=scenario,
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
        last_error: Optional[str] = None
        for conversation_id in conversation_ids:
            try:
                detected_intent = await asyncio.to_thread(
                    conversations_client.get_participant_attribute,
                    conversation_id=conversation_id,
                    attribute_name=intent_attribute_name,
                    participant_id=client.participant_id,
                    retries=retries,
                    retry_delay_seconds=retry_delay_seconds,
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

    async def run_attempt(self, scenario: TestScenario, attempt_number: int) -> AttemptResult:
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
        expected_intent = (
            scenario.expected_intent.strip().lower()
            if scenario.expected_intent
            else None
        )
        started_at = self._now_utc()
        attempt_start_monotonic = time.monotonic()
        client = WebMessagingClient(
            region=self.web_msg_config["region"],
            deployment_id=self.web_msg_config["deployment_id"],
            timeout=self.web_msg_config.get("timeout", 30),
            origin=self.web_msg_config.get("origin", "https://localhost"),
            debug_capture_frames=self.web_msg_config.get("debug_capture_frames", False),
            debug_capture_frame_limit=self.web_msg_config.get("debug_capture_frame_limit", 8),
        )

        def build_attempt_result(
            *,
            success: bool,
            explanation: str,
            error: Optional[str] = None,
            timed_out: bool = False,
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
                detected_intent=detected_intent,
                started_at=started_at,
                completed_at=self._now_utc(),
                duration_seconds=time.monotonic() - attempt_start_monotonic,
                turn_durations_seconds=turn_durations_seconds,
                debug_frames=debug_frames,
            )

        try:
            # Connect, send join event, and wait for welcome message
            await client.connect()
            await client.send_join()
            welcome_text = await client.wait_for_welcome()

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
                await client.send_message(bootstrap_message)
                bootstrap_response = await client.receive_response()
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

            # Try to wait briefly for the expected greeting before sending test input.
            max_agent_messages_before_first_user = 5
            greeting_wait_timeout = min(
                self.web_msg_config.get("timeout", 30),
                self.web_msg_config.get("greeting_wait_timeout_seconds", 8),
            )
            deadline = time.monotonic() + greeting_wait_timeout
            while (
                not self._is_expected_greeting(conversation[-1].content)
                and len([m for m in conversation if m.role == MessageRole.AGENT])
                < max_agent_messages_before_first_user
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    agent_text = await asyncio.wait_for(
                        client.receive_response(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    break

                conversation.append(
                    Message(
                        role=MessageRole.AGENT,
                        content=agent_text,
                        timestamp=self._now_utc(),
                    )
                )
                greeting_detected = self._extract_intent_from_text(agent_text)
                if greeting_detected is not None:
                    detected_intent = greeting_detected
                    intent_detected_via_api_fallback = False

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
                    user_message = self.judge.generate_user_message(
                        persona=scenario.persona,
                        goal=scenario.goal,
                        conversation_history=conversation,
                    )
                user_sent_monotonic = time.monotonic()
                conversation.append(
                    Message(
                        role=MessageRole.USER,
                        content=user_message,
                        timestamp=self._now_utc(),
                    )
                )

                # Send to agent and receive response
                await client.send_message(user_message)
                try:
                    agent_response = await client.receive_response()
                except TimeoutError as timeout_error:
                    if expected_intent is not None:
                        fallback_intent, fallback_error = await self._get_intent_from_api_fallback(
                            client=client,
                            scenario=scenario,
                            conversation=conversation,
                        )
                        if fallback_intent is not None:
                            detected_intent = fallback_intent
                            intent_detected_via_api_fallback = True
                            matched = detected_intent == expected_intent
                            return build_attempt_result(
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
                        return build_attempt_result(
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
                        return build_attempt_result(
                            success=matched,
                            explanation=self._intent_result_explanation(
                                expected_intent,
                                response_detected_intent,
                                from_api_fallback=False,
                            ),
                        )

                    fallback_intent, fallback_error = await self._get_intent_from_api_fallback(
                        client=client,
                        scenario=scenario,
                        conversation=conversation,
                    )
                    if fallback_intent is not None:
                        detected_intent = fallback_intent
                        intent_detected_via_api_fallback = True
                        matched = detected_intent == expected_intent
                        return build_attempt_result(
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
                    evaluation = self.judge.evaluate_goal(
                        persona=scenario.persona,
                        goal=scenario.goal,
                        conversation_history=conversation,
                    )
                    if evaluation.success:
                        early_success = True
                        break
                except JudgeLLMError:
                    pass  # If evaluation fails mid-conversation, keep going

            if expected_intent is not None:
                detected_intent = self._find_detected_intent(conversation)
                if detected_intent is not None:
                    intent_detected_via_api_fallback = False
                if detected_intent is None:
                    fallback_intent, fallback_error = await self._get_intent_from_api_fallback(
                        client=client,
                        scenario=scenario,
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

                return build_attempt_result(
                    success=success,
                    explanation=explanation,
                )

            # Final evaluation
            if early_success:
                return build_attempt_result(
                    success=True,
                    explanation=evaluation.explanation,
                )

            # Reached max turns — do final evaluation
            evaluation = self.judge.evaluate_goal(
                persona=scenario.persona,
                goal=scenario.goal,
                conversation_history=conversation,
            )

            return build_attempt_result(
                success=evaluation.success,
                explanation=evaluation.explanation,
            )

        except TimeoutError as e:
            return build_attempt_result(
                success=False,
                explanation="Attempt failed due to timeout",
                error=str(e),
                timed_out=True,
            )
        except WebMessagingError as e:
            return build_attempt_result(
                success=False,
                explanation="Attempt failed due to web messaging error",
                error=str(e),
            )
        except JudgeLLMError as e:
            return build_attempt_result(
                success=False,
                explanation="Attempt failed due to Judge LLM error",
                error=str(e),
            )
        finally:
            await client.disconnect()
