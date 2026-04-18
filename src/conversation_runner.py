"""Conversation Runner for executing single test attempts."""

import asyncio
import re
import time
from datetime import datetime, timezone

from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import (
    AttemptResult,
    ContinueDecision,
    GoalEvaluation,
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
        started_at = self._now_utc()
        attempt_start_monotonic = time.monotonic()
        client = WebMessagingClient(
            region=self.web_msg_config["region"],
            deployment_id=self.web_msg_config["deployment_id"],
            timeout=self.web_msg_config.get("timeout", 30),
            origin=self.web_msg_config.get("origin", "https://localhost"),
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

            # Conversation loop — keep going until goal achieved or max turns
            turn_count = 0
            first_turn = True
            early_success = False
            while turn_count < self.max_turns:
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
                agent_response = await client.receive_response()
                turn_durations_seconds.append(time.monotonic() - user_sent_monotonic)
                conversation.append(
                    Message(
                        role=MessageRole.AGENT,
                        content=agent_response,
                        timestamp=self._now_utc(),
                    )
                )

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

            # Final evaluation
            if early_success:
                return AttemptResult(
                    attempt_number=attempt_number,
                    success=True,
                    conversation=conversation,
                    explanation=evaluation.explanation,
                    timed_out=False,
                    started_at=started_at,
                    completed_at=self._now_utc(),
                    duration_seconds=time.monotonic() - attempt_start_monotonic,
                    turn_durations_seconds=turn_durations_seconds,
                )

            # Reached max turns — do final evaluation
            evaluation = self.judge.evaluate_goal(
                persona=scenario.persona,
                goal=scenario.goal,
                conversation_history=conversation,
            )

            return AttemptResult(
                attempt_number=attempt_number,
                success=evaluation.success,
                conversation=conversation,
                explanation=evaluation.explanation,
                timed_out=False,
                started_at=started_at,
                completed_at=self._now_utc(),
                duration_seconds=time.monotonic() - attempt_start_monotonic,
                turn_durations_seconds=turn_durations_seconds,
            )

        except TimeoutError as e:
            return AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation="Attempt failed due to timeout",
                error=str(e),
                timed_out=True,
                started_at=started_at,
                completed_at=self._now_utc(),
                duration_seconds=time.monotonic() - attempt_start_monotonic,
                turn_durations_seconds=turn_durations_seconds,
            )
        except WebMessagingError as e:
            return AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation="Attempt failed due to web messaging error",
                error=str(e),
                timed_out=False,
                started_at=started_at,
                completed_at=self._now_utc(),
                duration_seconds=time.monotonic() - attempt_start_monotonic,
                turn_durations_seconds=turn_durations_seconds,
            )
        except JudgeLLMError as e:
            return AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation="Attempt failed due to Judge LLM error",
                error=str(e),
                timed_out=False,
                started_at=started_at,
                completed_at=self._now_utc(),
                duration_seconds=time.monotonic() - attempt_start_monotonic,
                turn_durations_seconds=turn_durations_seconds,
            )
        finally:
            await client.disconnect()
