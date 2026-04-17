"""Conversation Runner for executing single test attempts."""

import asyncio

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
            conversation.append(Message(role=MessageRole.AGENT, content=welcome_text))

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
                conversation.append(Message(role=MessageRole.USER, content=user_message))

                # Send to agent and receive response
                await client.send_message(user_message)
                agent_response = await client.receive_response()
                conversation.append(Message(role=MessageRole.AGENT, content=agent_response))

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
            )

        except TimeoutError as e:
            return AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation="Attempt failed due to timeout",
                error=str(e),
            )
        except WebMessagingError as e:
            return AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation="Attempt failed due to web messaging error",
                error=str(e),
            )
        except JudgeLLMError as e:
            return AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation="Attempt failed due to Judge LLM error",
                error=str(e),
            )
        finally:
            await client.disconnect()
