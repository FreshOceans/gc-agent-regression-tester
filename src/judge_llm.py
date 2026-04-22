"""Judge LLM client for communicating with Ollama to drive conversations and evaluate goals."""

import json
from typing import Optional

import requests

from .language_profiles import get_language_profile, normalize_language_code
from .models import (
    ContinueDecision,
    GoalEvaluation,
    JourneyValidationResult,
    Message,
    MessageRole,
)


class JudgeLLMError(Exception):
    """Raised when the Judge LLM encounters an error (connection, parsing, etc.)."""

    pass


class JudgeLLMClient:
    """Client for interacting with Ollama to generate user messages and evaluate goals."""

    def __init__(self, base_url: str, model: str, timeout: int = 120):
        """Initialize with Ollama connection details.

        Args:
            base_url: The base URL of the Ollama instance (e.g., http://localhost:11434).
            model: The name of the model to use for generation.
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def verify_connection(self) -> None:
        """Check Ollama is reachable and model is available via HTTP GET to /api/tags.

        Raises:
            JudgeLLMError: If Ollama is unreachable or the model is not available.
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise JudgeLLMError(
                f"Failed to connect to Ollama at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise JudgeLLMError(
                f"Invalid response from Ollama at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        models = data.get("models", [])
        available_names = [m.get("name", "") for m in models]
        # Check both exact match and match without tag (e.g., "llama3" matches "llama3:latest")
        model_found = any(
            self.model == name or self.model == name.split(":")[0]
            for name in available_names
        )

        if not model_found:
            raise JudgeLLMError(
                f"Model '{self.model}' not found at Ollama instance {self.base_url}. "
                f"Available models: {available_names}"
            )

    def _is_gemma4_model(self) -> bool:
        return str(self.model or "").strip().lower().startswith("gemma4:")

    def _chat_options_for_operation(self, operation: str) -> Optional[dict]:
        if not self._is_gemma4_model():
            return None

        normalized = str(operation or "").strip().lower()
        if normalized == "generate_user_message":
            return {"temperature": 0.6, "top_p": 0.95}
        if normalized == "should_continue":
            return {"temperature": 0.2, "top_p": 0.9}
        if normalized in {
            "evaluate_goal",
            "classify_primary_category",
            "infer_containment",
            "evaluate_journey",
            "extract_conversation_id",
        }:
            return {"temperature": 0.0, "top_p": 0.9, "seed": 42}
        if normalized == "warm_up":
            return {"temperature": 0.0}
        return None

    def generate_user_message(
        self,
        persona: str,
        goal: str,
        conversation_history: list[Message],
        language_code: str = "en",
    ) -> str:
        """Generate the next user message given persona, goal, and conversation history.

        The initial prompt contains the persona, goal, AND the agent's welcome message.
        Subsequent prompts include the full conversation history.

        Args:
            persona: The persona description for the simulated user.
            goal: The goal the simulated user is trying to achieve.
            conversation_history: The conversation so far (starts with agent's welcome message).

        Returns:
            The generated user message text.

        Raises:
            JudgeLLMError: If the LLM response cannot be parsed or the request fails.
        """
        language_instruction = self._language_instruction(language_code)
        if self._is_gemma4_model():
            system_prompt = (
                "Write the next customer message in a live support chat.\n\n"
                f"PERSONA: {persona}\n\n"
                f"GOAL: {goal}\n\n"
                f"LANGUAGE: {language_instruction}\n\n"
                "Rules:\n"
                "- Keep it short.\n"
                "- Answer directly when asked for details.\n"
                "- Stay on goal.\n"
                "Return only the next user message."
            )
        else:
            system_prompt = (
                "You are pretending to be a customer talking to a service agent. "
                "Be direct and straightforward — don't overthink your responses.\n\n"
                f"WHO YOU ARE: {persona}\n\n"
                f"WHAT YOU WANT: {goal}\n\n"
                f"LANGUAGE: {language_instruction}\n\n"
                "RULES:\n"
                "- Keep your messages short and simple, like a real person would text.\n"
                "- When the agent asks for information (account numbers, auth codes, names, etc.), "
                "provide it directly from your persona details.\n"
                "- Don't be overly polite or verbose. Just answer naturally.\n"
                "- Stay focused on achieving your goal.\n\n"
                "Output ONLY the next message. No labels, no quotes, no explanation."
            )

        messages = [{"role": "system", "content": system_prompt}]

        # Build conversation context from history
        for msg in conversation_history:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})

        # Add instruction to generate next user message
        messages.append(
            {
                "role": "user",
                "content": "Generate the next user message to continue working toward the goal.",
            }
        )

        response_text = self._call_chat(messages, operation="generate_user_message")
        return response_text.strip()

    def should_continue(
        self,
        persona: str,
        goal: str,
        conversation_history: list[Message],
        language_code: str = "en",
    ) -> ContinueDecision:
        """Determine if the conversation should continue, goal is achieved, or goal is unachievable.

        Args:
            persona: The persona description for the simulated user.
            goal: The goal the simulated user is trying to achieve.
            conversation_history: The full conversation history.

        Returns:
            ContinueDecision indicating whether to continue and if the goal was achieved.

        Raises:
            JudgeLLMError: If the LLM response cannot be parsed or the request fails.
        """
        language_instruction = self._language_instruction(language_code)
        if self._is_gemma4_model():
            system_prompt = (
                "Decide whether the conversation should continue.\n\n"
                f"GOAL: {goal}\n\n"
                f"LANGUAGE: {language_instruction}\n"
                "Use that language for explanation text values only. JSON keys stay English.\n\n"
                "Return one JSON object only:\n"
                '{"should_continue":true,"goal_achieved":null,"explanation":"..."}\n'
                "Use goal_achieved=true or false only when should_continue=false."
            )
        else:
            system_prompt = (
                "You are deciding if a customer's goal has been achieved in a conversation.\n\n"
                f"GOAL: {goal}\n\n"
                f"LANGUAGE: {language_instruction}\n"
                "Use that language for explanation text values only. JSON keys must stay in English.\n\n"
                "Look at the LAST agent message and decide:\n\n"
                "STOP (goal achieved) — The agent has provided the answer, confirmed the action, "
                "or delivered what the goal asked for. Examples: balance shown, appointment confirmed, "
                "transfer completed, password reset sent.\n\n"
                "CONTINUE — The agent is asking for information needed to fulfill the goal "
                "(login code, account number, verification, etc). This is normal progress.\n\n"
                "STOP (goal failed) — The agent has explicitly refused, said it cannot help, "
                "or the request is clearly impossible.\n\n"
                "IMPORTANT: Once the goal is achieved, STOP immediately. Do not continue for "
                "pleasantries, follow-up offers, or 'anything else?' questions.\n\n"
                "Respond with ONLY valid JSON, nothing else:\n"
                '{"should_continue": true, "goal_achieved": null, "explanation": "..."}\n'
                '{"should_continue": false, "goal_achieved": true, "explanation": "..."}\n'
                '{"should_continue": false, "goal_achieved": false, "explanation": "..."}'
            )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in conversation_history:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})

        messages.append(
            {
                "role": "user",
                "content": "Should this conversation continue? Respond with JSON only.",
            }
        )

        response_text = self._call_chat(messages, operation="should_continue")
        return self._parse_continue_decision(response_text)

    def evaluate_goal(
        self,
        persona: str,
        goal: str,
        conversation_history: list[Message],
        language_code: str = "en",
    ) -> GoalEvaluation:
        """Evaluate whether the goal was achieved in the conversation.

        Args:
            persona: The persona description for the simulated user.
            goal: The goal the simulated user was trying to achieve.
            conversation_history: The full conversation history.

        Returns:
            GoalEvaluation with success/failure and explanation.

        Raises:
            JudgeLLMError: If the LLM response cannot be parsed or the request fails.
        """
        language_instruction = self._language_instruction(language_code)
        if self._is_gemma4_model():
            system_prompt = (
                "Evaluate whether the goal was achieved.\n\n"
                f"PERSONA: {persona}\n\n"
                f"GOAL: {goal}\n\n"
                f"LANGUAGE: {language_instruction}\n"
                "Use that language for explanation text values only. JSON keys stay English.\n\n"
                "Return exactly one JSON object:\n"
                '{"success":true,"explanation":"..."}'
            )
        else:
            system_prompt = (
                "You are evaluating whether a conversation achieved its goal.\n\n"
                f"PERSONA: {persona}\n\n"
                f"GOAL: {goal}\n\n"
                f"LANGUAGE: {language_instruction}\n"
                "Use that language for explanation text values only. JSON keys must stay in English.\n\n"
                "Review the conversation and determine if the goal was successfully achieved.\n"
                "Respond with a JSON object with these fields:\n"
                '- "success": boolean - true if the goal was achieved, false otherwise\n'
                '- "explanation": string - brief explanation of why the goal was or was not achieved\n\n'
                "Respond ONLY with the JSON object, no other text."
            )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in conversation_history:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})

        messages.append(
            {
                "role": "user",
                "content": "Was the goal achieved? Respond with JSON only.",
            }
        )

        response_text = self._call_chat(messages, operation="evaluate_goal")
        return self._parse_goal_evaluation(response_text)

    def classify_primary_category(
        self,
        *,
        first_message: str,
        categories: list[dict],
        language_code: str = "en",
    ) -> dict:
        """Classify first utterance into one of the configured primary categories."""
        language_instruction = self._language_instruction(language_code)
        category_lines = []
        for row in categories:
            name = str(row.get("name") or "").strip().lower()
            if not name:
                continue
            keywords = row.get("keywords") or []
            rubric = str(row.get("rubric") or "").strip()
            category_lines.append(
                f"- {name}: keywords={keywords} rubric={rubric or 'n/a'}"
            )
        category_block = "\n".join(category_lines) or "- general"
        system_prompt = (
            "Classify the customer's first utterance into one primary journey category.\n\n"
            f"LANGUAGE: {language_instruction}\n"
            "Use that language for explanation text only. JSON keys stay English.\n\n"
            "Valid categories:\n"
            f"{category_block}\n\n"
            "Return exactly one JSON object:\n"
            '{"category":"<category-or-null>","confidence":0.0,"explanation":"..."}'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"First customer utterance: {first_message}",
            },
        ]
        response_text = self._call_chat(messages, operation="classify_primary_category")
        parsed = self._parse_json_payload(response_text, "primary category")
        category = parsed.get("category")
        confidence = parsed.get("confidence")
        explanation = str(parsed.get("explanation") or "").strip() or None
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        return {
            "category": str(category).strip().lower() if isinstance(category, str) else None,
            "confidence": confidence_value,
            "explanation": explanation,
        }

    def infer_containment(
        self,
        *,
        conversation_history: list[Message],
        language_code: str = "en",
    ) -> dict:
        """Infer whether the journey stayed contained in automation."""
        language_instruction = self._language_instruction(language_code)
        system_prompt = (
            "Infer whether the customer journey stayed contained in automation.\n\n"
            f"LANGUAGE: {language_instruction}\n"
            "Use that language for explanation text only. JSON keys stay English.\n\n"
            "Return exactly one JSON object:\n"
            '{"contained":true,"confidence":0.0,"explanation":"..."}'
        )
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})
        messages.append(
            {
                "role": "user",
                "content": "Infer containment now. Return JSON only.",
            }
        )
        response_text = self._call_chat(messages, operation="infer_containment")
        parsed = self._parse_json_payload(response_text, "containment inference")
        contained = parsed.get("contained")
        confidence = parsed.get("confidence")
        explanation = str(parsed.get("explanation") or "").strip() or ""
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        return {
            "contained": bool(contained) if isinstance(contained, bool) else None,
            "confidence": confidence_value,
            "explanation": explanation,
        }

    def evaluate_journey(
        self,
        *,
        persona: str,
        goal: str,
        expected_category: Optional[str],
        path_rubric: Optional[str],
        category_rubric: Optional[str],
        conversation_history: list[Message],
        language_code: str = "en",
        known_contained: Optional[bool] = None,
    ) -> JourneyValidationResult:
        """Evaluate full journey quality and containment correctness."""
        language_instruction = self._language_instruction(language_code)
        expected = str(expected_category or "").strip().lower() or "none"
        known_contained_block = (
            "Containment signal from metadata is unavailable."
            if known_contained is None
            else (
                f"Containment signal from metadata is authoritative: contained={str(known_contained).lower()}."
            )
        )
        system_prompt = (
            "Evaluate the full customer journey.\n\n"
            f"PERSONA: {persona}\n\n"
            f"GOAL: {goal}\n\n"
            f"EXPECTED_PRIMARY_CATEGORY: {expected}\n\n"
            f"PATH_RUBRIC: {path_rubric or 'n/a'}\n\n"
            f"CATEGORY_RUBRIC_OVERRIDE: {category_rubric or 'n/a'}\n\n"
            f"LANGUAGE: {language_instruction}\n"
            "Use that language for explanation text only. JSON keys stay English.\n\n"
            f"{known_contained_block}\n\n"
            "Return exactly one JSON object:\n"
            '{"category_match":true,"fulfilled":true,"path_correct":true,"contained":true,"actual_category":"...","confidence":0.0,"explanation":"..."}'
        )
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})
        messages.append(
            {
                "role": "user",
                "content": "Evaluate journey now. Return JSON only.",
            }
        )
        response_text = self._call_chat(messages, operation="evaluate_journey")
        parsed = self._parse_json_payload(response_text, "journey evaluation")
        confidence = parsed.get("confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None

        contained_value = parsed.get("contained")
        contained = (
            bool(contained_value)
            if isinstance(contained_value, bool)
            else None
        )

        return JourneyValidationResult(
            category_match=(
                bool(parsed.get("category_match"))
                if isinstance(parsed.get("category_match"), bool)
                else None
            ),
            fulfilled=bool(parsed.get("fulfilled")),
            path_correct=bool(parsed.get("path_correct")),
            contained=contained,
            expected_category=(expected_category or None),
            actual_category=(
                str(parsed.get("actual_category")).strip().lower()
                if isinstance(parsed.get("actual_category"), str)
                and str(parsed.get("actual_category")).strip()
                else None
            ),
            confidence=confidence_value,
            explanation=str(parsed.get("explanation") or "").strip(),
        )

    def warm_up(
        self,
        prompt: Optional[str] = None,
        language_code: str = "en",
    ) -> str:
        """Warm up the configured model with a lightweight chat request.

        This helps avoid first-request latency for large local models.

        Args:
            prompt: Small user prompt used for warm-up.

        Returns:
            The model response text.

        Raises:
            JudgeLLMError: If the chat call fails.
        """
        warmup_prompt = prompt
        if warmup_prompt is None:
            canonical = normalize_language_code(language_code, default="en")
            if canonical in {"fr", "fr-CA"}:
                warmup_prompt = "Repondez uniquement avec OK."
            elif canonical == "es":
                warmup_prompt = "Responde solo con OK."
            else:
                warmup_prompt = "Reply with OK."
        messages = [{"role": "user", "content": warmup_prompt}]
        return self._call_chat(messages, operation="warm_up").strip()

    def extract_conversation_id(
        self,
        conversation_history: list[Message],
        language_code: str = "en",
    ) -> Optional[str]:
        """Ask the Judge LLM to extract a conversation id from the transcript, if present."""
        language_instruction = self._language_instruction(language_code)
        system_prompt = (
            "You extract a Genesys conversation id from chat transcript text.\n\n"
            f"LANGUAGE: {language_instruction}\n"
            "Rules:\n"
            "- Return ONLY the conversation id string if found.\n"
            "- If no conversation id appears in the transcript, return ONLY: NONE\n"
            "- Do not add explanation.\n"
            "- Prefer values labelled conversationId or conversation_id.\n"
            "- If multiple ids exist, return the most recent conversation id.\n"
        )
        messages = [{"role": "system", "content": system_prompt}]

        for msg in conversation_history:
            role = "assistant" if msg.role == MessageRole.AGENT else "user"
            messages.append({"role": role, "content": msg.content})

        messages.append(
            {
                "role": "user",
                "content": "Extract and return only the conversation id or NONE.",
            }
        )
        response_text = self._call_chat(messages, operation="extract_conversation_id").strip()
        if not response_text or response_text.upper() == "NONE":
            return None
        # Trim common wrapping quotes/fences if model added them.
        return response_text.strip("`\"' ").strip() or None

    def _language_instruction(self, language_code: Optional[str]) -> str:
        profile = get_language_profile(language_code)
        return str(profile.get("judge_language_instruction", "Use English for natural-language outputs."))

    def _call_chat(self, messages: list[dict], *, operation: str = "chat") -> str:
        """Call the Ollama /api/chat endpoint and return the response content.

        Args:
            messages: The messages to send to the chat API.

        Returns:
            The response content text.

        Raises:
            JudgeLLMError: If the request fails or response is invalid.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        options = self._chat_options_for_operation(operation)
        if options:
            payload["options"] = options

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise JudgeLLMError(
                f"Failed to call Ollama chat API at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise JudgeLLMError(
                f"Invalid JSON response from Ollama at {self.base_url} "
                f"for model '{self.model}': {e}"
            )

        message = data.get("message", {})
        content = message.get("content", "")

        if not content:
            raise JudgeLLMError(
                f"Empty response from Ollama at {self.base_url} "
                f"for model '{self.model}'"
            )

        return content

    def _parse_json_payload(self, response_text: str, label: str) -> dict:
        json_str = self._extract_json(response_text)
        try:
            payload = json.loads(json_str)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise JudgeLLMError(
                f"Failed to parse {label} payload from LLM response: {e}. "
                f"Response was: {response_text[:200]}"
            ) from e
        if not isinstance(payload, dict):
            raise JudgeLLMError(
                f"Failed to parse {label} payload from LLM response: expected JSON object."
            )
        return payload

    def _parse_continue_decision(self, response_text: str) -> ContinueDecision:
        """Parse a ContinueDecision from LLM response text.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            A validated ContinueDecision object.

        Raises:
            JudgeLLMError: If the response cannot be parsed as valid JSON or doesn't match schema.
        """
        try:
            data = self._parse_json_payload(response_text, "ContinueDecision")
            return ContinueDecision(**data)
        except (TypeError, ValueError) as e:
            raise JudgeLLMError(
                f"Failed to parse ContinueDecision from LLM response: {e}. "
                f"Response was: {response_text[:200]}"
            )

    def _parse_goal_evaluation(self, response_text: str) -> GoalEvaluation:
        """Parse a GoalEvaluation from LLM response text.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            A validated GoalEvaluation object.

        Raises:
            JudgeLLMError: If the response cannot be parsed as valid JSON or doesn't match schema.
        """
        try:
            data = self._parse_json_payload(response_text, "GoalEvaluation")
            return GoalEvaluation(**data)
        except (TypeError, ValueError) as e:
            raise JudgeLLMError(
                f"Failed to parse GoalEvaluation from LLM response: {e}. "
                f"Response was: {response_text[:200]}"
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown code fences or extra whitespace.

        Args:
            text: The raw text that should contain JSON.

        Returns:
            The extracted JSON string.
        """
        text = text.strip()
        # Handle markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        # Fix case-sensitive boolean values (LLMs sometimes output TRUE/FALSE/True/False)
        import re
        text = re.sub(r'\bTRUE\b', 'true', text)
        text = re.sub(r'\bFALSE\b', 'false', text)
        text = re.sub(r'\bNULL\b', 'null', text)
        text = re.sub(r':\s*True\b', ': true', text)
        text = re.sub(r':\s*False\b', ': false', text)
        text = re.sub(r':\s*None\b', ': null', text)
        return text
