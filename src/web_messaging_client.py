"""Web Messaging Client for Genesys Cloud Web Messaging Guest API."""

import asyncio
import json
import re
import uuid
from typing import Any, Optional

import websockets


class WebMessagingError(Exception):
    """Custom exception for Web Messaging connection and protocol errors."""

    pass


class WebMessagingClient:
    """Client for communicating with Genesys Cloud agents via the Web Messaging Guest API.

    Manages WebSocket connections, session lifecycle, and message exchange
    with the Genesys Cloud Web Messaging protocol.
    """

    def __init__(
        self,
        region: str,
        deployment_id: str,
        timeout: int = 30,
        origin: str = "https://localhost",
        debug_capture_frames: bool = False,
        debug_capture_frame_limit: int = 8,
    ):
        """Initialize with Genesys Cloud connection details.

        Args:
            region: The Genesys Cloud region (e.g., 'mypurecloud.com').
            deployment_id: The Web Messaging deployment ID.
            timeout: Timeout in seconds for waiting on messages (default 30).
            origin: The origin header value (must match an allowed origin on the deployment).
        """
        self.region = region
        self.deployment_id = deployment_id
        self.timeout = timeout
        self.origin = origin
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._token: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.participant_id: Optional[str] = None
        self._conversation_id_candidates: list[str] = []
        self._debug_capture_frames = debug_capture_frames
        self._debug_capture_frame_limit = max(1, debug_capture_frame_limit)
        self._debug_frames: list[dict[str, Any]] = []

    _UUID_PATTERN = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
    )

    @property
    def ws_url(self) -> str:
        """Construct the WebSocket URL for the Web Messaging Guest API."""
        return f"wss://webmessaging.{self.region}/v1?deploymentId={self.deployment_id}"

    async def connect(self) -> None:
        """Establish a new WebSocket session with the Genesys Cloud Web Messaging Guest API.

        Sends a configureSession message to initialize the session.

        Raises:
            WebMessagingError: If the connection cannot be established.
                The error message includes both the deployment ID and region.
        """
        try:
            self._ws = await websockets.connect(
                self.ws_url,
                additional_headers={"Origin": self.origin},
            )
        except Exception as e:
            raise WebMessagingError(
                f"Failed to connect to Web Messaging API: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

        # Send configureSession to initialize the session
        self._token = str(uuid.uuid4())
        configure_message = {
            "action": "configureSession",
            "deploymentId": self.deployment_id,
            "token": self._token,
        }
        try:
            await self._ws.send(json.dumps(configure_message))
        except Exception as e:
            raise WebMessagingError(
                f"Failed to configure session: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

        # Wait for session confirmation
        try:
            deadline = asyncio.get_event_loop().time() + self.timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()

                response = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    continue
                self._update_conversation_metadata(data)
                self._capture_debug_frame(data, stage="connect")

                # Session established successfully
                if data.get("type") == "SessionResponse":
                    break

                # Some deployments do not emit SessionResponse consistently,
                # but if we already receive protocol messages the session is usable.
                if self._is_session_ready_fallback(data):
                    break
        except asyncio.TimeoutError:
            raise WebMessagingError(
                f"Timed out waiting for session confirmation: deployment_id={self.deployment_id}, "
                f"region={self.region}"
            )
        except Exception as e:
            raise WebMessagingError(
                f"Error during session setup: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

    def _is_session_ready_fallback(self, payload: object) -> bool:
        """Return True when a non-SessionResponse payload still proves session readiness."""
        if not isinstance(payload, dict):
            return False
        msg_type = payload.get("type")
        if msg_type == "error":
            return False
        return msg_type in {"message", "response"}

    async def wait_for_welcome(self) -> str:
        """Wait for the agent's welcome message.

        Returns:
            The text content of the agent's welcome message.

        Raises:
            TimeoutError: If no welcome message is received within the configured timeout.
            WebMessagingError: If the connection is not established or a protocol error occurs.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        try:
            text = await self._receive_agent_message()
            return text
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for welcome message after {self.timeout}s"
            )

    async def send_join(self) -> None:
        """Send a Join presence event to start the conversation.

        This triggers the bot flow to begin and send a welcome message.

        Raises:
            WebMessagingError: If the connection is not established or sending fails.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        join_message = {
            "action": "onMessage",
            "token": self._token,
            "message": {
                "type": "Event",
                "events": [
                    {
                        "eventType": "Presence",
                        "presence": {
                            "type": "Join"
                        }
                    }
                ]
            }
        }
        try:
            await self._ws.send(json.dumps(join_message))
        except Exception as e:
            raise WebMessagingError(
                f"Failed to send join event: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

    async def send_message(self, text: str) -> None:
        """Send a user message through the active session.

        Args:
            text: The message text to send.

        Raises:
            WebMessagingError: If the connection is not established or sending fails.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        message = {
            "action": "onMessage",
            "token": self._token,
            "message": {
                "type": "Text",
                "text": text,
            },
        }
        try:
            await self._ws.send(json.dumps(message))
        except Exception as e:
            raise WebMessagingError(
                f"Failed to send message: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

    async def receive_response(self) -> str:
        """Wait for and return the next agent response.

        Returns:
            The text content of the agent's response message.

        Raises:
            TimeoutError: If no response is received within the configured timeout.
            WebMessagingError: If the connection is not established or a protocol error occurs.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        try:
            text = await self._receive_agent_message()
            return text
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for agent response after {self.timeout}s"
            )

    async def disconnect(self) -> None:
        """Close the WebSocket session.

        Safe to call even if not connected.
        """
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass  # Best-effort close
            finally:
                self._ws = None
                self._token = None
                self.conversation_id = None
                self.participant_id = None
                self._conversation_id_candidates = []
                self._debug_frames = []

    async def _receive_agent_message(self) -> str:
        """Wait for and extract text from the next agent message.

        Parses incoming WebSocket messages according to the Genesys Cloud
        Web Messaging protocol, filtering for structured message types
        that contain agent text content.

        Returns:
            The extracted text content from the agent's message.

        Raises:
            asyncio.TimeoutError: If no agent message arrives within timeout.
            WebMessagingError: If a protocol error occurs.
        """
        deadline = asyncio.get_event_loop().time() + self.timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                raise

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue  # Skip non-JSON messages

            self._update_conversation_metadata(data)
            self._capture_debug_frame(data, stage="receive")

            # Handle structured message types from the Web Messaging protocol
            msg_type = data.get("type", "")
            msg_class = data.get("class", "")

            # Skip echo of our own inbound messages and events
            body = data.get("body", {})
            if isinstance(body, dict):
                direction = body.get("direction", "")
                if direction == "Inbound":
                    continue
                # Skip typing indicators and presence events
                body_type = body.get("type", "")
                if body_type == "Event":
                    continue

            # Look for agent messages in the "message" type responses
            if msg_type == "message" and msg_class == "StructuredMessage":
                if isinstance(body, dict):
                    text = body.get("text", "")
                    if text:
                        return text

            # Also handle simpler response format
            if msg_type == "message":
                # Try to extract text from body directly
                body = data.get("body", "")
                if isinstance(body, str) and body:
                    return body
                # Try nested text field
                if isinstance(body, dict):
                    text = body.get("text", "")
                    if text:
                        return text

            # Handle "response" type messages (some protocol variants)
            if msg_type == "response":
                body = data.get("body", {})
                if isinstance(body, dict):
                    text = body.get("text", "")
                    if text:
                        return text
                elif isinstance(body, str) and body:
                    return body

    def _update_conversation_metadata(self, payload: object) -> None:
        """Best-effort extraction of conversation and participant IDs from payloads."""

        if isinstance(payload, dict):
            msg_type = payload.get("type")
            msg_class = payload.get("class")
            body = payload.get("body")
            if (
                (msg_type == "SessionResponse" or msg_class == "SessionResponse")
                and isinstance(body, dict)
                and self.conversation_id is None
            ):
                # Some deployments include the conversation id as body.id on session response.
                body_id = body.get("id")
                if isinstance(body_id, str) and body_id.strip():
                    normalized_body_id = body_id.strip()
                    if self._is_likely_conversation_id(normalized_body_id):
                        self.conversation_id = normalized_body_id
                        self._add_conversation_id_candidate(self.conversation_id)

            if isinstance(body, dict):
                # Prefer explicit conversation ID markers in message body/metadata.
                self._capture_conversation_id_candidate(
                    body.get("conversationId"), is_explicit=True
                )
                self._capture_conversation_id_candidate(
                    body.get("conversation_id"), is_explicit=True
                )
                metadata = body.get("metadata")
                if isinstance(metadata, dict):
                    self._capture_conversation_id_candidate(
                        metadata.get("conversationId"), is_explicit=True
                    )
                    self._capture_conversation_id_candidate(
                        metadata.get("conversation_id"), is_explicit=True
                    )
                    self._capture_conversation_id_candidate(
                        metadata.get("conversationid"), is_explicit=True
                    )

        def _set_if_missing(attr_name: str, value: object) -> None:
            if getattr(self, attr_name) is not None:
                return
            if isinstance(value, str) and value.strip():
                normalized = value.strip()
                if attr_name == "conversation_id" and not self._is_likely_conversation_id(normalized):
                    return
                setattr(self, attr_name, normalized)

        def _walk(node: object, parent_key: Optional[str] = None) -> None:
            if isinstance(node, dict):
                _set_if_missing("conversation_id", node.get("conversationId"))
                _set_if_missing("conversation_id", node.get("conversation_id"))
                _set_if_missing("participant_id", node.get("participantId"))
                _set_if_missing("participant_id", node.get("participant_id"))
                self._capture_conversation_id_candidate(
                    node.get("conversationId"), is_explicit=True
                )
                self._capture_conversation_id_candidate(
                    node.get("conversation_id"), is_explicit=True
                )

                conversation_obj = node.get("conversation")
                if isinstance(conversation_obj, dict):
                    _set_if_missing("conversation_id", conversation_obj.get("id"))
                    self._capture_conversation_id_candidate(
                        conversation_obj.get("id"), is_explicit=True
                    )

                participant_obj = node.get("participant")
                if isinstance(participant_obj, dict):
                    _set_if_missing("participant_id", participant_obj.get("id"))

                if parent_key == "conversation":
                    _set_if_missing("conversation_id", node.get("id"))
                if parent_key == "participant":
                    _set_if_missing("participant_id", node.get("id"))

                for key, value in node.items():
                    _walk(value, parent_key=key)
            elif isinstance(node, list):
                for item in node:
                    _walk(item, parent_key=parent_key)

        _walk(payload)
        if self.conversation_id:
            self._add_conversation_id_candidate(self.conversation_id)

    def _capture_conversation_id_candidate(
        self, value: object, is_explicit: bool = False
    ) -> None:
        if not isinstance(value, str):
            return
        normalized = value.strip()
        if not normalized:
            return
        if not is_explicit:
            return
        self._add_conversation_id_candidate(normalized)
        if self.conversation_id is None and self._is_likely_conversation_id(normalized):
            self.conversation_id = normalized

    def _add_conversation_id_candidate(self, value: str) -> None:
        if value not in self._conversation_id_candidates:
            self._conversation_id_candidates.append(value)

    def _is_likely_conversation_id(self, value: str) -> bool:
        return bool(self._UUID_PATTERN.match(value))

    def _capture_debug_frame(self, payload: object, stage: str) -> None:
        """Capture compact debug metadata from parsed WebSocket frames."""
        if not self._debug_capture_frames:
            return
        if len(self._debug_frames) >= self._debug_capture_frame_limit:
            return
        if not isinstance(payload, dict):
            return

        body = payload.get("body")
        summary: dict[str, Any] = {
            "stage": stage,
            "type": payload.get("type"),
            "class": payload.get("class"),
            "top_level_keys": sorted(payload.keys()),
            "body_type": body.get("type") if isinstance(body, dict) else None,
            "body_direction": body.get("direction") if isinstance(body, dict) else None,
            "body_keys": sorted(body.keys()) if isinstance(body, dict) else None,
            "body_id": body.get("id") if isinstance(body, dict) else None,
            "metadata_keys": sorted(body.get("metadata", {}).keys())
            if isinstance(body, dict) and isinstance(body.get("metadata"), dict)
            else None,
            "conversation_id": self.conversation_id,
            "participant_id": self.participant_id,
            "conversation_id_candidates": list(self._conversation_id_candidates),
        }
        self._debug_frames.append(summary)

    def get_debug_frames(self) -> list[dict[str, Any]]:
        """Return captured debug frame summaries."""
        return [dict(frame) for frame in self._debug_frames]

    def get_conversation_id_candidates(self) -> list[str]:
        """Return best-effort list of conversation id candidates seen in frames."""
        return list(self._conversation_id_candidates)
