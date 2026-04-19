"""Genesys Cloud Conversations API helper for participant attribute lookup."""

import time
from typing import Any, Optional

import requests


class GenesysConversationsError(Exception):
    """Raised when the Conversations API fallback cannot be completed."""

    pass


class GenesysConversationsClient:
    """Small client for reading participant attributes from messaging conversations."""

    def __init__(
        self,
        region: str,
        client_id: str,
        client_secret: str,
        timeout: int = 30,
    ):
        self.region = region.strip()
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.timeout = timeout
        self._access_token: Optional[str] = None
        self._token_expiry_monotonic = 0.0

    @property
    def _api_base_url(self) -> str:
        return f"https://api.{self.region}"

    @property
    def _oauth_url(self) -> str:
        return f"https://login.{self.region}/oauth/token"

    def _get_access_token(self) -> str:
        now = time.monotonic()
        if self._access_token and now < self._token_expiry_monotonic:
            return self._access_token

        try:
            response = requests.post(
                self._oauth_url,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            raise GenesysConversationsError(
                f"OAuth token request failed for region '{self.region}': {e}"
            ) from e
        except ValueError as e:
            raise GenesysConversationsError(
                f"Invalid OAuth token response for region '{self.region}': {e}"
            ) from e

        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 300)
        if not isinstance(token, str) or not token:
            raise GenesysConversationsError("OAuth token response missing access_token")

        # Refresh a little early to avoid edge-expiry races.
        refresh_padding = 30
        safe_ttl = max(30, int(expires_in) - refresh_padding)
        self._access_token = token
        self._token_expiry_monotonic = time.monotonic() + safe_ttl
        return token

    def _fetch_conversation(self, conversation_id: str) -> dict[str, Any]:
        token = self._get_access_token()
        url = f"{self._api_base_url}/api/v2/conversations/messages/{conversation_id}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            raise GenesysConversationsError(
                f"Conversation lookup failed for '{conversation_id}': {e}"
            ) from e
        except ValueError as e:
            raise GenesysConversationsError(
                f"Invalid conversation payload for '{conversation_id}': {e}"
            ) from e
        if not isinstance(payload, dict):
            raise GenesysConversationsError(
                f"Unexpected conversation payload type for '{conversation_id}'"
            )
        return payload

    def get_conversation_payload(self, conversation_id: str) -> dict[str, Any]:
        """Return full conversation payload for metadata-driven diagnostics."""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            raise GenesysConversationsError("Conversation ID is required")
        return self._fetch_conversation(conversation_id)

    def get_participant_attribute(
        self,
        conversation_id: str,
        attribute_name: str,
        participant_id: Optional[str] = None,
        retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> Optional[str]:
        """Return participant attribute from messaging conversation, if available."""
        conversation_id = conversation_id.strip()
        attribute_name = attribute_name.strip()
        participant_id = participant_id.strip() if participant_id else None
        if not conversation_id:
            raise GenesysConversationsError("Conversation ID is required")
        if not attribute_name:
            raise GenesysConversationsError("Participant attribute name is required")

        for attempt in range(1, retries + 1):
            payload = self._fetch_conversation(conversation_id)
            participants = payload.get("participants", [])
            if isinstance(participants, list):
                for participant in participants:
                    if not isinstance(participant, dict):
                        continue
                    if participant_id:
                        participant_id_value = participant.get("id")
                        if participant_id_value != participant_id:
                            continue
                    attributes = participant.get("attributes", {})
                    if not isinstance(attributes, dict):
                        continue
                    value = attributes.get(attribute_name)
                    normalized_value = self._normalize_attribute_value(value)
                    if normalized_value:
                        return normalized_value

            if attempt < retries:
                time.sleep(retry_delay_seconds)

        return None

    def get_participant_attributes(
        self,
        conversation_id: str,
        participant_id: Optional[str] = None,
        retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> dict[str, Any]:
        """Return merged participant attributes from a conversation payload.

        If participant_id is provided, only matching participant attributes are merged.
        Otherwise, attributes from all participants are merged in encounter order.
        """
        conversation_id = conversation_id.strip()
        participant_id = participant_id.strip() if participant_id else None
        if not conversation_id:
            raise GenesysConversationsError("Conversation ID is required")

        merged: dict[str, Any] = {}
        for attempt in range(1, retries + 1):
            payload = self._fetch_conversation(conversation_id)
            participants = payload.get("participants", [])
            merged = {}
            if isinstance(participants, list):
                for participant in participants:
                    if not isinstance(participant, dict):
                        continue
                    if participant_id:
                        participant_id_value = participant.get("id")
                        if participant_id_value != participant_id:
                            continue
                    attributes = participant.get("attributes", {})
                    if not isinstance(attributes, dict):
                        continue
                    for key, value in attributes.items():
                        key_name = str(key).strip()
                        if not key_name:
                            continue
                        # Preserve first value for deterministic behavior.
                        if key_name not in merged:
                            merged[key_name] = value

            if merged:
                return merged
            if attempt < retries:
                time.sleep(retry_delay_seconds)

        return merged

    def _normalize_attribute_value(self, value: Any) -> Optional[str]:
        """Normalize attribute values to lower-case strings for intent comparison."""
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized or None
        if isinstance(value, dict):
            intent = value.get("intent")
            if isinstance(intent, str) and intent.strip():
                return intent.strip().lower()
        normalized = str(value).strip().lower()
        return normalized or None
