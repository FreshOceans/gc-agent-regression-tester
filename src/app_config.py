"""Application configuration loading, merging, and validation."""

import os
from pathlib import Path
from typing import Any

import yaml

from .models import AppConfig


# Environment variable to config field mapping
_ENV_VAR_MAP: dict[str, str] = {
    "GC_REGION": "gc_region",
    "GC_DEPLOYMENT_ID": "gc_deployment_id",
    "GC_CLIENT_ID": "gc_client_id",
    "GC_CLIENT_SECRET": "gc_client_secret",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_MODEL": "ollama_model",
    "GC_TESTER_INTENT_ATTRIBUTE_NAME": "intent_attribute_name",
    "GC_TESTER_JUDGE_WARMUP_ENABLED": "judge_warmup_enabled",
    "GC_TESTER_DEFAULT_ATTEMPTS": "default_attempts",
    "GC_TESTER_MAX_TURNS": "max_turns",
    "GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS": "min_attempt_interval_seconds",
    "GC_TESTER_RESPONSE_TIMEOUT": "response_timeout",
    "GC_TESTER_SUCCESS_THRESHOLD": "success_threshold",
    "GC_TESTER_EXPECTED_GREETING": "expected_greeting",
    "GC_TESTER_DEBUG_CAPTURE_FRAMES": "debug_capture_frames",
    "GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT": "debug_capture_frame_limit",
}

# Fields that require type conversion from string env vars
_INT_FIELDS = {
    "default_attempts",
    "max_turns",
    "min_attempt_interval_seconds",
    "response_timeout",
    "debug_capture_frame_limit",
}
_FLOAT_FIELDS = {"success_threshold"}
_BOOL_FIELDS = {"debug_capture_frames", "judge_warmup_enabled"}


def _to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value '{value}' for configuration field. "
        "Use true/false, yes/no, on/off, or 1/0."
    )

# Required fields that must be present for a test run
_REQUIRED_FIELDS = ("gc_region", "gc_deployment_id", "ollama_model")


def _load_config_file() -> dict[str, Any]:
    """Load configuration from a YAML config file.

    The config file path is determined by the GC_TESTER_CONFIG_FILE env var,
    or defaults to 'config.yaml' in the current directory.

    Returns:
        A dictionary of config values from the file, or empty dict if
        the file doesn't exist.
    """
    config_path = os.environ.get("GC_TESTER_CONFIG_FILE", "config.yaml")
    path = Path(config_path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return {}

    return data


def _load_env_vars() -> dict[str, Any]:
    """Load configuration values from environment variables.

    Returns:
        A dictionary of config field names to their values from env vars.
        Only includes env vars that are actually set.
    """
    result: dict[str, Any] = {}

    for env_var, field_name in _ENV_VAR_MAP.items():
        value = os.environ.get(env_var)
        if value is None:
            continue

        # Convert to appropriate type
        if field_name in _INT_FIELDS:
            result[field_name] = int(value)
        elif field_name in _FLOAT_FIELDS:
            result[field_name] = float(value)
        elif field_name in _BOOL_FIELDS:
            result[field_name] = _to_bool(value)
        else:
            result[field_name] = value

    return result


def load_app_config() -> AppConfig:
    """Load configuration from env vars and config file.

    Precedence (highest to lowest):
    1. Environment variables
    2. Config file
    3. Model defaults

    Returns:
        A fully resolved AppConfig instance.
    """
    # Start with config file values
    file_config = _load_config_file()

    # Overlay env vars (higher precedence)
    env_config = _load_env_vars()

    # Merge: env vars override file config
    merged = {**file_config, **env_config}

    return AppConfig(**merged)


def merge_config(base: AppConfig, web_overrides: dict) -> AppConfig:
    """Merge web UI overrides on top of base config.

    Web UI values take highest precedence. Only non-None values
    from web_overrides are applied.

    Args:
        base: The base AppConfig (from env vars / config file).
        web_overrides: Dictionary of override values from the Web UI.

    Returns:
        A new AppConfig with web overrides applied.
    """
    base_dict = base.model_dump()

    # Only apply overrides that are not None and not empty strings
    for key, value in web_overrides.items():
        if value is not None and value != "":
            # Convert types for numeric fields
            if key in _INT_FIELDS:
                base_dict[key] = int(value)
            elif key in _FLOAT_FIELDS:
                base_dict[key] = float(value)
            elif key in _BOOL_FIELDS and isinstance(value, str):
                base_dict[key] = _to_bool(value)
            else:
                base_dict[key] = value

    return AppConfig(**base_dict)


def validate_required_config(config: AppConfig) -> list[str]:
    """Return list of missing required config fields.

    Required fields are: gc_region, gc_deployment_id, ollama_model.

    Args:
        config: The AppConfig to validate.

    Returns:
        A list of field names that are missing (None). Empty list if
        all required fields are present.
    """
    missing = []
    for field in _REQUIRED_FIELDS:
        value = getattr(config, field)
        if value is None:
            missing.append(field)
    return missing
