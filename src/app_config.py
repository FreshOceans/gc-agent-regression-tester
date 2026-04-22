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
    "GC_TESTER_STEP_SKIP_TIMEOUT_SECONDS": "step_skip_timeout_seconds",
    "GC_TESTER_KNOWLEDGE_MODE_TIMEOUT_SECONDS": "knowledge_mode_timeout_seconds",
    "GC_TESTER_DEFAULT_ATTEMPTS": "default_attempts",
    "GC_TESTER_MAX_TURNS": "max_turns",
    "GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS": "min_attempt_interval_seconds",
    "GC_TESTER_RESPONSE_TIMEOUT": "response_timeout",
    "GC_TESTER_SUCCESS_THRESHOLD": "success_threshold",
    "GC_TESTER_EXPECTED_GREETING": "expected_greeting",
    "GC_TESTER_DEBUG_CAPTURE_FRAMES": "debug_capture_frames",
    "GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT": "debug_capture_frame_limit",
    "GC_TESTER_HISTORY_DIR": "history_dir",
    "GC_TESTER_HISTORY_MAX_RUNS": "history_max_runs",
    "GC_TESTER_HISTORY_FULL_JSON_RUNS": "history_full_json_runs",
    "GC_TESTER_HISTORY_GZIP_RUNS": "history_gzip_runs",
    "GC_TESTER_TRANSCRIPT_IMPORT_ENABLED": "transcript_import_enabled",
    "GC_TESTER_TRANSCRIPT_IMPORT_TIME": "transcript_import_time",
    "GC_TESTER_TRANSCRIPT_IMPORT_TIMEZONE": "transcript_import_timezone",
    "GC_TESTER_TRANSCRIPT_IMPORT_MAX_IDS": "transcript_import_max_ids",
    "GC_TESTER_TRANSCRIPT_IMPORT_FILTER_JSON": "transcript_import_filter_json",
    "GC_TESTER_TRANSCRIPT_IMPORT_DIR": "transcript_import_dir",
    "GC_TESTER_TRANSCRIPT_URL_ALLOWLIST": "transcript_url_allowlist",
    "GC_TESTER_TRANSCRIPT_URL_TIMEOUT_SECONDS": "transcript_url_timeout_seconds",
    "GC_TESTER_TRANSCRIPT_URL_MAX_BYTES": "transcript_url_max_bytes",
    "GC_TESTER_TOOL_ATTRIBUTE_KEYS": "tool_attribute_keys",
    "GC_TESTER_TOOL_MARKER_PREFIXES": "tool_marker_prefixes",
    "GC_TESTER_HARNESS_MODE": "harness_mode",
    "GC_TESTER_JOURNEY_CATEGORY_STRATEGY": "journey_category_strategy",
    "GC_TESTER_JOURNEY_PRIMARY_CATEGORIES_JSON": "journey_primary_categories_json",
    "GC_TESTER_JOURNEY_PRIMARY_CATEGORIES_FILE": "journey_primary_categories_file",
    "GC_TESTER_JUDGING_MECHANICS_ENABLED": "judging_mechanics_enabled",
    "GC_TESTER_JUDGING_OBJECTIVE_PROFILE": "judging_objective_profile",
    "GC_TESTER_JUDGING_STRICTNESS": "judging_strictness",
    "GC_TESTER_JUDGING_TOLERANCE": "judging_tolerance",
    "GC_TESTER_JUDGING_CONTAINMENT_WEIGHT": "judging_containment_weight",
    "GC_TESTER_JUDGING_FULFILLMENT_WEIGHT": "judging_fulfillment_weight",
    "GC_TESTER_JUDGING_PATH_WEIGHT": "judging_path_weight",
    "GC_TESTER_JUDGING_EXPLANATION_MODE": "judging_explanation_mode",
    "GC_TESTER_JOURNEY_DASHBOARD_ENABLED": "journey_dashboard_enabled",
    "GC_TESTER_JOURNEY_TAXONOMY_OVERRIDES_JSON": "journey_taxonomy_overrides_json",
    "GC_TESTER_JOURNEY_TAXONOMY_OVERRIDES_FILE": "journey_taxonomy_overrides_file",
    "GC_TESTER_ANALYTICS_JOURNEY_ENABLED": "analytics_journey_enabled",
    "GC_TESTER_ANALYTICS_JOURNEY_AUTH_MODE": "analytics_journey_auth_mode",
    "GC_TESTER_ANALYTICS_JOURNEY_DETAILS_PAGE_SIZE_CAP": "analytics_journey_details_page_size_cap",
    "GC_TESTER_ANALYTICS_JOURNEY_DEFAULT_PAGE_SIZE": "analytics_journey_default_page_size",
    "GC_TESTER_ANALYTICS_JOURNEY_DEFAULT_MAX_CONVERSATIONS": "analytics_journey_default_max_conversations",
    "GC_TESTER_ANALYTICS_JOURNEY_POLICY_MAP_JSON": "analytics_journey_policy_map_json",
    "GC_TESTER_ANALYTICS_JOURNEY_POLICY_MAP_FILE": "analytics_journey_policy_map_file",
    "GC_TESTER_ANALYTICS_JOURNEY_JUDGE_MODEL": "analytics_journey_judge_model",
    "GC_TESTER_ANALYTICS_JUDGE_EXECUTION_MODE": "analytics_judge_execution_mode",
    "GC_TESTER_ANALYTICS_JUDGE_SINGLE_MODEL": "analytics_judge_single_model",
    "GC_TESTER_ANALYTICS_JOURNEY_DEFAULT_LANGUAGE_FILTER": "analytics_journey_default_language_filter",
    "GC_TESTER_ANALYTICS_JOURNEY_ARTIFACT_DIR": "analytics_journey_artifact_dir",
    "GC_TESTER_ATTEMPT_PARALLEL_ENABLED": "attempt_parallel_enabled",
    "GC_TESTER_MAX_PARALLEL_ATTEMPT_WORKERS": "max_parallel_attempt_workers",
    "GC_TESTER_ADAPTIVE_ATTEMPT_PACING_ENABLED": "adaptive_attempt_pacing_enabled",
    "GC_TESTER_WEB_AUTH_ENABLED": "web_auth_enabled",
    "GC_TESTER_WEB_AUTH_USERNAME": "web_auth_username",
    "GC_TESTER_WEB_AUTH_PASSWORD": "web_auth_password",
    "GC_TESTER_WEB_SESSION_IDLE_MINUTES": "web_session_idle_minutes",
    "GC_TESTER_LANGUAGE": "language",
    "GC_TESTER_EVALUATION_RESULTS_LANGUAGE": "evaluation_results_language",
    "GC_TESTER_JUDGE_EXECUTION_MODE": "judge_execution_mode",
    "GC_TESTER_JUDGE_SINGLE_MODEL": "judge_single_model",
}

# Fields that require type conversion from string env vars
_INT_FIELDS = {
    "default_attempts",
    "max_turns",
    "response_timeout",
    "debug_capture_frame_limit",
    "step_skip_timeout_seconds",
    "knowledge_mode_timeout_seconds",
    "history_max_runs",
    "history_full_json_runs",
    "history_gzip_runs",
    "transcript_import_max_ids",
    "transcript_url_timeout_seconds",
    "transcript_url_max_bytes",
    "analytics_journey_default_page_size",
    "analytics_journey_details_page_size_cap",
    "analytics_journey_default_max_conversations",
    "max_parallel_attempt_workers",
    "web_session_idle_minutes",
}
_FLOAT_FIELDS = {"success_threshold", "min_attempt_interval_seconds"}
_BOOL_FIELDS = {
    "debug_capture_frames",
    "judge_warmup_enabled",
    "transcript_import_enabled",
    "judging_mechanics_enabled",
    "journey_dashboard_enabled",
    "analytics_journey_enabled",
    "attempt_parallel_enabled",
    "adaptive_attempt_pacing_enabled",
    "web_auth_enabled",
}
_FLOAT_FIELDS.update(
    {
        "judging_tolerance",
        "judging_containment_weight",
        "judging_fulfillment_weight",
        "judging_path_weight",
    }
)


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
_REQUIRED_FIELDS = ("gc_region", "gc_deployment_id")


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

    Required fields are: gc_region, gc_deployment_id, plus an effective judge model.

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
    judge_model = str(config.ollama_model or "").strip() or str(
        config.judge_single_model or ""
    ).strip()
    if not judge_model:
        missing.append("judge_model")
    return missing
