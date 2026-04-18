"""Unit tests for app_config module."""

import os
import json
import tempfile

import pytest
import yaml

from src.app_config import (
    load_app_config,
    merge_config,
    validate_required_config,
)
from src.models import AppConfig


# --- load_app_config tests ---


class TestLoadAppConfig:
    def test_defaults_when_no_config(self, monkeypatch, tmp_path):
        """With no config file and no env vars, defaults are used."""
        monkeypatch.chdir(tmp_path)
        # Clear all relevant env vars
        for var in [
            "GC_REGION", "GC_DEPLOYMENT_ID", "GC_CLIENT_ID", "GC_CLIENT_SECRET", "GC_TESTER_INTENT_ATTRIBUTE_NAME", "GC_TESTER_DEBUG_CAPTURE_FRAMES", "GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT", "OLLAMA_BASE_URL",
            "OLLAMA_MODEL", "GC_TESTER_DEFAULT_ATTEMPTS",
            "GC_TESTER_MAX_TURNS", "GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS",
            "GC_TESTER_RESPONSE_TIMEOUT", "GC_TESTER_SUCCESS_THRESHOLD",
            "GC_TESTER_EXPECTED_GREETING", "GC_TESTER_CONFIG_FILE",
        ]:
            monkeypatch.delenv(var, raising=False)

        config = load_app_config()
        assert config.gc_region is None
        assert config.gc_deployment_id is None
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model is None
        assert config.default_attempts == 5
        assert config.max_turns == 10
        assert config.min_attempt_interval_seconds == 30
        assert config.response_timeout == 30
        assert config.success_threshold == 0.8
        assert config.debug_capture_frames is False

    def test_loads_from_config_file(self, monkeypatch, tmp_path):
        """Values are loaded from config.yaml."""
        config_data = {
            "gc_region": "us-east-1",
            "gc_deployment_id": "deploy-123",
            "ollama_model": "llama3",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        monkeypatch.chdir(tmp_path)
        for var in [
            "GC_REGION", "GC_DEPLOYMENT_ID", "GC_CLIENT_ID", "GC_CLIENT_SECRET", "GC_TESTER_INTENT_ATTRIBUTE_NAME", "GC_TESTER_DEBUG_CAPTURE_FRAMES", "GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT", "OLLAMA_BASE_URL",
            "OLLAMA_MODEL", "GC_TESTER_DEFAULT_ATTEMPTS",
            "GC_TESTER_MAX_TURNS", "GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS",
            "GC_TESTER_RESPONSE_TIMEOUT", "GC_TESTER_SUCCESS_THRESHOLD",
            "GC_TESTER_EXPECTED_GREETING", "GC_TESTER_CONFIG_FILE",
        ]:
            monkeypatch.delenv(var, raising=False)

        config = load_app_config()
        assert config.gc_region == "us-east-1"
        assert config.gc_deployment_id == "deploy-123"
        assert config.ollama_model == "llama3"

    def test_env_vars_override_config_file(self, monkeypatch, tmp_path):
        """Environment variables take precedence over config file."""
        config_data = {
            "gc_region": "file-region",
            "gc_deployment_id": "file-deploy",
            "ollama_model": "file-model",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GC_REGION", "env-region")
        monkeypatch.setenv("GC_DEPLOYMENT_ID", "env-deploy")
        # Don't set OLLAMA_MODEL - should come from file
        monkeypatch.delenv("GC_CLIENT_ID", raising=False)
        monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("GC_TESTER_INTENT_ATTRIBUTE_NAME", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAMES", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("GC_TESTER_DEFAULT_ATTEMPTS", raising=False)
        monkeypatch.delenv("GC_TESTER_MAX_TURNS", raising=False)
        monkeypatch.delenv("GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS", raising=False)
        monkeypatch.delenv("GC_TESTER_RESPONSE_TIMEOUT", raising=False)
        monkeypatch.delenv("GC_TESTER_SUCCESS_THRESHOLD", raising=False)
        monkeypatch.delenv("GC_TESTER_EXPECTED_GREETING", raising=False)
        monkeypatch.delenv("GC_TESTER_CONFIG_FILE", raising=False)

        config = load_app_config()
        assert config.gc_region == "env-region"
        assert config.gc_deployment_id == "env-deploy"
        assert config.ollama_model == "file-model"

    def test_custom_config_file_path(self, monkeypatch, tmp_path):
        """GC_TESTER_CONFIG_FILE env var specifies custom config path."""
        config_data = {"gc_region": "custom-region"}
        custom_file = tmp_path / "custom.yaml"
        custom_file.write_text(yaml.dump(config_data))

        monkeypatch.setenv("GC_TESTER_CONFIG_FILE", str(custom_file))
        monkeypatch.delenv("GC_REGION", raising=False)
        monkeypatch.delenv("GC_DEPLOYMENT_ID", raising=False)
        monkeypatch.delenv("GC_CLIENT_ID", raising=False)
        monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("GC_TESTER_INTENT_ATTRIBUTE_NAME", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAMES", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_TESTER_DEFAULT_ATTEMPTS", raising=False)
        monkeypatch.delenv("GC_TESTER_MAX_TURNS", raising=False)
        monkeypatch.delenv("GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS", raising=False)
        monkeypatch.delenv("GC_TESTER_RESPONSE_TIMEOUT", raising=False)
        monkeypatch.delenv("GC_TESTER_SUCCESS_THRESHOLD", raising=False)
        monkeypatch.delenv("GC_TESTER_EXPECTED_GREETING", raising=False)

        config = load_app_config()
        assert config.gc_region == "custom-region"

    def test_numeric_env_vars_converted(self, monkeypatch, tmp_path):
        """Numeric env vars are properly converted to int/float."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GC_TESTER_DEFAULT_ATTEMPTS", "10")
        monkeypatch.setenv("GC_TESTER_MAX_TURNS", "30")
        monkeypatch.setenv("GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS", "90")
        monkeypatch.setenv("GC_TESTER_RESPONSE_TIMEOUT", "60")
        monkeypatch.setenv("GC_TESTER_SUCCESS_THRESHOLD", "0.9")
        monkeypatch.delenv("GC_REGION", raising=False)
        monkeypatch.delenv("GC_DEPLOYMENT_ID", raising=False)
        monkeypatch.delenv("GC_CLIENT_ID", raising=False)
        monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("GC_TESTER_INTENT_ATTRIBUTE_NAME", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAMES", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_TESTER_CONFIG_FILE", raising=False)

        config = load_app_config()
        assert config.default_attempts == 10
        assert config.max_turns == 30
        assert config.min_attempt_interval_seconds == 90
        assert config.response_timeout == 60
        assert config.success_threshold == 0.9

    def test_loads_intent_fallback_env_vars(self, monkeypatch, tmp_path):
        """Intent fallback env vars are loaded into AppConfig."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GC_CLIENT_ID", "client-id-123")
        monkeypatch.setenv("GC_CLIENT_SECRET", "client-secret-456")
        monkeypatch.setenv("GC_TESTER_INTENT_ATTRIBUTE_NAME", "intent_name")
        monkeypatch.delenv("GC_REGION", raising=False)
        monkeypatch.delenv("GC_DEPLOYMENT_ID", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_TESTER_CONFIG_FILE", raising=False)

        config = load_app_config()
        assert config.gc_client_id == "client-id-123"
        assert config.gc_client_secret == "client-secret-456"
        assert config.intent_attribute_name == "intent_name"

    def test_loads_debug_capture_frame_env_vars(self, monkeypatch, tmp_path):
        """Debug frame env vars should parse into config."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GC_TESTER_DEBUG_CAPTURE_FRAMES", "true")
        monkeypatch.setenv("GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT", "12")
        monkeypatch.delenv("GC_REGION", raising=False)
        monkeypatch.delenv("GC_DEPLOYMENT_ID", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_TESTER_CONFIG_FILE", raising=False)

        config = load_app_config()
        assert config.debug_capture_frames is True
        assert config.debug_capture_frame_limit == 12

    def test_nonexistent_config_file_uses_defaults(self, monkeypatch, tmp_path):
        """If config file doesn't exist, defaults are used without error."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GC_TESTER_CONFIG_FILE", "/nonexistent/path.yaml")
        monkeypatch.delenv("GC_REGION", raising=False)
        monkeypatch.delenv("GC_DEPLOYMENT_ID", raising=False)
        monkeypatch.delenv("GC_CLIENT_ID", raising=False)
        monkeypatch.delenv("GC_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("GC_TESTER_INTENT_ATTRIBUTE_NAME", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAMES", raising=False)
        monkeypatch.delenv("GC_TESTER_DEBUG_CAPTURE_FRAME_LIMIT", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("GC_TESTER_DEFAULT_ATTEMPTS", raising=False)
        monkeypatch.delenv("GC_TESTER_MAX_TURNS", raising=False)
        monkeypatch.delenv("GC_TESTER_MIN_ATTEMPT_INTERVAL_SECONDS", raising=False)
        monkeypatch.delenv("GC_TESTER_RESPONSE_TIMEOUT", raising=False)
        monkeypatch.delenv("GC_TESTER_SUCCESS_THRESHOLD", raising=False)
        monkeypatch.delenv("GC_TESTER_EXPECTED_GREETING", raising=False)

        config = load_app_config()
        assert config.gc_region is None
        assert config.default_attempts == 5


# --- merge_config tests ---


class TestMergeConfig:
    def test_web_overrides_applied(self):
        """Web UI overrides take highest precedence."""
        base = AppConfig(gc_region="base-region", gc_deployment_id="base-deploy")
        overrides = {"gc_region": "web-region"}

        result = merge_config(base, overrides)
        assert result.gc_region == "web-region"
        assert result.gc_deployment_id == "base-deploy"

    def test_none_overrides_ignored(self):
        """None values in overrides don't replace base values."""
        base = AppConfig(gc_region="base-region")
        overrides = {"gc_region": None}

        result = merge_config(base, overrides)
        assert result.gc_region == "base-region"

    def test_empty_string_overrides_ignored(self):
        """Empty string overrides don't replace base values."""
        base = AppConfig(gc_region="base-region")
        overrides = {"gc_region": ""}

        result = merge_config(base, overrides)
        assert result.gc_region == "base-region"

    def test_numeric_overrides_converted(self):
        """Numeric fields from web overrides are converted properly."""
        base = AppConfig(default_attempts=5, success_threshold=0.8)
        overrides = {"default_attempts": "10", "success_threshold": "0.95"}

        result = merge_config(base, overrides)
        assert result.default_attempts == 10
        assert result.success_threshold == 0.95

    def test_multiple_overrides(self):
        """Multiple fields can be overridden at once."""
        base = AppConfig(
            gc_region="base-region",
            gc_deployment_id="base-deploy",
            ollama_model="base-model",
        )
        overrides = {
            "gc_region": "web-region",
            "gc_deployment_id": "web-deploy",
            "ollama_model": "web-model",
        }

        result = merge_config(base, overrides)
        assert result.gc_region == "web-region"
        assert result.gc_deployment_id == "web-deploy"
        assert result.ollama_model == "web-model"

    def test_returns_new_instance(self):
        """merge_config returns a new AppConfig, not mutating the base."""
        base = AppConfig(gc_region="original")
        overrides = {"gc_region": "overridden"}

        result = merge_config(base, overrides)
        assert base.gc_region == "original"
        assert result.gc_region == "overridden"


# --- validate_required_config tests ---


class TestValidateRequiredConfig:
    def test_all_present(self):
        """No missing fields when all required fields are set."""
        config = AppConfig(
            gc_region="us-east-1",
            gc_deployment_id="deploy-123",
            ollama_model="llama3",
        )
        assert validate_required_config(config) == []

    def test_all_missing(self):
        """All required fields reported when none are set."""
        config = AppConfig()
        missing = validate_required_config(config)
        assert "gc_region" in missing
        assert "gc_deployment_id" in missing
        assert "ollama_model" in missing
        assert len(missing) == 3

    def test_partial_missing(self):
        """Only missing fields are reported."""
        config = AppConfig(gc_region="us-east-1")
        missing = validate_required_config(config)
        assert "gc_region" not in missing
        assert "gc_deployment_id" in missing
        assert "ollama_model" in missing
        assert len(missing) == 2

    def test_single_missing(self):
        """Single missing field is reported."""
        config = AppConfig(
            gc_region="us-east-1",
            gc_deployment_id="deploy-123",
        )
        missing = validate_required_config(config)
        assert missing == ["ollama_model"]
