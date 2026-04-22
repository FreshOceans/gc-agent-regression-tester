"""Unit tests for CLI argument parsing and config overrides."""

from src.cli import _merge_cli_overrides, _parse_args
from src.models import AppConfig


def test_parse_args_supports_analytics_journey_mode():
    args = _parse_args(
        [
            "analytics-journey",
            "--bot-flow-id",
            "flow-123",
            "--interval",
            "2026-04-20T00:00:00.000Z/2026-04-21T00:00:00.000Z",
            "--analytics-auth-mode",
            "manual_bearer",
            "--analytics-bearer-token",
            "token-123",
        ]
    )

    assert args.command == "analytics-journey"
    assert args.bot_flow_id == "flow-123"
    assert args.analytics_auth_mode == "manual_bearer"
    assert args.analytics_bearer_token == "token-123"
    assert args.interval.startswith("2026-04-20T00:00:00.000Z")


def test_merge_cli_overrides_applies_analytics_auth_mode_and_cap():
    base = AppConfig()
    args = _parse_args(
        [
            "analytics-journey",
            "--bot-flow-id",
            "flow-xyz",
            "--interval",
            "2026-04-20T00:00:00.000Z/2026-04-21T00:00:00.000Z",
            "--analytics-auth-mode",
            "manual_bearer",
            "--analytics-page-size-cap",
            "77",
            "--ollama-model",
            "llama3",
            "--region",
            "usw2.pure.cloud",
        ]
    )
    merged = _merge_cli_overrides(base, args)

    assert merged.analytics_journey_auth_mode == "manual_bearer"
    assert merged.analytics_journey_details_page_size_cap == 77
    assert merged.analytics_journey_judge_model == "llama3"
    assert merged.gc_region == "usw2.pure.cloud"


def test_parse_args_supports_gemma_judge_flags():
    args = _parse_args(
        [
            "run",
            "suite.yaml",
            "--judge-mode",
            "dual_strict_fallback",
            "--judge-model",
            "gemma4:31b",
        ]
    )

    assert args.command == "run"
    assert args.test_suite == "suite.yaml"
    assert args.judge_mode == "dual_strict_fallback"
    assert args.judge_model == "gemma4:31b"


def test_merge_cli_overrides_applies_gemma_judge_overrides():
    base = AppConfig()
    args = _parse_args(
        [
            "run",
            "suite.yaml",
            "--judge-mode",
            "dual_strict_fallback",
            "--judge-model",
            "gemma4:31b",
            "--ollama-model",
            "custom-model",
        ]
    )

    merged = _merge_cli_overrides(base, args)

    assert merged.judge_execution_mode == "dual_strict_fallback"
    assert merged.judge_single_model == "gemma4:31b"
    assert merged.ollama_model == "custom-model"
