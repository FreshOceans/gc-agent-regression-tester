"""Unit tests for CLI argument parsing and config overrides."""

from src.cli import _merge_cli_overrides, _parse_args
from src.models import AppConfig


def test_parse_args_supports_analytics_journey_mode():
    args = _parse_args(
        [
            "analytics-journey",
            "--interval",
            "2026-04-20T00:00:00.000Z/2026-04-21T00:00:00.000Z",
            "--analytics-auth-mode",
            "manual_bearer",
            "--analytics-bearer-token",
            "token-123",
        ]
    )

    assert args.command == "analytics-journey"
    assert args.analytics_auth_mode == "manual_bearer"
    assert args.analytics_bearer_token == "token-123"
    assert args.interval.startswith("2026-04-20T00:00:00.000Z")


def test_merge_cli_overrides_applies_analytics_auth_mode_and_cap():
    base = AppConfig()
    args = _parse_args(
        [
            "analytics-journey",
            "--interval",
            "2026-04-20T00:00:00.000Z/2026-04-21T00:00:00.000Z",
            "--analytics-auth-mode",
            "manual_bearer",
            "--analytics-page-size-cap",
            "77",
            "--region",
            "usw2.pure.cloud",
        ]
    )
    merged = _merge_cli_overrides(base, args)

    assert merged.analytics_journey_auth_mode == "manual_bearer"
    assert merged.analytics_journey_details_page_size_cap == 77
    assert merged.gc_region == "usw2.pure.cloud"
