"""CLI entry point for the Regression Test Harness.

Parses command-line arguments, loads configuration, runs the test suite,
prints progress and results to the console, and exits with a non-zero
code if regressions are detected.
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .analytics_journey_runner import AnalyticsJourneyRunRequest, AnalyticsJourneyRunner
from .app_config import load_app_config, validate_required_config
from .config_loader import load_test_suite
from .judge_execution import build_judge_execution_client
from .judge_llm import JudgeLLMError
from .models import (
    ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS,
    ANALYTICS_AUTH_MODE_MANUAL_BEARER,
    AppConfig,
    normalize_gemma_single_model,
    ProgressEvent,
    ProgressEventType,
    TestReport,
    normalize_analytics_auth_mode,
    normalize_judge_execution_mode,
)
from .orchestrator import TestOrchestrator
from .progress import ProgressEmitter


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "test_suite",
        help="Path to test suite file (JSON or YAML)",
    )
    parser.add_argument(
        "--region",
        help="Genesys Cloud region override",
    )
    parser.add_argument(
        "--deployment-id",
        help="Genesys Cloud deployment ID override",
    )
    parser.add_argument(
        "--ollama-url",
        help="Ollama base URL override",
    )
    parser.add_argument(
        "--ollama-model",
        help="Legacy custom Ollama model override for single judge mode",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["single", "dual_strict_fallback"],
        help="Judge execution mode override",
    )
    parser.add_argument(
        "--judge-model",
        choices=["gemma4:e4b", "gemma4:31b"],
        help="Gemma single-judge model override",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        help="Default number of attempts per scenario override",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum conversation turns override",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Response timeout in seconds override",
    )
    parser.add_argument(
        "--knowledge-timeout",
        type=int,
        help="Knowledge-mode timeout in seconds override",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Success threshold (0.0-1.0) override",
    )
    parser.add_argument(
        "--language",
        help="Run language override (en, fr, fr-CA, es)",
    )
    parser.add_argument(
        "--attempt-parallel-enabled",
        choices=["true", "false"],
        help="Override parallel execution enabled state",
    )
    parser.add_argument(
        "--max-parallel-attempt-workers",
        type=int,
        help="Override max parallel attempt workers (1-3)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Force serial execution (parallel disabled, workers=1)",
    )


def _parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Regression Test Harness — LLM-as-judge testing for Genesys Cloud agents"
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a suite")
    _add_common_run_arguments(run_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark serial vs parallel run performance",
    )
    _add_common_run_arguments(benchmark_parser)
    benchmark_parser.add_argument(
        "--candidate-workers",
        type=int,
        default=2,
        help="Parallel worker count for benchmark candidate run (default: 2)",
    )

    analytics_parser = subparsers.add_parser(
        "analytics-journey",
        help="Run analytics journey regression via bot reporting-turns API",
    )
    analytics_parser.add_argument(
        "--region",
        help="Genesys Cloud region override",
    )
    analytics_parser.add_argument(
        "--client-id",
        help="Genesys OAuth client ID override",
    )
    analytics_parser.add_argument(
        "--client-secret",
        help="Genesys OAuth client secret override",
    )
    analytics_parser.add_argument(
        "--ollama-url",
        help="Ollama base URL override",
    )
    analytics_parser.add_argument(
        "--ollama-model",
        help="Legacy custom Ollama model override for analytics single judge mode",
    )
    analytics_parser.add_argument(
        "--analytics-judge-mode",
        choices=["single", "dual_strict_fallback"],
        help="Analytics judge execution mode override",
    )
    analytics_parser.add_argument(
        "--analytics-judge-model",
        choices=["gemma4:e4b", "gemma4:31b"],
        help="Analytics Gemma single-judge model override",
    )
    analytics_parser.add_argument(
        "--timeout",
        type=int,
        help="Response timeout in seconds override",
    )
    analytics_parser.add_argument(
        "--knowledge-timeout",
        type=int,
        help="Knowledge-mode timeout in seconds override",
    )
    analytics_parser.add_argument(
        "--language",
        help="Run language override (en, fr, fr-CA, es)",
    )
    analytics_parser.add_argument(
        "--evaluation-results-language",
        help="Evaluation/results language override (inherit, en, fr, fr-CA, es)",
    )
    analytics_parser.add_argument(
        "--analytics-auth-mode",
        choices=[ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS, ANALYTICS_AUTH_MODE_MANUAL_BEARER],
        help="Analytics auth mode override",
    )
    analytics_parser.add_argument(
        "--analytics-bearer-token",
        help="Manual bearer token used when --analytics-auth-mode=manual_bearer",
    )
    analytics_parser.add_argument(
        "--analytics-page-size-cap",
        type=int,
        help="Cap for AJR reporting-turns page size",
    )
    analytics_parser.add_argument(
        "--bot-flow-id",
        required=True,
        help="Bot flow ID used by /api/v2/analytics/botflows/{botFlowId}/divisions/reportingturns",
    )
    analytics_parser.add_argument(
        "--interval",
        required=True,
        help="Analytics interval (startISO/endISO)",
    )
    analytics_parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Requested page size before cap (default: 50)",
    )
    analytics_parser.add_argument(
        "--max-conversations",
        type=int,
        default=150,
        help="Max conversations to evaluate (default: 150)",
    )
    analytics_parser.add_argument(
        "--divisions",
        default="",
        help="Optional comma-separated division IDs",
    )
    analytics_parser.add_argument(
        "--language-filter",
        default="",
        help="Optional conversation language filter",
    )
    analytics_parser.add_argument(
        "--filter-json",
        default="",
        help="Optional raw JSON object for supported reporting-turns query keys (unsupported keys are ignored)",
    )

    args_list = list(argv if argv is not None else sys.argv[1:])
    if not args_list:
        parser.print_help()
        parser.exit(2)

    if args_list[0] not in {"run", "benchmark", "analytics-journey"}:
        args_list = ["run", *args_list]

    return parser.parse_args(args_list)


def _merge_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Merge CLI argument overrides into the base config.

    CLI values take highest precedence over env vars and config file.

    Args:
        config: Base AppConfig loaded from env/file.
        args: Parsed CLI arguments.

    Returns:
        New AppConfig with CLI overrides applied.
    """
    data = config.model_dump()

    if getattr(args, "region", None) is not None:
        data["gc_region"] = args.region
    if getattr(args, "client_id", None) is not None:
        data["gc_client_id"] = args.client_id
    if getattr(args, "client_secret", None) is not None:
        data["gc_client_secret"] = args.client_secret
    if getattr(args, "deployment_id", None) is not None:
        data["gc_deployment_id"] = args.deployment_id
    if getattr(args, "ollama_url", None) is not None:
        data["ollama_base_url"] = args.ollama_url
    if getattr(args, "judge_mode", None) is not None:
        data["judge_execution_mode"] = normalize_judge_execution_mode(args.judge_mode)
    if getattr(args, "judge_model", None) is not None:
        data["judge_single_model"] = normalize_gemma_single_model(args.judge_model)
    if getattr(args, "ollama_model", None) is not None:
        data["ollama_model"] = args.ollama_model
        if getattr(args, "command", "") == "analytics-journey":
            data["analytics_journey_judge_model"] = args.ollama_model
    if getattr(args, "analytics_judge_mode", None) is not None:
        data["analytics_judge_execution_mode"] = normalize_judge_execution_mode(
            args.analytics_judge_mode
        )
    if getattr(args, "analytics_judge_model", None) is not None:
        data["analytics_judge_single_model"] = normalize_gemma_single_model(
            args.analytics_judge_model
        )
    if getattr(args, "attempts", None) is not None:
        data["default_attempts"] = args.attempts
    if getattr(args, "max_turns", None) is not None:
        data["max_turns"] = args.max_turns
    if getattr(args, "timeout", None) is not None:
        data["response_timeout"] = args.timeout
    if getattr(args, "knowledge_timeout", None) is not None:
        data["knowledge_mode_timeout_seconds"] = args.knowledge_timeout
    if getattr(args, "threshold", None) is not None:
        data["success_threshold"] = args.threshold
    if getattr(args, "language", None) is not None:
        data["language"] = args.language
    if getattr(args, "evaluation_results_language", None) is not None:
        data["evaluation_results_language"] = args.evaluation_results_language
    if getattr(args, "attempt_parallel_enabled", None) is not None:
        data["attempt_parallel_enabled"] = args.attempt_parallel_enabled == "true"
    if getattr(args, "max_parallel_attempt_workers", None) is not None:
        data["max_parallel_attempt_workers"] = args.max_parallel_attempt_workers
    if getattr(args, "analytics_auth_mode", None) is not None:
        data["analytics_journey_auth_mode"] = args.analytics_auth_mode
    if getattr(args, "analytics_page_size_cap", None) is not None:
        data["analytics_journey_details_page_size_cap"] = args.analytics_page_size_cap
    if getattr(args, "serial", False):
        data["attempt_parallel_enabled"] = False
        data["max_parallel_attempt_workers"] = 1

    return AppConfig(**data)


def _progress_printer(progress_queue, stop_event: threading.Event) -> None:
    """Print progress events from the queue to the console.

    Runs in a separate thread, consuming events until stop_event is set
    and the queue is drained.

    Args:
        progress_queue: Queue subscribed to the ProgressEmitter.
        stop_event: Event signaling that execution is complete.
    """
    while not stop_event.is_set() or not progress_queue.empty():
        try:
            event: ProgressEvent = progress_queue.get(timeout=0.5)
            _print_progress_event(event)
        except Exception:
            continue


def _print_progress_event(event: ProgressEvent) -> None:
    """Format and print a single progress event to the console.

    Args:
        event: The progress event to print.
    """
    prefix = {
        ProgressEventType.SUITE_STARTED: "🚀",
        ProgressEventType.SCENARIO_STARTED: "📋",
        ProgressEventType.ATTEMPT_COMPLETED: "  ✓" if event.success else "  ✗",
        ProgressEventType.SCENARIO_COMPLETED: "📊",
        ProgressEventType.SUITE_COMPLETED: "🏁",
    }.get(event.event_type, "•")

    print(f"{prefix} {event.message}")


def _print_report(report: TestReport) -> None:
    """Print a formatted test report summary to the console.

    Args:
        report: The TestReport to display.
    """
    print("\n" + "=" * 60)
    print(f"TEST REPORT: {report.suite_name}")
    print("=" * 60)
    print(f"Duration: {report.duration_seconds:.1f}s")
    print(
        f"Overall: {report.overall_successes}/{report.overall_attempts} "
        f"({report.overall_success_rate:.0%} success rate)"
    )
    print(f"Threshold: {report.regression_threshold:.0%}")
    print("-" * 60)

    for result in report.scenario_results:
        status = "⚠️  REGRESSION" if result.is_regression else "✅ PASS"
        print(
            f"  {result.scenario_name}: "
            f"{result.successes}/{result.attempts} "
            f"({result.success_rate:.0%}) — {status}"
        )

    print("-" * 60)
    if report.has_regressions:
        print("❌ REGRESSIONS DETECTED")
    else:
        print("✅ ALL SCENARIOS PASSED")
    print("=" * 60)


def _extract_attempt_durations(report: TestReport) -> list[float]:
    durations: list[float] = []
    for scenario in report.scenario_results:
        for attempt in scenario.attempt_results:
            if attempt.duration_seconds is None:
                continue
            durations.append(float(attempt.duration_seconds))
    return durations


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 1:
        return sorted_values[-1]
    rank = (len(sorted_values) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def _run_suite(config: AppConfig, suite, show_progress: bool) -> TestReport:
    emitter = ProgressEmitter()
    printer_thread = None
    stop_event = threading.Event()

    if show_progress:
        progress_queue = emitter.subscribe()
        printer_thread = threading.Thread(
            target=_progress_printer,
            args=(progress_queue, stop_event),
            daemon=True,
        )
        printer_thread.start()

    orchestrator = TestOrchestrator(config=config, progress_emitter=emitter)
    report = asyncio.run(orchestrator.run_suite(suite))

    if show_progress and printer_thread is not None:
        stop_event.set()
        printer_thread.join(timeout=5)

    return report


def _run_analytics_journey(
    config: AppConfig,
    run_request: AnalyticsJourneyRunRequest,
    show_progress: bool,
) -> TestReport:
    emitter = ProgressEmitter()
    printer_thread = None
    stop_event = threading.Event()

    if show_progress:
        progress_queue = emitter.subscribe()
        printer_thread = threading.Thread(
            target=_progress_printer,
            args=(progress_queue, stop_event),
            daemon=True,
        )
        printer_thread.start()

    runner = AnalyticsJourneyRunner(
        config=config,
        progress_emitter=emitter,
        stop_event=stop_event,
    )
    report = asyncio.run(runner.run(run_request))

    if show_progress and printer_thread is not None:
        stop_event.set()
        printer_thread.join(timeout=5)

    return report


def _build_report_summary(report: TestReport, wall_clock_seconds: float) -> dict:
    durations = sorted(_extract_attempt_durations(report))
    attempts_per_second = (
        float(report.overall_attempts) / wall_clock_seconds
        if wall_clock_seconds > 0
        else 0.0
    )
    return {
        "wall_clock_seconds": wall_clock_seconds,
        "attempts_per_second": attempts_per_second,
        "p50_attempt_duration_seconds": _percentile(durations, 0.50),
        "p95_attempt_duration_seconds": _percentile(durations, 0.95),
        "p99_attempt_duration_seconds": _percentile(durations, 0.99),
        "overall_attempts": report.overall_attempts,
        "overall_successes": report.overall_successes,
        "overall_failures": report.overall_failures,
        "overall_timeouts": report.overall_timeouts,
        "overall_skipped": report.overall_skipped,
        "overall_success_rate": report.overall_success_rate,
    }


def _run_benchmark(config: AppConfig, suite, candidate_workers: int) -> int:
    serial_config = config.model_copy(deep=True)
    serial_config.attempt_parallel_enabled = False
    serial_config.max_parallel_attempt_workers = 1

    parallel_config = config.model_copy(deep=True)
    parallel_config.attempt_parallel_enabled = True
    parallel_config.max_parallel_attempt_workers = max(1, min(int(candidate_workers), 3))

    print("\nRunning benchmark baseline (serial: workers=1)...")
    serial_start = time.perf_counter()
    serial_report = _run_suite(serial_config, suite, show_progress=False)
    serial_wall = max(0.001, time.perf_counter() - serial_start)

    print(
        f"Running benchmark candidate (parallel: workers={parallel_config.max_parallel_attempt_workers})..."
    )
    parallel_start = time.perf_counter()
    parallel_report = _run_suite(parallel_config, suite, show_progress=False)
    parallel_wall = max(0.001, time.perf_counter() - parallel_start)

    serial_summary = _build_report_summary(serial_report, serial_wall)
    parallel_summary = _build_report_summary(parallel_report, parallel_wall)

    runtime_improvement = max(0.0, (serial_wall - parallel_wall) / serial_wall)
    semantic_deltas = {
        "overall_attempts": parallel_report.overall_attempts - serial_report.overall_attempts,
        "overall_successes": parallel_report.overall_successes - serial_report.overall_successes,
        "overall_failures": parallel_report.overall_failures - serial_report.overall_failures,
        "overall_timeouts": parallel_report.overall_timeouts - serial_report.overall_timeouts,
        "overall_skipped": parallel_report.overall_skipped - serial_report.overall_skipped,
    }
    semantic_regression = any(delta != 0 for delta in semantic_deltas.values())
    benchmark_pass = (runtime_improvement >= 0.40) and (not semantic_regression)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": suite.name,
        "candidate_workers": parallel_config.max_parallel_attempt_workers,
        "benchmark_gate": {
            "required_runtime_improvement": 0.40,
            "runtime_improvement": runtime_improvement,
            "semantic_regression": semantic_regression,
            "semantic_deltas": semantic_deltas,
            "passed": benchmark_pass,
        },
        "serial": serial_summary,
        "parallel": parallel_summary,
    }

    benchmark_dir = Path(config.history_dir) / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = benchmark_dir / f"benchmark-{stamp}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBenchmark Summary")
    print("=" * 60)
    print(f"Serial wall-clock:   {serial_wall:.2f}s")
    print(f"Parallel wall-clock: {parallel_wall:.2f}s")
    print(f"Runtime improvement: {runtime_improvement:.1%}")
    print(f"Semantic regression: {'YES' if semantic_regression else 'NO'}")
    print(f"Gate pass:           {'YES' if benchmark_pass else 'NO'}")
    print(f"Artifact:            {output_path}")
    print("=" * 60)

    return 0 if benchmark_pass else 1


def main(argv=None) -> None:
    """CLI entry point. Parse args, load config, run suite, print report, exit with code.

    Args:
        argv: Optional argument list for testing (defaults to sys.argv[1:]).
    """
    args = _parse_args(argv)

    # Load base config from env vars / config file
    config = load_app_config()

    # Merge CLI overrides (highest precedence)
    config = _merge_cli_overrides(config, args)

    if args.command == "analytics-journey":
        try:
            auth_mode = normalize_analytics_auth_mode(
                args.analytics_auth_mode or config.analytics_journey_auth_mode
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if not str(args.interval or "").strip():
            print("Error: --interval is required for analytics-journey", file=sys.stderr)
            sys.exit(1)

        missing = []
        if not config.gc_region:
            missing.append("gc_region")
        if auth_mode == ANALYTICS_AUTH_MODE_CLIENT_CREDENTIALS:
            if not config.gc_client_id:
                missing.append("gc_client_id")
            if not config.gc_client_secret:
                missing.append("gc_client_secret")
        if auth_mode == ANALYTICS_AUTH_MODE_MANUAL_BEARER and not str(
            args.analytics_bearer_token or ""
        ).strip():
            missing.append("manual_bearer_token")
        if missing:
            print(
                "Error: Missing required analytics-journey configuration: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            extra_filter = (
                json.loads(args.filter_json)
                if str(args.filter_json or "").strip()
                else {}
            )
        except json.JSONDecodeError as e:
            print(f"Error: --filter-json must be valid JSON object: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(extra_filter, dict):
            print("Error: --filter-json must be a JSON object", file=sys.stderr)
            sys.exit(1)

        page_size = max(
            1,
            min(
                int(args.page_size),
                int(config.analytics_journey_details_page_size_cap),
            ),
        )
        max_conversations = max(1, int(args.max_conversations))
        divisions = [
            token.strip()
            for token in str(args.divisions or "").split(",")
            if token.strip()
        ]
        language_filter = str(args.language_filter or "").strip() or None
        run_request = AnalyticsJourneyRunRequest(
            bot_flow_id=str(args.bot_flow_id or "").strip(),
            interval=str(args.interval).strip(),
            page_size=page_size,
            max_conversations=max_conversations,
            auth_mode=auth_mode,
            manual_bearer_token=str(args.analytics_bearer_token or "").strip() or None,
            divisions=divisions,
            language_filter=language_filter,
            extra_query_params=extra_filter,
        )

        judge = build_judge_execution_client(config, analytics=True)
        try:
            judge.verify_connection()
        except JudgeLLMError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        report = _run_analytics_journey(config, run_request, show_progress=True)
        _print_report(report)
        if report.has_regressions:
            sys.exit(1)
        return

    # Load test suite
    try:
        suite = load_test_suite(args.test_suite)
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading test suite: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required config
    missing = validate_required_config(config)
    if missing:
        print(
            f"Error: Missing required configuration: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify Ollama connection once before runs.
    judge = build_judge_execution_client(config)
    try:
        judge.verify_connection()
    except JudgeLLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "benchmark":
        exit_code = _run_benchmark(config, suite, candidate_workers=args.candidate_workers)
        sys.exit(exit_code)

    report = _run_suite(config, suite, show_progress=True)

    # Print the formatted report
    _print_report(report)

    # Exit with non-zero code if regressions detected
    if report.has_regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
