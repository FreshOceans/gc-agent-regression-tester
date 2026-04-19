"""CLI entry point for the Regression Test Harness.

Parses command-line arguments, loads configuration, runs the test suite,
prints progress and results to the console, and exits with a non-zero
code if regressions are detected.
"""

import argparse
import asyncio
import sys
import threading

from .app_config import load_app_config, validate_required_config
from .config_loader import load_test_suite
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import AppConfig, ProgressEvent, ProgressEventType
from .orchestrator import TestOrchestrator
from .progress import ProgressEmitter


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
        help="Ollama model name override",
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
        "--threshold",
        type=float,
        help="Success threshold (0.0-1.0) override",
    )
    parser.add_argument(
        "--language",
        help="Run language override (en, fr, fr-CA, es)",
    )
    return parser.parse_args(argv)


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

    if args.region is not None:
        data["gc_region"] = args.region
    if args.deployment_id is not None:
        data["gc_deployment_id"] = args.deployment_id
    if args.ollama_url is not None:
        data["ollama_base_url"] = args.ollama_url
    if args.ollama_model is not None:
        data["ollama_model"] = args.ollama_model
    if args.attempts is not None:
        data["default_attempts"] = args.attempts
    if args.max_turns is not None:
        data["max_turns"] = args.max_turns
    if args.timeout is not None:
        data["response_timeout"] = args.timeout
    if args.threshold is not None:
        data["success_threshold"] = args.threshold
    if args.language is not None:
        data["language"] = args.language

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


def _print_report(report) -> None:
    """Print a formatted test report summary to the console.

    Args:
        report: The TestReport to display.
    """
    print("\n" + "=" * 60)
    print(f"TEST REPORT: {report.suite_name}")
    print("=" * 60)
    print(f"Duration: {report.duration_seconds:.1f}s")
    print(f"Overall: {report.overall_successes}/{report.overall_attempts} "
          f"({report.overall_success_rate:.0%} success rate)")
    print(f"Threshold: {report.regression_threshold:.0%}")
    print("-" * 60)

    for result in report.scenario_results:
        status = "⚠️  REGRESSION" if result.is_regression else "✅ PASS"
        print(f"  {result.scenario_name}: "
              f"{result.successes}/{result.attempts} "
              f"({result.success_rate:.0%}) — {status}")

    print("-" * 60)
    if report.has_regressions:
        print("❌ REGRESSIONS DETECTED")
    else:
        print("✅ ALL SCENARIOS PASSED")
    print("=" * 60)


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

    # Verify Ollama connection
    judge = JudgeLLMClient(
        base_url=config.ollama_base_url,
        model=config.ollama_model or "",
        timeout=config.response_timeout,
    )
    try:
        judge.verify_connection()
    except JudgeLLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up progress emitter and console printer thread
    emitter = ProgressEmitter()
    progress_queue = emitter.subscribe()
    stop_event = threading.Event()
    printer_thread = threading.Thread(
        target=_progress_printer,
        args=(progress_queue, stop_event),
        daemon=True,
    )
    printer_thread.start()

    # Run the test suite
    orchestrator = TestOrchestrator(config=config, progress_emitter=emitter)
    report = asyncio.run(orchestrator.run_suite(suite))

    # Signal printer thread to stop and wait for it
    stop_event.set()
    printer_thread.join(timeout=5)

    # Print the formatted report
    _print_report(report)

    # Exit with non-zero code if regressions detected
    if report.has_regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
