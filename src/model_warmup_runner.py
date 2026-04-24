"""Model warm-up runner for transport-only Web Messaging checks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event
from typing import Optional

from .models import (
    AppConfig,
    AttemptResult,
    Message,
    MessageRole,
    ModelWarmupRunMetadata,
    ProgressEvent,
    ProgressEventType,
    ScenarioResult,
    TestReport,
    TimeoutDiagnostics,
)
from .progress import ProgressEmitter
from .web_messaging_client import WebMessagingClient, WebMessagingError

MODEL_WARMUP_SUITE_NAME = "Model Warm Up Suite"
MODEL_WARMUP_SCENARIO_NAME = "No Help Needed Warm Up"
MODEL_WARMUP_FIXED_MESSAGE = "no help needed"
MODEL_WARMUP_ATTEMPTS = 227
MODEL_WARMUP_PACING_CHOICES = {2.5, 5.0, 7.5}


@dataclass(frozen=True)
class ModelWarmUpRunRequest:
    """Operator-selected inputs for a model warm-up run."""

    deployment_id: str
    region: str
    recorded_model: Optional[str] = None
    execution_mode: str = "serial"
    worker_count: int = 1
    pacing_seconds: float = 2.5


def normalize_model_warmup_execution_mode(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"serial", "parallel"}:
        raise ValueError("Model Warm Up execution mode must be serial or parallel.")
    return normalized


def normalize_model_warmup_workers(value: int | str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("Model Warm Up parallel workers must be a number.") from None
    return max(1, min(parsed, 5))


def normalize_model_warmup_pacing(value: float | str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("Model Warm Up pacing must be 2.5, 5.0, or 7.5 seconds.") from None
    if parsed not in MODEL_WARMUP_PACING_CHOICES:
        raise ValueError("Model Warm Up pacing must be 2.5, 5.0, or 7.5 seconds.")
    return parsed


def build_model_warmup_metadata(
    request: ModelWarmUpRunRequest,
    *,
    completed_attempts: int = 0,
) -> ModelWarmupRunMetadata:
    """Create report metadata for a model warm-up run."""

    execution_mode = normalize_model_warmup_execution_mode(request.execution_mode)
    worker_count = 1
    if execution_mode == "parallel":
        worker_count = normalize_model_warmup_workers(request.worker_count)
    return ModelWarmupRunMetadata(
        deployment_id=request.deployment_id,
        region=request.region,
        recorded_model=request.recorded_model,
        execution_mode=execution_mode,
        worker_count=worker_count,
        pacing_seconds=normalize_model_warmup_pacing(request.pacing_seconds),
        fixed_message=MODEL_WARMUP_FIXED_MESSAGE,
        planned_attempts=MODEL_WARMUP_ATTEMPTS,
        completed_attempts=max(0, int(completed_attempts)),
    )


class ModelWarmUpRunner:
    """Run Web Messaging transport attempts without Judge LLM evaluation."""

    def __init__(
        self,
        *,
        config: AppConfig,
        progress_emitter: ProgressEmitter,
        stop_event: Optional[Event] = None,
    ) -> None:
        self.config = config
        self.progress_emitter = progress_emitter
        self.stop_event = stop_event

    def _stop_requested(self) -> bool:
        return bool(self.stop_event is not None and self.stop_event.is_set())

    def _build_origin_from_region(self, region: str) -> str:
        normalized = (region or "").strip().lower()
        if normalized.startswith("https://"):
            normalized = normalized[len("https://") :]
        elif normalized.startswith("http://"):
            normalized = normalized[len("http://") :]
        normalized = normalized.split("/", 1)[0]
        if normalized.startswith("apps."):
            return f"https://{normalized}"
        if normalized.startswith("webmessaging."):
            normalized = normalized[len("webmessaging.") :]
        if not normalized:
            normalized = "mypurecloud.com"
        return f"https://apps.{normalized}"

    def _step_log_entry(
        self,
        step_log: list[dict],
        *,
        stage: str,
        message: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "message": message,
        }
        if duration_ms is not None:
            entry["duration_ms"] = max(0.0, float(duration_ms))
        step_log.append(entry)

    async def _run_step(
        self,
        *,
        step_log: list[dict],
        status_callback,
        start_stage: str,
        complete_stage: str,
        message: str,
        awaitable,
    ):
        if self._stop_requested():
            raise asyncio.CancelledError("Model warm-up stop requested")
        status_callback(message)
        started = time.monotonic()
        self._step_log_entry(step_log, stage=start_stage, message=message)
        try:
            result = await awaitable
        except Exception:
            self._step_log_entry(
                step_log,
                stage=f"{start_stage.rsplit('_', 1)[0]}_error",
                message=f"{message} failed",
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise
        self._step_log_entry(
            step_log,
            stage=complete_stage,
            message=f"{message} complete",
            duration_ms=(time.monotonic() - started) * 1000,
        )
        return result

    async def _run_attempt(
        self,
        request: ModelWarmUpRunRequest,
        *,
        attempt_number: int,
        status_callback,
    ) -> AttemptResult:
        conversation: list[Message] = []
        step_log: list[dict] = []
        started_at = datetime.now(timezone.utc)
        attempt_started = time.monotonic()
        last_step_name: Optional[str] = None
        client = WebMessagingClient(
            region=request.region,
            deployment_id=request.deployment_id,
            timeout=self.config.response_timeout,
            origin=self._build_origin_from_region(request.region),
            debug_capture_frames=self.config.debug_capture_frames,
            debug_capture_frame_limit=self.config.debug_capture_frame_limit,
        )

        def build_result(
            *,
            success: bool,
            explanation: str,
            error: Optional[str] = None,
            timed_out: bool = False,
            skipped: bool = False,
            timeout_class: Optional[str] = None,
        ) -> AttemptResult:
            duration_seconds = time.monotonic() - attempt_started
            timeout_diagnostics = None
            if timed_out:
                timeout_diagnostics = TimeoutDiagnostics(
                    timeout_class=timeout_class or "model_warm_up_timeout",
                    step_name=last_step_name,
                    configured_timeout_seconds=float(self.config.response_timeout),
                    elapsed_attempt_seconds=duration_seconds,
                    conversation_total_messages=len(conversation),
                    conversation_user_messages=sum(
                        1 for message in conversation if message.role == MessageRole.USER
                    ),
                    conversation_agent_messages=sum(
                        1 for message in conversation if message.role == MessageRole.AGENT
                    ),
                )
            return AttemptResult(
                attempt_number=attempt_number,
                success=success,
                conversation=conversation,
                explanation=explanation,
                error=error,
                timed_out=timed_out,
                skipped=skipped,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                duration_seconds=duration_seconds,
                step_log=step_log,
                timeout_diagnostics=timeout_diagnostics,
            )

        async def run_recorded_step(stage_prefix: str, message: str, awaitable):
            nonlocal last_step_name
            last_step_name = message
            return await self._run_step(
                step_log=step_log,
                status_callback=status_callback,
                start_stage=f"{stage_prefix}_start",
                complete_stage=f"{stage_prefix}_complete",
                message=message,
                awaitable=awaitable,
            )

        result_payload = {
            "success": False,
            "explanation": "Model warm-up attempt failed before completion.",
            "error": None,
            "timed_out": False,
            "skipped": False,
        }
        try:
            await run_recorded_step(
                "connect",
                "Connecting to Web Messaging",
                client.connect(),
            )
            await run_recorded_step("join", "Sending join event", client.send_join())
            welcome = await run_recorded_step(
                "welcome_wait",
                "Waiting for welcome message",
                client.wait_for_welcome(),
            )
            conversation.append(
                Message(
                    role=MessageRole.AGENT,
                    content=welcome,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            conversation.append(
                Message(
                    role=MessageRole.USER,
                    content=MODEL_WARMUP_FIXED_MESSAGE,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            await run_recorded_step(
                "message_send",
                f"Sending warm-up message: {MODEL_WARMUP_FIXED_MESSAGE}",
                client.send_message(MODEL_WARMUP_FIXED_MESSAGE),
            )
            agent_response = await run_recorded_step(
                "agent_response_wait",
                "Waiting for agent response",
                client.receive_response(),
            )
            conversation.append(
                Message(
                    role=MessageRole.AGENT,
                    content=agent_response,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            result_payload = {
                "success": True,
                "explanation": "Model warm-up completed; no judgement performed.",
            }
        except asyncio.CancelledError as exc:
            result_payload = {
                "success": False,
                "explanation": "Model warm-up attempt stopped before completion.",
                "error": str(exc),
                "skipped": True,
            }
        except TimeoutError as exc:
            result_payload = {
                "success": False,
                "explanation": "Model warm-up attempt timed out; no judgement performed.",
                "error": str(exc),
                "timed_out": True,
            }
        except WebMessagingError as exc:
            result_payload = {
                "success": False,
                "explanation": (
                    "Model warm-up attempt failed due to Web Messaging error; "
                    "no judgement performed."
                ),
                "error": str(exc),
            }
        except Exception as exc:
            result_payload = {
                "success": False,
                "explanation": "Model warm-up attempt failed; no judgement performed.",
                "error": str(exc),
            }
        finally:
            disconnect_started = time.monotonic()
            self._step_log_entry(
                step_log,
                stage="disconnect_start",
                message="Disconnecting from Web Messaging",
            )
            try:
                await client.disconnect()
            finally:
                self._step_log_entry(
                    step_log,
                    stage="disconnect_complete",
                    message="Disconnect complete",
                    duration_ms=(time.monotonic() - disconnect_started) * 1000,
                )
        return build_result(**result_payload)

    async def run(self, request: ModelWarmUpRunRequest) -> TestReport:
        """Execute the fixed model warm-up suite."""

        started = time.monotonic()
        execution_mode = normalize_model_warmup_execution_mode(request.execution_mode)
        worker_count = (
            normalize_model_warmup_workers(request.worker_count)
            if execution_mode == "parallel"
            else 1
        )
        pacing_seconds = normalize_model_warmup_pacing(request.pacing_seconds)
        completed_attempts = 0
        successes = 0
        timeouts = 0
        skipped = 0
        attempts: list[AttemptResult] = []
        attempt_queue: asyncio.Queue[int] = asyncio.Queue()
        for attempt_number in range(1, MODEL_WARMUP_ATTEMPTS + 1):
            attempt_queue.put_nowait(attempt_number)
        event_lock = asyncio.Lock()

        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_STARTED,
                suite_name=MODEL_WARMUP_SUITE_NAME,
                message=f"Starting model warm-up suite: {MODEL_WARMUP_SUITE_NAME}",
                planned_attempts=MODEL_WARMUP_ATTEMPTS,
                completed_attempts=completed_attempts,
            )
        )
        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.SCENARIO_STARTED,
                suite_name=MODEL_WARMUP_SUITE_NAME,
                scenario_name=MODEL_WARMUP_SCENARIO_NAME,
                message=(
                    f"Starting scenario: {MODEL_WARMUP_SCENARIO_NAME} "
                    f"({MODEL_WARMUP_ATTEMPTS} attempts)"
                ),
                planned_attempts=MODEL_WARMUP_ATTEMPTS,
                completed_attempts=completed_attempts,
            )
        )
        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.ATTEMPT_STATUS,
                suite_name=MODEL_WARMUP_SUITE_NAME,
                scenario_name=MODEL_WARMUP_SCENARIO_NAME,
                message=(
                    "Model Warm Up configured: "
                    f"mode={execution_mode}, workers={worker_count}, "
                    f"pacing={pacing_seconds:.1f}s, model={request.recorded_model or 'not recorded'}"
                ),
                planned_attempts=MODEL_WARMUP_ATTEMPTS,
                completed_attempts=completed_attempts,
            )
        )

        async def worker(worker_index: int) -> None:
            nonlocal completed_attempts, successes, timeouts, skipped
            last_start_monotonic: Optional[float] = None
            while not self._stop_requested():
                try:
                    attempt_number = attempt_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if last_start_monotonic is not None:
                    remaining = max(
                        0.0,
                        pacing_seconds - (time.monotonic() - last_start_monotonic),
                    )
                    while remaining > 0:
                        if self._stop_requested():
                            return
                        tick = min(0.2, remaining)
                        await asyncio.sleep(tick)
                        remaining -= tick
                if self._stop_requested():
                    return
                last_start_monotonic = time.monotonic()

                async with event_lock:
                    self.progress_emitter.emit(
                        ProgressEvent(
                            event_type=ProgressEventType.ATTEMPT_STARTED,
                            suite_name=MODEL_WARMUP_SUITE_NAME,
                            scenario_name=MODEL_WARMUP_SCENARIO_NAME,
                            attempt_number=attempt_number,
                            message=f"Attempt {attempt_number} started",
                            planned_attempts=MODEL_WARMUP_ATTEMPTS,
                            completed_attempts=completed_attempts,
                        )
                    )

                def emit_attempt_status(status_message: str) -> None:
                    self.progress_emitter.emit(
                        ProgressEvent(
                            event_type=ProgressEventType.ATTEMPT_STATUS,
                            suite_name=MODEL_WARMUP_SUITE_NAME,
                            scenario_name=MODEL_WARMUP_SCENARIO_NAME,
                            attempt_number=attempt_number,
                            message=f"Worker {worker_index}: {status_message}",
                            planned_attempts=MODEL_WARMUP_ATTEMPTS,
                            completed_attempts=completed_attempts,
                        )
                    )

                result = await self._run_attempt(
                    request,
                    attempt_number=attempt_number,
                    status_callback=emit_attempt_status,
                )

                async with event_lock:
                    attempts.append(result)
                    completed_attempts += 1
                    if result.success:
                        successes += 1
                    if result.timed_out:
                        timeouts += 1
                    if result.skipped:
                        skipped += 1
                    self.progress_emitter.emit(
                        ProgressEvent(
                            event_type=ProgressEventType.ATTEMPT_COMPLETED,
                            suite_name=MODEL_WARMUP_SUITE_NAME,
                            scenario_name=MODEL_WARMUP_SCENARIO_NAME,
                            attempt_number=result.attempt_number,
                            success=result.success,
                            message=(
                                f"Attempt {result.attempt_number}: "
                                f"{'success' if result.success else 'failure'} "
                                f"({completed_attempts}/{MODEL_WARMUP_ATTEMPTS})"
                            ),
                            attempt_result=result,
                            planned_attempts=MODEL_WARMUP_ATTEMPTS,
                            completed_attempts=completed_attempts,
                        )
                    )

        workers = [asyncio.create_task(worker(index + 1)) for index in range(worker_count)]
        await asyncio.gather(*workers)

        attempts.sort(key=lambda attempt: attempt.attempt_number)
        failures = max(0, len(attempts) - successes - timeouts - skipped)
        success_rate = successes / len(attempts) if attempts else 0.0
        scenario = ScenarioResult(
            scenario_name=MODEL_WARMUP_SCENARIO_NAME,
            attempts=len(attempts),
            successes=successes,
            failures=failures,
            timeouts=timeouts,
            skipped=skipped,
            success_rate=success_rate,
            is_regression=success_rate < self.config.success_threshold if attempts else False,
            attempt_results=attempts,
        )
        duration = time.monotonic() - started
        report = TestReport(
            suite_name=MODEL_WARMUP_SUITE_NAME,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=duration,
            scenario_results=[scenario] if attempts else [],
            overall_attempts=len(attempts),
            overall_successes=successes,
            overall_failures=failures,
            overall_timeouts=timeouts,
            overall_skipped=skipped,
            overall_success_rate=success_rate,
            model_warmup_run=build_model_warmup_metadata(
                request,
                completed_attempts=len(attempts),
            ),
            stopped_by_user=self._stop_requested(),
            stop_mode="immediate" if self._stop_requested() else None,
            has_regressions=scenario.is_regression if attempts else False,
            regression_threshold=self.config.success_threshold,
        )

        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.SCENARIO_COMPLETED,
                suite_name=MODEL_WARMUP_SUITE_NAME,
                scenario_name=MODEL_WARMUP_SCENARIO_NAME,
                success_rate=success_rate,
                message=(
                    f"Scenario completed: {MODEL_WARMUP_SCENARIO_NAME} — "
                    f"{success_rate:.0%} completion rate"
                ),
                planned_attempts=MODEL_WARMUP_ATTEMPTS,
                completed_attempts=completed_attempts,
            )
        )
        completed_message = (
            f"Model warm-up completed: {MODEL_WARMUP_SUITE_NAME} in {duration:.1f}s"
        )
        if self._stop_requested():
            completed_message = (
                f"Model warm-up stopped early: {MODEL_WARMUP_SUITE_NAME} "
                f"after {duration:.1f}s"
            )
        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.SUITE_COMPLETED,
                suite_name=MODEL_WARMUP_SUITE_NAME,
                message=completed_message,
                duration_seconds=duration,
                planned_attempts=MODEL_WARMUP_ATTEMPTS,
                completed_attempts=completed_attempts,
            )
        )
        return report
