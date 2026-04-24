"""Tests for the transport-only Model Warm Up runner."""

import asyncio

import pytest

from src.model_warmup_runner import (
    MODEL_WARMUP_ATTEMPTS,
    MODEL_WARMUP_FIXED_MESSAGE,
    ModelWarmUpRunRequest,
    ModelWarmUpRunner,
    build_model_warmup_metadata,
)
from src.models import AppConfig, MessageRole
from src.progress import ProgressEmitter


class _FakeWebMessagingClient:
    active_connections = 0
    max_active_connections = 0

    def __init__(self, *args, **kwargs):
        self.raise_timeout = kwargs.get("deployment_id") == "timeout"

    async def connect(self):
        type(self).active_connections += 1
        type(self).max_active_connections = max(
            type(self).max_active_connections,
            type(self).active_connections,
        )
        await asyncio.sleep(0.01)

    async def send_join(self):
        await asyncio.sleep(0)

    async def wait_for_welcome(self):
        if self.raise_timeout:
            raise TimeoutError("welcome timed out")
        return "Welcome"

    async def send_message(self, text):
        self.sent_message = text
        await asyncio.sleep(0)

    async def receive_response(self):
        return "Goodbye"

    async def disconnect(self):
        type(self).active_connections = max(0, type(self).active_connections - 1)


@pytest.fixture(autouse=True)
def reset_fake_client():
    _FakeWebMessagingClient.active_connections = 0
    _FakeWebMessagingClient.max_active_connections = 0


def _config() -> AppConfig:
    return AppConfig(
        gc_region="usw2.pure.cloud",
        gc_deployment_id="deploy-id",
        response_timeout=5,
        success_threshold=0.8,
    )


def test_model_warmup_metadata_uses_fixed_227_attempts():
    metadata = build_model_warmup_metadata(
        ModelWarmUpRunRequest(
            deployment_id="deploy-id",
            region="usw2.pure.cloud",
            recorded_model="gemma4:e4b",
            execution_mode="parallel",
            worker_count=9,
            pacing_seconds=2.5,
        )
    )

    assert MODEL_WARMUP_ATTEMPTS == 227
    assert metadata.planned_attempts == 227
    assert metadata.worker_count == 5
    assert metadata.fixed_message == MODEL_WARMUP_FIXED_MESSAGE


@pytest.mark.asyncio
async def test_model_warmup_success_records_conversation_and_steps(monkeypatch):
    monkeypatch.setattr("src.model_warmup_runner.MODEL_WARMUP_ATTEMPTS", 1)
    monkeypatch.setattr(
        "src.model_warmup_runner.WebMessagingClient",
        _FakeWebMessagingClient,
    )
    runner = ModelWarmUpRunner(config=_config(), progress_emitter=ProgressEmitter())

    report = await runner.run(
        ModelWarmUpRunRequest(
            deployment_id="deploy-id",
            region="usw2.pure.cloud",
            recorded_model="gemma4:e4b",
            execution_mode="serial",
            worker_count=1,
            pacing_seconds=2.5,
        )
    )

    attempt = report.scenario_results[0].attempt_results[0]
    assert report.overall_attempts == 1
    assert report.overall_successes == 1
    assert report.model_warmup_run is not None
    assert report.model_warmup_run.recorded_model == "gemma4:e4b"
    assert [message.role for message in attempt.conversation] == [
        MessageRole.AGENT,
        MessageRole.USER,
        MessageRole.AGENT,
    ]
    assert attempt.conversation[1].content == "no help needed"
    assert attempt.judge_diagnostics == []
    stages = {entry["stage"] for entry in attempt.step_log}
    assert "connect_start" in stages
    assert "agent_response_wait_complete" in stages
    assert "disconnect_complete" in stages
    assert any("duration_ms" in entry for entry in attempt.step_log)


@pytest.mark.asyncio
async def test_model_warmup_timeout_marks_attempt_timed_out(monkeypatch):
    monkeypatch.setattr("src.model_warmup_runner.MODEL_WARMUP_ATTEMPTS", 1)
    monkeypatch.setattr(
        "src.model_warmup_runner.WebMessagingClient",
        _FakeWebMessagingClient,
    )
    runner = ModelWarmUpRunner(config=_config(), progress_emitter=ProgressEmitter())

    report = await runner.run(
        ModelWarmUpRunRequest(
            deployment_id="timeout",
            region="usw2.pure.cloud",
            execution_mode="serial",
            worker_count=1,
            pacing_seconds=2.5,
        )
    )

    attempt = report.scenario_results[0].attempt_results[0]
    assert report.overall_timeouts == 1
    assert attempt.timed_out is True
    assert attempt.timeout_diagnostics is not None
    assert attempt.timeout_diagnostics.timeout_class == "model_warm_up_timeout"


@pytest.mark.asyncio
async def test_model_warmup_parallel_mode_uses_selected_workers(monkeypatch):
    monkeypatch.setattr("src.model_warmup_runner.MODEL_WARMUP_ATTEMPTS", 5)
    monkeypatch.setattr(
        "src.model_warmup_runner.WebMessagingClient",
        _FakeWebMessagingClient,
    )
    runner = ModelWarmUpRunner(config=_config(), progress_emitter=ProgressEmitter())

    report = await runner.run(
        ModelWarmUpRunRequest(
            deployment_id="deploy-id",
            region="usw2.pure.cloud",
            execution_mode="parallel",
            worker_count=5,
            pacing_seconds=2.5,
        )
    )

    assert report.overall_attempts == 5
    assert report.overall_successes == 5
    assert _FakeWebMessagingClient.max_active_connections == 5
    assert report.model_warmup_run is not None
    assert report.model_warmup_run.worker_count == 5
