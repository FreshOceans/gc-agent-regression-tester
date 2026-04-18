"""Unit tests for the ProgressEmitter."""

import queue
import threading

import pytest

from src.models import ProgressEvent, ProgressEventType
from src.progress import ProgressEmitter


def _make_event(event_type: ProgressEventType = ProgressEventType.SUITE_STARTED, message: str = "test") -> ProgressEvent:
    """Helper to create a ProgressEvent."""
    return ProgressEvent(event_type=event_type, message=message)


class TestProgressEmitterInit:
    """Tests for ProgressEmitter initialization."""

    def test_init_creates_empty_subscriber_list(self):
        emitter = ProgressEmitter()
        assert emitter._subscribers == []
        assert emitter.get_history() == []


class TestProgressEmitterSubscribe:
    """Tests for the subscribe method."""

    def test_subscribe_returns_queue(self):
        emitter = ProgressEmitter()
        q = emitter.subscribe()
        assert isinstance(q, queue.Queue)

    def test_subscribe_adds_to_subscribers(self):
        emitter = ProgressEmitter()
        q = emitter.subscribe()
        assert q in emitter._subscribers

    def test_multiple_subscribers(self):
        emitter = ProgressEmitter()
        q1 = emitter.subscribe()
        q2 = emitter.subscribe()
        assert len(emitter._subscribers) == 2
        assert q1 in emitter._subscribers
        assert q2 in emitter._subscribers


class TestProgressEmitterUnsubscribe:
    """Tests for the unsubscribe method."""

    def test_unsubscribe_removes_queue(self):
        emitter = ProgressEmitter()
        q = emitter.subscribe()
        emitter.unsubscribe(q)
        assert q not in emitter._subscribers

    def test_unsubscribe_nonexistent_queue_does_not_raise(self):
        emitter = ProgressEmitter()
        q: queue.Queue = queue.Queue()
        # Should not raise
        emitter.unsubscribe(q)

    def test_unsubscribe_only_removes_target(self):
        emitter = ProgressEmitter()
        q1 = emitter.subscribe()
        q2 = emitter.subscribe()
        emitter.unsubscribe(q1)
        assert q1 not in emitter._subscribers
        assert q2 in emitter._subscribers


class TestProgressEmitterEmit:
    """Tests for the emit method."""

    def test_emit_delivers_event_to_subscriber(self):
        emitter = ProgressEmitter()
        q = emitter.subscribe()
        event = _make_event()
        emitter.emit(event)
        received = q.get_nowait()
        assert received == event

    def test_emit_delivers_to_all_subscribers(self):
        emitter = ProgressEmitter()
        q1 = emitter.subscribe()
        q2 = emitter.subscribe()
        event = _make_event(message="broadcast")
        emitter.emit(event)
        assert q1.get_nowait() == event
        assert q2.get_nowait() == event

    def test_emit_does_not_deliver_to_unsubscribed(self):
        emitter = ProgressEmitter()
        q1 = emitter.subscribe()
        q2 = emitter.subscribe()
        emitter.unsubscribe(q1)
        event = _make_event()
        emitter.emit(event)
        assert q1.empty()
        assert q2.get_nowait() == event

    def test_emit_preserves_event_order(self):
        emitter = ProgressEmitter()
        q = emitter.subscribe()
        events = [
            _make_event(ProgressEventType.SUITE_STARTED, "start"),
            _make_event(ProgressEventType.SCENARIO_STARTED, "scenario"),
            _make_event(ProgressEventType.ATTEMPT_COMPLETED, "attempt"),
            _make_event(ProgressEventType.SUITE_COMPLETED, "done"),
        ]
        for e in events:
            emitter.emit(e)
        received = []
        while not q.empty():
            received.append(q.get_nowait())
        assert received == events

    def test_emit_prints_to_console(self, capsys):
        emitter = ProgressEmitter()
        event = _make_event(ProgressEventType.SCENARIO_STARTED, "Testing scenario X")
        emitter.emit(event)
        captured = capsys.readouterr()
        assert "[scenario_started]" in captured.out
        assert "Testing scenario X" in captured.out

    def test_emit_with_no_subscribers_does_not_raise(self):
        emitter = ProgressEmitter()
        event = _make_event()
        # Should not raise even with no subscribers
        emitter.emit(event)

    def test_emit_records_history(self):
        emitter = ProgressEmitter()
        e1 = _make_event(ProgressEventType.SUITE_STARTED, "start")
        e2 = _make_event(ProgressEventType.SUITE_COMPLETED, "done")
        emitter.emit(e1)
        emitter.emit(e2)

        history = emitter.get_history()
        assert history == [e1, e2]

    def test_get_history_limit_returns_recent_events(self):
        emitter = ProgressEmitter()
        events = [
            _make_event(ProgressEventType.SUITE_STARTED, "e1"),
            _make_event(ProgressEventType.SCENARIO_STARTED, "e2"),
            _make_event(ProgressEventType.ATTEMPT_COMPLETED, "e3"),
        ]
        for event in events:
            emitter.emit(event)

        assert emitter.get_history(limit=2) == events[-2:]


class TestProgressEmitterThreadSafety:
    """Tests for thread-safe behavior."""

    def test_concurrent_subscribe_and_emit(self):
        emitter = ProgressEmitter()
        results = []
        barrier = threading.Barrier(3)

        def subscriber_worker():
            barrier.wait()
            q = emitter.subscribe()
            # Wait a bit for events
            try:
                event = q.get(timeout=1)
                results.append(event)
            except queue.Empty:
                pass

        def emitter_worker():
            barrier.wait()
            for i in range(5):
                emitter.emit(_make_event(message=f"event-{i}"))

        threads = [
            threading.Thread(target=subscriber_worker),
            threading.Thread(target=subscriber_worker),
            threading.Thread(target=emitter_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # At least some events should have been received
        # (exact count depends on timing)
        # Main assertion: no exceptions were raised (thread safety)
