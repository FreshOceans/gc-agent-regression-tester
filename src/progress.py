"""Progress event emitter for the Regression Test Harness.

Provides thread-safe event distribution to multiple subscribers using queue.Queue.
"""

import queue
import threading
from typing import List, Optional

from .models import ProgressEvent


class ProgressEmitter:
    """Publishes progress events to subscribers via thread-safe queues.

    Subscribers receive events through individual queue.Queue instances,
    enabling both SSE (web) and console consumers to receive updates independently.
    """

    def __init__(self) -> None:
        """Initialize with empty subscriber list."""
        self._subscribers: List[queue.Queue] = []
        self._history: List[ProgressEvent] = []
        self._history_limit = 500
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        """Return a new queue that will receive progress events.

        Returns:
            A queue.Queue instance that will receive all future ProgressEvent objects.
        """
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove a subscriber so it no longer receives events.

        Args:
            q: The queue previously returned by subscribe().
        """
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass  # Already removed or never subscribed

    def emit(self, event: ProgressEvent) -> None:
        """Publish a progress event to all subscribers and print to console.

        Args:
            event: The ProgressEvent to distribute.
        """
        print(f"[{event.event_type.value}] {event.message}")
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]
            for q in self._subscribers:
                q.put_nowait(event)

    def get_history(self, limit: Optional[int] = None) -> List[ProgressEvent]:
        """Return a snapshot of emitted events, optionally limited to most recent N."""
        with self._lock:
            history = list(self._history)
        if limit is not None and limit > 0:
            return history[-limit:]
        return history
