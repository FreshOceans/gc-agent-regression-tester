"""Progress event emitter for the GC Agent Regression Tester.

Provides thread-safe event distribution to multiple subscribers using queue.Queue.
"""

import queue
import threading
from typing import List

from .models import ProgressEvent


class ProgressEmitter:
    """Publishes progress events to subscribers via thread-safe queues.

    Subscribers receive events through individual queue.Queue instances,
    enabling both SSE (web) and console consumers to receive updates independently.
    """

    def __init__(self) -> None:
        """Initialize with empty subscriber list."""
        self._subscribers: List[queue.Queue] = []
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
            for q in self._subscribers:
                q.put_nowait(event)
