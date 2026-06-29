from __future__ import annotations

from collections import defaultdict, deque

from warehouse_ai.core.events import WarehouseEvent


class InMemoryEventStream:
    """Redis Streams and Kafka-compatible interface for local orchestration."""

    def __init__(self, retention: int = 5000) -> None:
        self.retention = retention
        self.topics: dict[str, deque[WarehouseEvent]] = defaultdict(
            lambda: deque(maxlen=self.retention)
        )

    def publish(self, event: WarehouseEvent) -> None:
        self.topics[event.topic].append(event)

    def extend(self, events: list[WarehouseEvent]) -> None:
        for event in events:
            self.publish(event)

    def latest(self, topic: str, limit: int = 50) -> list[WarehouseEvent]:
        queue = self.topics.get(topic)
        if not queue:
            return []
        return list(queue)[-limit:]

    def tail_all(self, limit: int = 200) -> list[WarehouseEvent]:
        merged: list[WarehouseEvent] = []
        for queue in self.topics.values():
            merged.extend(list(queue))
        return sorted(merged, key=lambda event: event.timestamp)[-limit:]

