from __future__ import annotations

from collections import deque


class PerclosTracker:
    """Rolling fraction of valid closed-eye observations over camera time."""

    def __init__(
        self,
        window_size: int = 150,
        min_observations: int = 30,
        window_seconds: float | None = None,
        min_duration_seconds: float = 0.0,
    ):
        self.history = deque(maxlen=None if window_seconds else int(window_size))
        self.min_observations = int(min_observations)
        self.window_seconds = float(window_seconds) if window_seconds else None
        self.min_duration_seconds = float(min_duration_seconds)

    def update(self, eye_closed: bool | None, timestamp_seconds: float | None = None) -> dict:
        timestamp = float(timestamp_seconds) if timestamp_seconds is not None else None
        if self.window_seconds is not None and timestamp is None:
            previous = self.history[-1][0] if self.history else -1 / 30
            timestamp = float(previous) + 1 / 30
        if self.window_seconds is not None:
            cutoff = timestamp - self.window_seconds
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()
        if eye_closed is not None:
            self.history.append((timestamp, bool(eye_closed)) if self.window_seconds is not None else bool(eye_closed))
        values = [item[1] for item in self.history] if self.window_seconds is not None else list(self.history)
        value = sum(values) / len(values) if values else 0.0
        duration = 0.0
        timed = [item for item in self.history if item[0] is not None] if self.window_seconds is not None else []
        if len(timed) >= 2:
            duration = max(0.0, timed[-1][0] - timed[0][0])
        return {
            "value": float(value),
            "ready": len(values) >= self.min_observations and duration >= self.min_duration_seconds,
            "observations": len(values),
            "duration_seconds": float(duration),
        }

    def reset(self):
        self.history.clear()
