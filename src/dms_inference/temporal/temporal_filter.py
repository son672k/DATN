from __future__ import annotations

from collections import deque


class TemporalFilter:
    """Probability smoothing with hysteresis for stable UI warnings."""

    def __init__(
        self,
        window_size: int = 8,
        enter_drowsy: float = 0.65,
        exit_drowsy: float = 0.45,
        min_observations: int = 4,
    ):
        if exit_drowsy >= enter_drowsy:
            raise ValueError("exit_drowsy phải nhỏ hơn enter_drowsy")
        self.history = deque(maxlen=int(window_size))
        self.enter_drowsy = float(enter_drowsy)
        self.exit_drowsy = float(exit_drowsy)
        self.min_observations = int(min_observations)
        self.state = "awake"

    def update(self, drowsy_probability: float) -> dict:
        probability = min(1.0, max(0.0, float(drowsy_probability)))
        self.history.append(probability)
        smoothed = sum(self.history) / len(self.history)
        ready = len(self.history) >= self.min_observations
        if ready:
            if self.state == "awake" and smoothed >= self.enter_drowsy:
                self.state = "drowsy"
            elif self.state == "drowsy" and smoothed <= self.exit_drowsy:
                self.state = "awake"
        return {
            "state": self.state,
            "smoothed_drowsy_probability": float(smoothed),
            "ready": ready,
            "observations": len(self.history),
        }

    def reset(self):
        self.history.clear()
        self.state = "awake"

