from __future__ import annotations


class BBoxSmoother:
    """EMA box smoother; intentionally simpler than multi-object ByteTrack."""

    def __init__(self, alpha: float = 0.65):
        if not 0 < alpha <= 1:
            raise ValueError("alpha phải nằm trong (0, 1]")
        self.alpha = float(alpha)
        self._box = None

    def update(self, box: tuple[int, int, int, int]):
        if self._box is None:
            self._box = tuple(float(value) for value in box)
        else:
            self._box = tuple(
                self.alpha * new + (1 - self.alpha) * old
                for new, old in zip(box, self._box)
            )
        return tuple(int(round(value)) for value in self._box)

    def reset(self):
        self._box = None

