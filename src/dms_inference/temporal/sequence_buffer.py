from __future__ import annotations

from collections import deque

import numpy as np


class SequenceBuffer:
    def __init__(self, sequence_length: int = 16, feature_dim: int = 1280):
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self._buffer = deque(maxlen=self.sequence_length)

    def push(self, feature: np.ndarray):
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (self.feature_dim,):
            raise ValueError(
                f"Feature phải có shape ({self.feature_dim},), nhận {feature.shape}"
            )
        self._buffer.append(feature.copy())

    @property
    def ready(self) -> bool:
        return len(self._buffer) == self.sequence_length

    def get(self) -> np.ndarray | None:
        return np.stack(self._buffer) if self.ready else None

    def reset(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

