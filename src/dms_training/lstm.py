from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SubjectWiseLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(self.norm(sequence))
        weights = torch.softmax(self.attention(output), dim=1)
        context = (output * weights).sum(dim=1)
        return self.classifier(context)


class FeatureSequenceDataset(Dataset):
    def __init__(self, feature_path, sequence_path):
        self.features = np.load(feature_path, mmap_mode="r")
        payload = np.load(sequence_path, allow_pickle=False)
        self.indices = payload["indices"].astype(np.int64, copy=False)
        self.labels = payload["labels"].astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        # Copy the small selected window so PyTorch receives writable memory.
        sequence = np.array(self.features[self.indices[index]], dtype=np.float32)
        return torch.from_numpy(sequence), int(self.labels[index])

