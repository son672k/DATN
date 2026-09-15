from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dms_training.lstm import SubjectWiseLSTMClassifier


class LSTMPredictor:
    def __init__(self, checkpoint: Path, device: torch.device):
        try:
            metadata = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            metadata = torch.load(checkpoint, map_location=device)
        self.metadata = metadata
        self.device = device
        self.class_names = list(metadata.get("class_names", ["awake", "drowsy"]))
        self.sequence_length = int(metadata["seq_len"])
        self.feature_dim = int(metadata["input_dim"])
        self.model = SubjectWiseLSTMClassifier(
            input_dim=self.feature_dim,
            hidden_dim=int(metadata["hidden_dim"]),
            num_layers=int(metadata["num_layers"]),
            num_classes=int(metadata.get("num_classes", 2)),
            dropout=float(metadata.get("dropout", 0.3)),
        ).to(device)
        self.model.load_state_dict(metadata["model_state"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> dict:
        sequence = np.asarray(sequence, dtype=np.float32)
        expected = (self.sequence_length, self.feature_dim)
        if sequence.shape != expected:
            raise ValueError(f"Sequence phải có shape {expected}, nhận {sequence.shape}")
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.amp.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        values = probabilities.float().cpu().numpy()
        return {
            "label": self.class_names[int(values.argmax())],
            "confidence": float(values.max()),
            "probabilities": {
                name: float(values[index]) for index, name in enumerate(self.class_names)
            },
        }

