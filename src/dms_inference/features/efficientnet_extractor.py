from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dms_training.cnn import eval_transform, extract_features, load_cnn_checkpoint


class EfficientNetExtractor:
    def __init__(self, checkpoint: Path, device: torch.device):
        self.device = device
        self.model, self.metadata = load_cnn_checkpoint(checkpoint, device)
        self.transform = eval_transform()
        self.class_names = list(self.metadata.get("class_names", ["awake", "drowsy"]))
        self.feature_dim = int(self.metadata.get("feature_dim", 1280))

    @torch.no_grad()
    def predict_and_extract(self, face_bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.amp.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            feature = extract_features(self.model, tensor)
            logits = self.model.classifier(feature)
            probabilities = torch.softmax(logits, dim=1)[0]
        values = probabilities.float().cpu().numpy()
        return {
            "label": self.class_names[int(values.argmax())],
            "confidence": float(values.max()),
            "probabilities": {
                name: float(values[index]) for index, name in enumerate(self.class_names)
            },
            "feature": feature.float().cpu().numpy()[0],
        }

