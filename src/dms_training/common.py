from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ["awake", "drowsy"]
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def project_src_on_path(script_file: str) -> Path:
    project_root = Path(script_file).resolve().parents[1]
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return project_root

