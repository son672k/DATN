#!/usr/bin/env python3
"""Generate the offline Vietnamese warning WAV files used by the Edge demo."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".tools" / "edge_tts"))

from backend.app.services.audio_alert_service import ALERT_MESSAGES  # noqa: E402


async def generate(voice: str) -> None:
    try:
        import edge_tts
        import imageio_ffmpeg
    except ImportError as error:
        raise SystemExit(
            "Thiếu edge-tts hoặc imageio-ffmpeg. Cài edge-tts vào .tools/edge_tts trước."
        ) from error

    output_dir = PROJECT_ROOT / "backend" / "app" / "services"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    messages = {**ALERT_MESSAGES, "danger": "Cảnh báo nguy hiểm trong cabin. Hãy giảm tốc độ và xử lý an toàn."}
    with tempfile.TemporaryDirectory(prefix="dms-vi-audio-") as temporary:
        temporary_dir = Path(temporary)
        for name, message in messages.items():
            mp3 = temporary_dir / f"{name}.mp3"
            target = output_dir / f"audio_vi_{name}.wav"
            await edge_tts.Communicate(message, voice, rate="-4%").save(str(mp3))
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target)],
                check=True,
            )
            print(f"created {target.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="vi-VN-HoaiMyNeural")
    args = parser.parse_args()
    asyncio.run(generate(args.voice))


if __name__ == "__main__":
    main()
