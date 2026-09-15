from __future__ import annotations

import base64
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..config import settings


ALERT_MESSAGES = {
    "microsleep": "Cảnh báo, mắt nhắm kéo dài. Hãy giảm tốc độ và dừng xe tại vị trí an toàn.",
    "drowsy": "Cảnh báo buồn ngủ. Hãy tập trung và nghỉ ngơi khi cần.",
    "high_perclos": "Cảnh báo, tỷ lệ nhắm mắt đang ở mức cao.",
    "distraction": "Cảnh báo mất tập trung thị giác. Hãy nhìn về phía trước.",
    "phone": "Cảnh báo. Không sử dụng điện thoại khi lái xe.",
    "cigarette": "Cảnh báo. Không hút thuốc khi lái xe.",
    "no_seatbelt": "Cảnh báo. Hãy thắt dây an toàn.",
    "yawn": "Cảnh báo mệt mỏi. Hãy nghỉ ngơi khi cần.",
}
ALERT_PRIORITY = (
    "microsleep", "drowsy", "high_perclos", "distraction",
    "phone", "cigarette", "no_seatbelt", "yawn",
)


def selected_warning(warnings: list[str] | None) -> str:
    warning_set = set(warnings or [])
    return next((item for item in ALERT_PRIORITY if item in warning_set), "danger")


def audio_backend(warnings: list[str] | None = None) -> str:
    warning = selected_warning(warnings)
    audio_file = Path(__file__).with_name(f"audio_vi_{warning}.wav")
    if sys.platform == "win32" and audio_file.is_file():
        return "vi-VN WAV"
    return "Windows vi-VN TTS / beep" if sys.platform == "win32" else "terminal beep"


def play_vehicle_alert(severity: str, warnings: list[str] | None = None) -> None:
    """Play one non-blocking Vietnamese cabin warning, with local fallbacks."""
    if not settings.edge_audio_alert:
        return

    def emit() -> None:
        warning = selected_warning(warnings)
        try:
            if sys.platform == "win32":
                import winsound

                audio_file = Path(__file__).with_name(f"audio_vi_{warning}.wav")
                if audio_file.is_file():
                    winsound.PlaySound(str(audio_file), winsound.SND_FILENAME)
                    return

                message = ALERT_MESSAGES.get(warning, "Cảnh báo nguy hiểm trong cabin.")
                escaped = message.replace("'", "''")
                script = (
                    "Add-Type -AssemblyName System.Speech;"
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                    "$v=$s.GetInstalledVoices()|Where-Object {$_.VoiceInfo.Culture.Name -eq 'vi-VN'}|Select-Object -First 1;"
                    "if($null -eq $v){$s.Dispose();exit 3};"
                    "$s.SelectVoice($v.VoiceInfo.Name);"
                    f"$s.Speak('{escaped}');$s.Dispose()"
                )
                encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
                completed = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=12,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError("Không có giọng vi-VN trên Windows")
            else:
                print("\a", end="", flush=True)
        except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired):
            try:
                import winsound
                count = 3 if severity == "danger" else 2
                for _ in range(count):
                    winsound.Beep(1100 if severity == "danger" else 850, 180)
                    time.sleep(0.08)
            except (ImportError, OSError, RuntimeError):
                pass

    threading.Thread(target=emit, name="dms-edge-alert", daemon=True).start()
