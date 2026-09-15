from __future__ import annotations

import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIP_ROOT = PROJECT_ROOT / "outputs" / "backend_sessions"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clips = sorted(CLIP_ROOT.glob("*/clips/*.mp4"))
    backup_root = PROJECT_ROOT / "outputs" / "backups" / "clips_before_h264" / datetime.now().strftime("%Y%m%d-%H%M%S")
    converted = 0
    failed = 0
    for clip in clips:
        target = clip.with_name(f"{clip.stem}.h264.mp4")
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(clip), "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            backup = backup_root / clip.relative_to(CLIP_ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clip, backup)
            target.replace(clip)
            converted += 1
            print(f"OK: {clip.relative_to(PROJECT_ROOT)}")
        else:
            target.unlink(missing_ok=True)
            failed += 1
            print(f"FAIL: {clip.name}", file=sys.stderr)
    print(f"Hoàn tất: {converted} clip H.264; {failed} lỗi.")
    if converted:
        print(f"Bản gốc được giữ tại: {backup_root.relative_to(PROJECT_ROOT)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
