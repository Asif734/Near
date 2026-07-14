from pathlib import Path
import shutil
import subprocess

from app.pipeline import media_duration, normalize_recording


def test_browser_webm_is_normalized_to_mp4(tmp_path: Path):
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    source = tmp_path / "recording.webm"
    output = tmp_path / "recording.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
            "-c:v", "libvpx-vp9", "-c:a", "libopus", str(source),
        ],
        check=True, capture_output=True,
    )
    normalize_recording(source, output, ffmpeg, ffprobe)
    assert output.is_file()
    assert media_duration(output, ffprobe) > 0.9

