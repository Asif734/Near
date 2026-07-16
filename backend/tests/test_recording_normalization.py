from pathlib import Path
import shutil
import subprocess

from app import pipeline
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


def test_uploaded_webcam_webm_is_normalized_before_engine(tmp_path: Path, monkeypatch):
    source = tmp_path / "inputs" / "webcam.webm"
    source.parent.mkdir()
    source.write_bytes(b"webm")
    normalized_paths = []

    def fake_normalize(input_path, output_path, _ffmpeg, _ffprobe):
        assert input_path == source
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        normalized_paths.append(output_path)
        return output_path

    def fake_engine(project, output_dir, _report):
        assert Path(project["input_path"]) == normalized_paths[0]
        output = output_dir / "output.mp4"
        output.write_bytes(b"translated")
        return output

    monkeypatch.setattr(pipeline, "normalize_recording", fake_normalize)
    monkeypatch.setattr(pipeline, "run_engine", fake_engine)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: name)

    output = pipeline.process_video(
        {"id": "project-id", "input_path": str(source), "input_type": "upload"},
        lambda *_: None,
    )

    assert output.read_bytes() == b"translated"
    assert normalized_paths[0].name == "recording.mp4"
