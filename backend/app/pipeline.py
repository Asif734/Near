"""Production translation pipeline boundary and browser-recording normalization."""
from pathlib import Path
import os
import shutil
import subprocess

from .engine_adapter import run_engine


class PipelineError(RuntimeError):
    pass


def _run(command: list[str], message: str) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip()[-1600:] or "No FFmpeg diagnostic was returned"
        raise PipelineError(f"{message}: {detail}")
    return result


def media_duration(path: Path, ffprobe: str) -> float:
    result = _run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        "Unable to inspect video",
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise PipelineError(
            "FFprobe found no readable duration. The browser recording may be incomplete or corrupted."
        ) from exc
    if duration <= 0:
        raise PipelineError("The video has no playable duration")
    return duration


def normalize_recording(source: Path, output: Path, ffmpeg: str, ffprobe: str) -> Path:
    """Finalize a MediaRecorder WebM/MP4 for the translation engine."""
    _run(
        [
            ffmpeg, "-y", "-fflags", "+genpts", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero", str(output),
        ],
        "Unable to finalize browser recording",
    )
    media_duration(output, ffprobe)
    return output


def process_video(project: dict, report) -> Path:
    source = Path(project["input_path"])
    output_dir = source.parent.parent / "outputs" / project["id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    report("Validating video engine", 3)
    ffmpeg = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
    ffprobe = os.getenv("FFPROBE_BINARY") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise PipelineError("FFmpeg and FFprobe are required but were not found")

    engine_project = dict(project)
    if project.get("input_type") == "recording":
        report("Finalizing browser recording", 4)
        normalized = normalize_recording(source, output_dir / "recording.mp4", ffmpeg, ffprobe)
        engine_project["input_path"] = str(normalized)
    else:
        media_duration(source, ffprobe)

    output = run_engine(engine_project, output_dir, report)
    if not output.is_file() or output.stat().st_size == 0:
        raise PipelineError("Pipeline produced no output")
    return output

