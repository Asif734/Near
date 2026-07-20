"""Connects the local video engine to FastAPI project records."""
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Callable

from .languages import LANGUAGES

LANGUAGE_BY_CODE = {item["code"]: item for item in LANGUAGES}
ENGINE_BASELINE: dict[str, object] = {}


def configure_engine(engine, input_type: str) -> None:
    names = (
        "TRANSLATION_CONCURRENCY", "TTS_CONCURRENCY", "AUDIO_FIT_CONCURRENCY",
        "REWRITE_CONCURRENCY", "TRANSCRIPT_QA_ENABLED", "TRANSLATION_QA_ENABLED",
    )
    if not ENGINE_BASELINE:
        ENGINE_BASELINE.update({name: getattr(engine, name) for name in names})
    for name, value in ENGINE_BASELINE.items():
        setattr(engine, name, value)
    if input_type != "recording" or os.getenv("VIDEO_RECORDING_FAST_MODE", "true").lower() not in {"1", "true", "yes", "on"}:
        return
    engine.TRANSLATION_CONCURRENCY = int(os.getenv("VIDEO_RECORDING_TRANSLATION_CONCURRENCY", "10"))
    engine.TTS_CONCURRENCY = int(os.getenv("VIDEO_RECORDING_TTS_CONCURRENCY", "6"))
    engine.AUDIO_FIT_CONCURRENCY = int(os.getenv("VIDEO_RECORDING_AUDIO_FIT_CONCURRENCY", "6"))
    engine.REWRITE_CONCURRENCY = int(os.getenv("VIDEO_RECORDING_REWRITE_CONCURRENCY", "4"))
    # Recordings use the same transcript and translation QA stages as uploads.
    # Fast mode only increases concurrency for the expensive processing stages.


@dataclass
class LanguageAdapter:
    name: str
    lang_code: str
    lang_model: str


class FileAdapter:
    def __init__(self, path: str | Path):
        self.path = str(path)

    def __bool__(self) -> bool:
        return bool(self.path)


class ProjectAdapter:
    def __init__(self, project: dict):
        source = LANGUAGE_BY_CODE[project["source_language"]]
        target = LANGUAGE_BY_CODE[project["target_language"]]
        self.id = project["id"]
        self.input_video = FileAdapter(project["input_path"])
        self.output_video = None
        self.subtitle = None
        self.segments = None
        self.original_duration = 0.0
        self.tempo_factor = 1.0
        self.source_language = LanguageAdapter(source["name"], source["code"], source["voice"])
        gender = project.get("voice_gender", "female")
        selected_voice = target["male_voice"] if gender == "male" else target["female_voice"]
        self.deepgram_tts_model = selected_voice.removeprefix("deepgram:") if selected_voice.startswith("deepgram:") else None
        edge_voice = target["female_voice"] if self.deepgram_tts_model else selected_voice
        self.target_language = LanguageAdapter(target["name"], target["code"], edge_voice)

    def save(self, **_) -> None:
        """FastAPI persists project state separately from engine artifacts."""


def load_engine(media_root: Path):
    from django.conf import settings as django_settings
    if not django_settings.configured:
        django_settings.configure(
            SECRET_KEY="fastapi-video-engine",
            MEDIA_ROOT=str(media_root),
            MEDIA_URL="/media/",
            DEEPGRAM_API_KEY=os.getenv("DEEPGRAM_API_KEY"),
            FFMPEG_BINARY=os.getenv("FFMPEG_BINARY"),
            FFPROBE_BINARY=os.getenv("FFPROBE_BINARY"),
        )
    django_settings.MEDIA_ROOT = str(media_root)
    django_settings.DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    from .video_engine import video_service
    return video_service


def run_engine(project_data: dict, media_root: Path, report: Callable[[str, int], None]) -> Path:
    media_root.mkdir(parents=True, exist_ok=True)
    engine = load_engine(media_root)
    configure_engine(engine, project_data.get("input_type", "upload"))
    project = ProjectAdapter(project_data)
    engine.DEEPGRAM_TTS_MODEL = project.deepgram_tts_model

    report("Extracting audio", 7)
    audio_path = engine.get_audio(project)
    report("Deepgram transcription, recovery and segmentation", 18)
    segments_path = engine.get_segments_translation(project, audio_path)
    report("Generating and timing translated speech", 52)
    asyncio.run(engine.generate_audio_synced(project, segments_path))
    report("Mixing segment audio timeline", 74)
    engine.mix_final_audio(project, segments_path)
    report("Generating SRT and ASS subtitles", 84)
    engine.create_subtitle(project, segments_path)
    report("Burning subtitles and translated audio", 92)
    output = Path(engine.burn_subtitle(project, segments_path))

    deepgram_source = media_root / f"{project.id}_deepgram_output.json"
    segmentation_source = Path(segments_path)
    if deepgram_source.is_file():
        shutil.copyfile(deepgram_source, media_root / "deepgram.json")
    if segmentation_source.is_file():
        shutil.copyfile(segmentation_source, media_root / "segmentation.json")

    report("Cleaning intermediate speech files", 97)
    engine.cleanup_intermediate_files(project.id)

    manifest = {
        "engine": "app.video_engine.video_service",
        "deepgram": "deepgram.json",
        "segmentation": "segmentation.json",
        "output": "output.mp4",
    }
    (media_root / "artifacts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    final_output = media_root / "output.mp4"
    if output != final_output:
        shutil.copyfile(output, final_output)
    return final_output
