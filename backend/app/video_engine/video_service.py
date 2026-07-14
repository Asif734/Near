import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests
from deep_translator import GoogleTranslator
from django.conf import settings
from edge_tts import Communicate
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_silence

from .sync_translation import rewrite_translation_for_timing
from .translation_quality import apply_source_transcript_review, apply_translation_review

logger = logging.getLogger(__name__)

FFMPEG_PATH = getattr(settings, "FFMPEG_BINARY", None) or os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
FFPROBE_PATH = getattr(settings, "FFPROBE_BINARY", None) or os.environ.get("FFPROBE_BINARY") or shutil.which("ffprobe")
CURL_PATH = shutil.which("curl")

if not FFMPEG_PATH or not FFPROBE_PATH:
    missing = ", ".join(name for name, path in (("ffmpeg", FFMPEG_PATH), ("ffprobe", FFPROBE_PATH)) if not path)
    raise RuntimeError(
        f"Missing required binary: {missing}. Install FFmpeg and ensure both ffmpeg and ffprobe are available in PATH."
    )


def _runtime_setting(name: str, default: Any) -> Any:
    return os.environ.get(name, getattr(settings, name, default))


MIN_SENTENCE_GAP_SEC = float(_runtime_setting("VIDEO_MIN_SENTENCE_GAP_SEC", 0.7))
MERGE_GAP_THRESHOLD_SEC = float(_runtime_setting("VIDEO_MERGE_GAP_THRESHOLD_SEC", 0.45))
MIN_SEGMENT_WORDS = int(_runtime_setting("VIDEO_MIN_SEGMENT_WORDS", 5))
MAX_SEGMENT_SECONDS = float(_runtime_setting("VIDEO_MAX_SEGMENT_SECONDS", 12.0))
MAX_CONTIGUOUS_SEGMENT_SECONDS = float(_runtime_setting("VIDEO_MAX_CONTIGUOUS_SEGMENT_SECONDS", 15.0))
CONTIGUOUS_GAP_TOLERANCE_SEC = float(_runtime_setting("VIDEO_CONTIGUOUS_GAP_TOLERANCE_SEC", 0.05))
MIN_SILENCE_MS = int(_runtime_setting("VIDEO_MIN_SILENCE_MS", 700))
DEEPGRAM_LONG_WORD_ANOMALY_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_LONG_WORD_ANOMALY_SEC", 1.5))
DEEPGRAM_MAX_SANITIZED_WORD_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_MAX_SANITIZED_WORD_SEC", 0.85))
DEEPGRAM_MIN_SANITIZED_WORD_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_MIN_SANITIZED_WORD_SEC", 0.05))

TRANSLATION_CONCURRENCY = int(_runtime_setting("VIDEO_TRANSLATION_CONCURRENCY", 5))
TTS_CONCURRENCY = int(_runtime_setting("VIDEO_TTS_CONCURRENCY", 2))
AUDIO_FIT_CONCURRENCY = int(_runtime_setting("VIDEO_AUDIO_FIT_CONCURRENCY", 3))
TRANSLATION_MAX_BYTES = int(_runtime_setting("VIDEO_TRANSLATION_MAX_BYTES", 450))
TRANSLATION_QA_ENABLED = str(_runtime_setting("VIDEO_TRANSLATION_QA_ENABLED", "true")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRANSLATION_QA_BATCH_SIZE = int(_runtime_setting("VIDEO_TRANSLATION_QA_BATCH_SIZE", 8))
TRANSCRIPT_QA_ENABLED = str(_runtime_setting("VIDEO_TRANSCRIPT_QA_ENABLED", "true")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRANSCRIPT_QA_BATCH_SIZE = int(_runtime_setting("VIDEO_TRANSCRIPT_QA_BATCH_SIZE", 8))
STRETCH_TOLERANCE_MS = int(_runtime_setting("VIDEO_STRETCH_TOLERANCE_MS", 80))
MIN_ATEMPO_RATIO = float(_runtime_setting("VIDEO_MIN_ATEMPO_RATIO", 0.5))
MAX_ATEMPO_RATIO = float(_runtime_setting("VIDEO_MAX_ATEMPO_RATIO", 2.0))
EXTREME_ATEMPO_RATIO = float(_runtime_setting("VIDEO_EXTREME_ATEMPO_RATIO", 1.8))
SHORT_AUDIO_STRETCH_THRESHOLD = float(_runtime_setting("VIDEO_SHORT_AUDIO_STRETCH_THRESHOLD", 0.85))
SHORT_AUDIO_DELAY_MS = int(_runtime_setting("VIDEO_SHORT_AUDIO_DELAY_MS", 500))
MAX_NATURAL_SPEEDUP_RATIO = float(_runtime_setting("VIDEO_MAX_NATURAL_SPEEDUP_RATIO", 1.10))
MIN_NATURAL_SLOWDOWN_RATIO = float(_runtime_setting("VIDEO_MIN_NATURAL_SLOWDOWN_RATIO", 0.92))
MIN_AUDIO_SEGMENT_GAP_MS = int(_runtime_setting("VIDEO_MIN_AUDIO_SEGMENT_GAP_MS", 180))
SUBTITLE_MAX_CJK_CHARS_PER_LINE = int(_runtime_setting("VIDEO_SUBTITLE_MAX_CJK_CHARS_PER_LINE", 24))
SUBTITLE_MAX_WORDS_PER_LINE = int(_runtime_setting("VIDEO_SUBTITLE_MAX_WORDS_PER_LINE", 9))
SUBTITLE_VIDEO_CODEC = _runtime_setting("VIDEO_SUBTITLE_VIDEO_CODEC", "auto")
SUBTITLE_VIDEO_PRESET = _runtime_setting("VIDEO_SUBTITLE_VIDEO_PRESET", "ultrafast")
SUBTITLE_VIDEO_CRF = str(_runtime_setting("VIDEO_SUBTITLE_VIDEO_CRF", 23))
SUBTITLE_VIDEO_BITRATE = str(_runtime_setting("VIDEO_SUBTITLE_VIDEO_BITRATE", "5000k"))
TTS_RETRY_ATTEMPTS = int(_runtime_setting("VIDEO_TTS_RETRY_ATTEMPTS", 3))
TTS_RETRY_BASE_DELAY_SEC = float(_runtime_setting("VIDEO_TTS_RETRY_BASE_DELAY_SEC", 0.8))
DEEPGRAM_TTS_MODEL = None
REWRITE_GAP_THRESHOLD_MS = int(_runtime_setting("VIDEO_REWRITE_GAP_THRESHOLD_MS", 3000))
REWRITE_MAX_ATTEMPTS = int(_runtime_setting("VIDEO_REWRITE_MAX_ATTEMPTS", 3))
HARD_FIT_REWRITE_MAX_ATTEMPTS = int(_runtime_setting("VIDEO_HARD_FIT_REWRITE_MAX_ATTEMPTS", 5))
REWRITE_CONCURRENCY = int(_runtime_setting("VIDEO_REWRITE_CONCURRENCY", 2))
DEEPGRAM_TAIL_RETRY_GAP_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_TAIL_RETRY_GAP_SEC", 3.0))
DEEPGRAM_TAIL_RETRY_PREROLL_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_TAIL_RETRY_PREROLL_SEC", 1.0))
DEEPGRAM_RETRY_TIMEOUT_SEC = max(0.1, float(_runtime_setting("VIDEO_DEEPGRAM_RETRY_TIMEOUT_SEC", 30.0)))
DEEPGRAM_UPLOAD_ATTEMPTS = max(1, int(_runtime_setting("VIDEO_DEEPGRAM_UPLOAD_ATTEMPTS", 3)))
DEEPGRAM_UPLOAD_RETRY_BASE_SEC = max(0.0, float(_runtime_setting("VIDEO_DEEPGRAM_UPLOAD_RETRY_BASE_SEC", 2.0)))
DEEPGRAM_COMPRESS_UPLOAD_MIN_BYTES = int(_runtime_setting("VIDEO_DEEPGRAM_COMPRESS_UPLOAD_MIN_BYTES", 262144))
DEEPGRAM_UPLOAD_MP3_BITRATE = str(_runtime_setting("VIDEO_DEEPGRAM_UPLOAD_MP3_BITRATE", "32k"))
DEEPGRAM_TAIL_OVERLAP_TOLERANCE_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_TAIL_OVERLAP_TOLERANCE_SEC", 0.05))
DEEPGRAM_TAIL_RETRY_FILTER = _runtime_setting(
    "VIDEO_DEEPGRAM_TAIL_RETRY_FILTER",
    "highpass=f=120,lowpass=f=3800,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
)
DEEPGRAM_GAP_RETRY_GAP_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_GAP_RETRY_GAP_SEC", 3.0))
DEEPGRAM_GAP_RETRY_PREROLL_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_GAP_RETRY_PREROLL_SEC", 0.5))
DEEPGRAM_GAP_RETRY_POSTROLL_SEC = float(_runtime_setting("VIDEO_DEEPGRAM_GAP_RETRY_POSTROLL_SEC", 0.5))
DEEPGRAM_GAP_RETRY_MAX_GAPS = int(_runtime_setting("VIDEO_DEEPGRAM_GAP_RETRY_MAX_GAPS", 1))
DEEPGRAM_GAP_RETRY_SILENCE_RATIO_MAX = float(_runtime_setting("VIDEO_DEEPGRAM_GAP_RETRY_SILENCE_RATIO_MAX", 0.7))
DEEPGRAM_GAP_RETRY_SILENCE_THRESH_DBFS = float(_runtime_setting("VIDEO_DEEPGRAM_GAP_RETRY_SILENCE_THRESH_DBFS", -45))
DEEPGRAM_KEYTERMS = [
    item.strip()
    for item in str(_runtime_setting("VIDEO_DEEPGRAM_KEYTERMS", "")).split(",")
    if item.strip()
]
NON_SPEECH_ARTIFACT_TERMS = {
    item.strip().lower()
    for item in str(
        _runtime_setting(
            "VIDEO_NON_SPEECH_ARTIFACT_TERMS",
            "মিউজিক,music,সঙ্গীত,applause,clapping,noise,[music],(music)",
        )
    ).split(",")
    if item.strip()
}
NON_SPEECH_ARTIFACT_CONFIDENCE_MAX = float(_runtime_setting("VIDEO_NON_SPEECH_ARTIFACT_CONFIDENCE_MAX", 0.45))
NON_SPEECH_ARTIFACT_MAX_SECONDS = float(_runtime_setting("VIDEO_NON_SPEECH_ARTIFACT_MAX_SECONDS", 1.5))
NON_SPEECH_ARTIFACT_MIN_ISOLATION_GAP_SEC = float(
    _runtime_setting("VIDEO_NON_SPEECH_ARTIFACT_MIN_ISOLATION_GAP_SEC", 0.7)
)

_DURATION_CACHE: dict[str, float] = {}
_FFMPEG_FILTERS_CACHE: set[str] | None = None
_FFMPEG_ENCODERS_CACHE: set[str] | None = None

PUNCTUATION_NO_SPACE_BEFORE = r"([,.?!:;।])"
SENTENCE_END_RE = re.compile(r"[.?!।]$")
PHRASE_END_RE = re.compile(r"[,;:]$")


class Lines:
    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        trans: str = "",
        actual_audio_size: float = 0,
        audio_offset_ms: int = 0,
        natural_audio_ms: float = 0,
        fitted_audio_ms: float = 0,
        fit_mode: str = "",
        audio_gap_before_ms: int = 0,
        scheduled_start_ms: int = 0,
        scheduled_end_ms: int = 0,
        **extra,
    ):
        self.start = start
        self.end = end
        self.text = text
        self.trans = trans
        self.actual_audio_size = actual_audio_size
        self.audio_offset_ms = audio_offset_ms
        self.natural_audio_ms = natural_audio_ms
        self.fitted_audio_ms = fitted_audio_ms
        self.fit_mode = fit_mode
        self.audio_gap_before_ms = audio_gap_before_ms
        self.scheduled_start_ms = scheduled_start_ms
        self.scheduled_end_ms = scheduled_end_ms
        self.extra = extra


@contextmanager
def timed_stage(stage_name: str):
    started = time.perf_counter()
    logger.info("%s started", stage_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        logger.info("%s finished in %.2fs", stage_name, elapsed)


def _run_subprocess(cmd: list[str], error_message: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{error_message}:\n{result.stderr.strip()}")
    return result


def _media_path(file_name: str) -> str:
    return os.path.join(settings.MEDIA_ROOT, file_name)


def _relative_media_name(path: str) -> str:
    try:
        return os.path.relpath(path, settings.MEDIA_ROOT)
    except ValueError:
        return os.path.basename(path)


def _normalize_text(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+" + PUNCTUATION_NO_SPACE_BEFORE, r"\1", value)
    value = re.sub(r"([,.?!:;।])(?=\S)", r"\1 ", value)
    return re.sub(r"\s+", " ", value).strip()


def _word_text(word: dict[str, Any]) -> str:
    return word.get("punctuated_word") or word.get("word") or ""


def _is_punctuation_only_word(word: dict[str, Any]) -> bool:
    return bool(re.fullmatch(r"[,.?!:;।]+", _word_text(word).strip()))


def _word_count(text: str) -> int:
    return len([part for part in text.strip().split() if part])


def _word_confidences(words: list[dict[str, Any]]) -> list[float]:
    confidences = []
    for word in words:
        value = word.get("confidence")
        if isinstance(value, (int, float)):
            confidences.append(float(value))
    return confidences


def _sanitize_deepgram_word_timings(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = [{**word} for word in words]
    adjusted = 0

    for index, word in enumerate(sanitized):
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue

        duration = end - start
        next_word = sanitized[index + 1] if index < len(sanitized) - 1 else None
        next_start = None
        if next_word:
            try:
                next_start = float(next_word["start"])
            except (KeyError, TypeError, ValueError):
                next_start = None

        has_sentence_punctuation = bool(SENTENCE_END_RE.search(_word_text(word)))
        overlaps_next = next_start is not None and next_start < end - CONTIGUOUS_GAP_TOLERANCE_SEC
        if _is_punctuation_only_word(word) and overlaps_next:
            clamped_end = next_start if next_start and next_start > start else start + DEEPGRAM_MIN_SANITIZED_WORD_SEC
            if clamped_end < end:
                word["end"] = round(clamped_end, 6)
                adjusted += 1
            continue

        if has_sentence_punctuation and overlaps_next and next_start and next_start > start + DEEPGRAM_MIN_SANITIZED_WORD_SEC:
            word["end"] = round(next_start, 6)
            adjusted += 1
            continue

        if duration <= DEEPGRAM_LONG_WORD_ANOMALY_SEC:
            continue

        if not has_sentence_punctuation and not overlaps_next:
            continue

        clamped_end = start + DEEPGRAM_MAX_SANITIZED_WORD_SEC
        if next_start is not None:
            clamped_end = min(clamped_end, next_start)
        clamped_end = max(start + DEEPGRAM_MIN_SANITIZED_WORD_SEC, clamped_end)

        if clamped_end < end:
            word["end"] = round(clamped_end, 6)
            adjusted += 1

    if adjusted:
        logger.info("Sanitized %s anomalous Deepgram word timings", adjusted)
    return sanitized


def _segment_from_words(words: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = _word_confidences(words)
    segment = {
        "text": _normalize_text(" ".join(_word_text(word) for word in words)),
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
    }
    if confidences:
        segment["segment_confidence"] = round(sum(confidences) / len(confidences), 4)
        segment["min_word_confidence"] = round(min(confidences), 4)
    return segment


def _artifact_key(text: str) -> str:
    value = _normalize_text(text).lower()
    value = re.sub(r"^[\[\(（{]+|[\]\)）}]+$", "", value)
    value = re.sub(r"[.?!,;:।]+$", "", value)
    return value.strip()


def _is_low_confidence_non_speech_artifact(
    segment: dict[str, Any],
    previous_segment: dict[str, Any] | None,
    next_segment: dict[str, Any] | None,
) -> bool:
    text = _normalize_text(segment.get("text", ""))
    if _artifact_key(text) not in NON_SPEECH_ARTIFACT_TERMS:
        return False
    if _word_count(text) > 3:
        return False

    confidence = segment.get("segment_confidence", segment.get("min_word_confidence", 1.0))
    if not isinstance(confidence, (int, float)) or float(confidence) > NON_SPEECH_ARTIFACT_CONFIDENCE_MAX:
        return False

    duration = float(segment["end"]) - float(segment["start"])
    if duration > NON_SPEECH_ARTIFACT_MAX_SECONDS:
        return False

    previous_gap = (
        float(segment["start"]) - float(previous_segment["end"])
        if previous_segment
        else NON_SPEECH_ARTIFACT_MIN_ISOLATION_GAP_SEC
    )
    next_gap = (
        float(next_segment["start"]) - float(segment["end"])
        if next_segment
        else NON_SPEECH_ARTIFACT_MIN_ISOLATION_GAP_SEC
    )
    return max(previous_gap, next_gap) >= NON_SPEECH_ARTIFACT_MIN_ISOLATION_GAP_SEC


def _drop_non_speech_artifacts(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []

    filtered = []
    dropped = 0
    for index, segment in enumerate(segments):
        previous_segment = segments[index - 1] if index > 0 else None
        next_segment = segments[index + 1] if index < len(segments) - 1 else None
        if _is_low_confidence_non_speech_artifact(segment, previous_segment, next_segment):
            dropped += 1
            logger.info(
                "Dropping low-confidence non-speech artifact segment %.3f-%.3f: %s",
                segment["start"],
                segment["end"],
                segment.get("text", ""),
            )
            continue
        filtered.append(segment)

    if dropped:
        logger.info("Dropped %s low-confidence non-speech artifact segments", dropped)
    return filtered


def _best_forced_split_index(words: list[dict[str, Any]]) -> int:
    if len(words) <= 1:
        return len(words) - 1

    split_candidates = range(MIN_SEGMENT_WORDS - 1, len(words) - 1)

    sentence_candidates = [
        index
        for index in range(0, len(words) - 1)
        if SENTENCE_END_RE.search(_word_text(words[index]))
        and len(words) - index - 1 >= MIN_SEGMENT_WORDS
    ]
    if sentence_candidates:
        return sentence_candidates[-1]

    phrase_candidates = [
        index
        for index in split_candidates
        if PHRASE_END_RE.search(_word_text(words[index]))
        and len(words) - index - 1 >= MIN_SEGMENT_WORDS
    ]
    if phrase_candidates:
        return phrase_candidates[-1]

    return len(words) - 1


def _duration_seconds(path: str) -> float:
    abs_path = os.path.abspath(path)
    if abs_path in _DURATION_CACHE:
        return _DURATION_CACHE[abs_path]

    result = _run_subprocess(
        [
            FFPROBE_PATH,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            abs_path,
        ],
        f"Unable to read media duration for {abs_path}",
    )
    duration = float(result.stdout.strip())
    _DURATION_CACHE[abs_path] = duration
    return duration


def _duration_ms(path: str) -> float:
    return _duration_seconds(path) * 1000


def _ffmpeg_filter_exists(filter_name: str) -> bool:
    global _FFMPEG_FILTERS_CACHE
    if _FFMPEG_FILTERS_CACHE is None:
        result = _run_subprocess([FFMPEG_PATH, "-hide_banner", "-filters"], "Unable to inspect FFmpeg filters")
        _FFMPEG_FILTERS_CACHE = {
            line.split()[1]
            for line in result.stdout.splitlines()
            if line.startswith(" ") and len(line.split()) >= 2
        }
    return filter_name in _FFMPEG_FILTERS_CACHE


def _ffmpeg_encoder_exists(encoder_name: str) -> bool:
    global _FFMPEG_ENCODERS_CACHE
    if _FFMPEG_ENCODERS_CACHE is None:
        result = _run_subprocess([FFMPEG_PATH, "-hide_banner", "-encoders"], "Unable to inspect FFmpeg encoders")
        _FFMPEG_ENCODERS_CACHE = {
            line.split()[1]
            for line in result.stdout.splitlines()
            if line.startswith(" ") and len(line.split()) >= 2
        }
    return encoder_name in _FFMPEG_ENCODERS_CACHE


def get_audio(project) -> str:
    file_name = f"{project.id}_extracted.wav"
    file_path = _media_path(file_name)

    with timed_stage("audio extraction"):
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i",
            project.input_video.path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            file_path,
        ]
        _run_subprocess(cmd, "FFmpeg audio extraction failed")
        project.original_duration = _duration_seconds(project.input_video.path)
        project.save(update_fields=["original_duration"])
        logger.info("Extracted audio for project %s to %s", project.id, file_path)

    return file_path


def _extract_deepgram_words(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("words", [])


def _extract_deepgram_transcript(data: dict[str, Any]) -> str:
    return data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")


def _deepgram_request_params(lang_temp: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": "nova-3",
        "language": lang_temp,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
    }
    if DEEPGRAM_KEYTERMS:
        params["keyterm"] = DEEPGRAM_KEYTERMS
    return params


def _post_deepgram_audio(
    api_key: str,
    audio_path: str,
    lang_temp: str,
    timeout_sec: float = 300,
) -> dict[str, Any]:
    upload_path = audio_path
    content_type = "audio/wav"
    temporary_upload = None
    if os.path.getsize(audio_path) >= DEEPGRAM_COMPRESS_UPLOAD_MIN_BYTES:
        temporary_upload = f"{os.path.splitext(audio_path)[0]}_deepgram_upload.mp3"
        _run_subprocess(
            [
                FFMPEG_PATH, "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "16000",
                "-codec:a", "libmp3lame", "-b:a", DEEPGRAM_UPLOAD_MP3_BITRATE, temporary_upload,
            ],
            "FFmpeg Deepgram upload compression failed",
        )
        upload_path = temporary_upload
        content_type = "audio/mpeg"
        logger.info(
            "Compressed Deepgram upload from %.2fMB WAV to %.2fMB MP3",
            os.path.getsize(audio_path) / (1024 * 1024),
            os.path.getsize(upload_path) / (1024 * 1024),
        )

    with open(upload_path, "rb") as audio:
        payload = audio.read()
    if temporary_upload:
        try:
            os.remove(temporary_upload)
        except OSError:
            logger.warning("Unable to remove temporary Deepgram upload %s", temporary_upload)
    if not payload:
        raise RuntimeError("Deepgram upload audio is empty")
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": content_type,
        "Content-Length": str(len(payload)),
    }

    last_error = None
    for attempt in range(1, DEEPGRAM_UPLOAD_ATTEMPTS + 1):
        try:
            response = requests.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=_deepgram_request_params(lang_temp),
                data=payload,
                timeout=timeout_sec,
            )
            logger.info(
                "Deepgram status for %s (attempt %s/%s): %s",
                os.path.basename(audio_path), attempt, DEEPGRAM_UPLOAD_ATTEMPTS, response.status_code,
            )
            if response.ok:
                return response.json()
            last_error = RuntimeError(
                f"Deepgram request failed ({response.status_code}): {response.text[:500]}"
            )
            retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
            if not retryable or attempt == DEEPGRAM_UPLOAD_ATTEMPTS:
                raise last_error
        except requests.RequestException as exc:
            last_error = exc
            if attempt == DEEPGRAM_UPLOAD_ATTEMPTS:
                raise RuntimeError(f"Deepgram network request failed: {exc}") from exc

        delay = DEEPGRAM_UPLOAD_RETRY_BASE_SEC * (2 ** (attempt - 1))
        logger.warning(
            "Retrying Deepgram upload for %s in %.1fs after attempt %s/%s: %s",
            os.path.basename(audio_path), delay, attempt, DEEPGRAM_UPLOAD_ATTEMPTS, last_error,
        )
        if delay:
            time.sleep(delay)

    raise RuntimeError(f"Deepgram request failed after retries: {last_error}")


def _post_deepgram_retry(api_key: str, audio_path: str, lang_temp: str) -> dict[str, Any] | None:
    if not CURL_PATH:
        logger.warning("Skipping Deepgram retry because curl is not installed")
        return None

    query = urlencode(_deepgram_request_params(lang_temp), doseq=True)
    url = f"https://api.deepgram.com/v1/listen?{query}"

    def curl_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    # Pass the token through stdin instead of exposing it in the process command line.
    curl_config = "\n".join(
        [
            "silent",
            "show-error",
            f'max-time = "{DEEPGRAM_RETRY_TIMEOUT_SEC}"',
            f'connect-timeout = "{min(10.0, DEEPGRAM_RETRY_TIMEOUT_SEC)}"',
            'request = "POST"',
            f'header = "Authorization: Token {curl_value(api_key)}"',
            'header = "Content-Type: audio/wav"',
            f'data-binary = "@{curl_value(os.path.abspath(audio_path))}"',
            f'url = "{curl_value(url)}"',
            'write-out = "\\n%{http_code}"',
        ]
    )

    try:
        result = subprocess.run(
            [CURL_PATH, "--config", "-"],
            input=curl_config,
            capture_output=True,
            text=True,
            timeout=DEEPGRAM_RETRY_TIMEOUT_SEC + 2,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Deepgram retry segment %s exceeded the %.1fs hard deadline; skipping",
            os.path.basename(audio_path),
            DEEPGRAM_RETRY_TIMEOUT_SEC,
        )
        return None
    except OSError as exc:
        logger.warning(
            "Unable to run Deepgram retry segment %s: %s",
            os.path.basename(audio_path),
            exc,
        )
        return None

    if result.returncode != 0:
        logger.warning(
            "Deepgram retry segment %s failed: %s",
            os.path.basename(audio_path),
            result.stderr.strip()[:500],
        )
        return None

    try:
        response_body, status_text = result.stdout.rsplit("\n", 1)
        status_code = int(status_text.strip())
    except (ValueError, TypeError):
        logger.warning("Deepgram retry segment %s returned an invalid response", os.path.basename(audio_path))
        return None

    logger.info("Deepgram retry status for %s: %s", os.path.basename(audio_path), status_code)
    if not 200 <= status_code < 300:
        logger.warning(
            "Deepgram retry segment %s failed (%s): %s",
            os.path.basename(audio_path),
            status_code,
            response_body[:500],
        )
        return None

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        logger.warning("Deepgram retry segment %s returned invalid JSON: %s", os.path.basename(audio_path), exc)
        return None


def _offset_deepgram_timestamps(value: Any, offset_sec: float) -> None:
    if isinstance(value, dict):
        for key in ("start", "end"):
            if key in value and isinstance(value[key], (int, float)):
                value[key] = float(value[key]) + offset_sec
        for child in value.values():
            _offset_deepgram_timestamps(child, offset_sec)
    elif isinstance(value, list):
        for child in value:
            _offset_deepgram_timestamps(child, offset_sec)


def _merge_deepgram_tail_retry(data: dict[str, Any], tail_data: dict[str, Any], tail_offset_sec: float) -> int:
    existing_words = _extract_deepgram_words(data)
    tail_words = _extract_deepgram_words(tail_data)
    if not existing_words or not tail_words:
        return 0

    last_word_end = float(existing_words[-1]["end"])
    recovered_words = []
    for word in tail_words:
        shifted = dict(word)
        shifted["start"] = float(shifted["start"]) + tail_offset_sec
        shifted["end"] = float(shifted["end"]) + tail_offset_sec
        if float(shifted["start"]) >= last_word_end - DEEPGRAM_TAIL_OVERLAP_TOLERANCE_SEC:
            recovered_words.append(shifted)

    if not recovered_words:
        return 0

    existing_words.extend(recovered_words)
    alternative = data["results"]["channels"][0]["alternatives"][0]
    alternative["transcript"] = _normalize_text(" ".join(_word_text(word) for word in existing_words))

    tail_utterances = tail_data.get("results", {}).get("utterances", [])
    if tail_utterances:
        shifted_utterances = []
        for utterance in tail_utterances:
            shifted = json.loads(json.dumps(utterance))
            _offset_deepgram_timestamps(shifted, tail_offset_sec)
            if float(shifted.get("start", 0)) >= last_word_end - DEEPGRAM_TAIL_OVERLAP_TOLERANCE_SEC:
                shifted_utterances.append(shifted)
        if shifted_utterances:
            data.setdefault("results", {}).setdefault("utterances", []).extend(shifted_utterances)

    tail_retry = data.setdefault("metadata", {}).setdefault("tail_retry", {"recovered_words": 0, "chunks": []})
    tail_retry["tail_offset_sec"] = round(tail_offset_sec, 3)
    tail_retry["recovered_words"] = int(tail_retry.get("recovered_words", 0)) + len(recovered_words)
    tail_retry.setdefault("chunks", []).append(
        {"tail_offset_sec": round(tail_offset_sec, 3), "recovered_words": len(recovered_words)}
    )
    return len(recovered_words)


def _same_deepgram_word(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _word_text(left).strip().lower() == _word_text(right).strip().lower()


def _word_time_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start = float(left.get("start", 0))
    left_end = float(left.get("end", left_start))
    right_start = float(right.get("start", 0))
    right_end = float(right.get("end", right_start))
    overlap = min(left_end, right_end) - max(left_start, right_start)
    if overlap <= 0:
        return 0.0
    shortest = max(min(left_end - left_start, right_end - right_start), DEEPGRAM_MIN_SANITIZED_WORD_SEC)
    return overlap / shortest


def _merge_deepgram_gap_retry(
    data: dict[str, Any],
    gap_data: dict[str, Any],
    gap_offset_sec: float,
    gap_start_sec: float,
    gap_end_sec: float,
) -> int:
    existing_words = _extract_deepgram_words(data)
    gap_words = _extract_deepgram_words(gap_data)
    if not existing_words or not gap_words:
        return 0

    lower_bound = gap_start_sec - DEEPGRAM_TAIL_OVERLAP_TOLERANCE_SEC
    upper_bound = gap_end_sec + DEEPGRAM_TAIL_OVERLAP_TOLERANCE_SEC
    recovered_words = []
    for word in gap_words:
        shifted = dict(word)
        shifted["start"] = float(shifted["start"]) + gap_offset_sec
        shifted["end"] = float(shifted["end"]) + gap_offset_sec
        if not (lower_bound <= float(shifted["start"]) and float(shifted["end"]) <= upper_bound):
            continue
        if any(_same_deepgram_word(shifted, current) and _word_time_overlap(shifted, current) >= 0.5 for current in existing_words):
            continue
        recovered_words.append(shifted)

    if not recovered_words:
        return 0

    existing_words.extend(recovered_words)
    existing_words.sort(key=lambda word: (float(word.get("start", 0)), float(word.get("end", 0))))
    alternative = data["results"]["channels"][0]["alternatives"][0]
    alternative["transcript"] = _normalize_text(" ".join(_word_text(word) for word in existing_words))

    gap_utterances = gap_data.get("results", {}).get("utterances", [])
    if gap_utterances:
        shifted_utterances = []
        for utterance in gap_utterances:
            shifted = json.loads(json.dumps(utterance))
            _offset_deepgram_timestamps(shifted, gap_offset_sec)
            if float(shifted.get("end", 0)) >= lower_bound and float(shifted.get("start", 0)) <= upper_bound:
                shifted_utterances.append(shifted)
        if shifted_utterances:
            utterances = data.setdefault("results", {}).setdefault("utterances", [])
            utterances.extend(shifted_utterances)
            utterances.sort(key=lambda utterance: (float(utterance.get("start", 0)), float(utterance.get("end", 0))))

    gap_retries = data.setdefault("metadata", {}).setdefault("gap_retries", [])
    gap_retries.append(
        {
            "gap_start_sec": round(gap_start_sec, 3),
            "gap_end_sec": round(gap_end_sec, 3),
            "retry_offset_sec": round(gap_offset_sec, 3),
            "recovered_words": len(recovered_words),
        }
    )
    return len(recovered_words)


def _audio_window_silence_ratio(audio_path: str, start_sec: float, end_sec: float) -> float:
    if end_sec <= start_sec:
        return 1.0

    audio = AudioSegment.from_file(audio_path)
    window = audio[int(start_sec * 1000) : int(end_sec * 1000)]
    if len(window) <= 0:
        return 1.0

    silences = detect_silence(
        window,
        min_silence_len=min(MIN_SILENCE_MS, max(100, len(window) // 2)),
        silence_thresh=DEEPGRAM_GAP_RETRY_SILENCE_THRESH_DBFS,
    )
    silent_ms = sum(end_ms - start_ms for start_ms, end_ms in silences)
    return silent_ms / len(window)


def _retry_untranscribed_gaps(data: dict[str, Any], project, audio_path: str, api_key: str, lang_temp: str) -> dict[str, Any]:
    words = _extract_deepgram_words(data)
    if len(words) < 2:
        return data

    audio_duration = _duration_seconds(audio_path)
    retried = 0
    for previous_word, next_word in zip(list(words), list(words)[1:]):
        gap_start = float(previous_word["end"])
        gap_end = float(next_word["start"])
        gap_size = gap_end - gap_start
        if gap_size < DEEPGRAM_GAP_RETRY_GAP_SEC:
            continue

        retry_start = max(0.0, gap_start - DEEPGRAM_GAP_RETRY_PREROLL_SEC)
        retry_end = min(audio_duration, gap_end + DEEPGRAM_GAP_RETRY_POSTROLL_SEC)
        silence_ratio = _audio_window_silence_ratio(audio_path, gap_start, gap_end)
        if silence_ratio > DEEPGRAM_GAP_RETRY_SILENCE_RATIO_MAX:
            logger.info(
                "Skipping Deepgram gap retry for project %s at %.2f-%.2fs because silence ratio is %.2f",
                project.id,
                gap_start,
                gap_end,
                silence_ratio,
            )
            continue

        gap_path = _media_path(f"{project.id}_deepgram_gap_retry_{retried + 1}.wav")
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-ss",
            f"{retry_start:.3f}",
            "-t",
            f"{retry_end - retry_start:.3f}",
            "-i",
            audio_path,
            "-vn",
            "-af",
            DEEPGRAM_TAIL_RETRY_FILTER,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            gap_path,
        ]
        _run_subprocess(cmd, "FFmpeg Deepgram gap retry extraction failed")
        logger.warning(
            "Deepgram left %.2fs untranscribed inside project %s at %.2f-%.2fs; retrying cleaned gap",
            gap_size,
            project.id,
            gap_start,
            gap_end,
        )

        gap_data = _post_deepgram_retry(api_key, gap_path, lang_temp)
        if gap_data is None:
            retried += 1
            if retried >= DEEPGRAM_GAP_RETRY_MAX_GAPS:
                break
            continue
        recovered_count = _merge_deepgram_gap_retry(data, gap_data, retry_start, gap_start, gap_end)
        if recovered_count:
            logger.info("Recovered %s Deepgram gap words for project %s", recovered_count, project.id)
        else:
            logger.warning("Deepgram gap retry recovered no words for project %s. Inspect %s.", project.id, gap_path)

        retried += 1
        if retried >= DEEPGRAM_GAP_RETRY_MAX_GAPS:
            break

    return data


def _retry_untranscribed_tail(data: dict[str, Any], project, audio_path: str, api_key: str, lang_temp: str) -> dict[str, Any]:
    words = _extract_deepgram_words(data)
    if not words:
        return data

    audio_duration = _duration_seconds(audio_path)
    last_word_end = float(words[-1]["end"])
    untranscribed_tail = audio_duration - last_word_end
    if untranscribed_tail < DEEPGRAM_TAIL_RETRY_GAP_SEC:
        return data

    retry_start = max(0.0, last_word_end - DEEPGRAM_TAIL_RETRY_PREROLL_SEC)
    tail_path = _media_path(f"{project.id}_deepgram_tail_retry.wav")
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss",
        f"{retry_start:.3f}",
        "-i",
        audio_path,
        "-vn",
        "-af",
        DEEPGRAM_TAIL_RETRY_FILTER,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        tail_path,
    ]
    _run_subprocess(cmd, "FFmpeg Deepgram tail retry extraction failed")
    logger.warning(
        "Retrying Deepgram tail once for project %s at %.2f-%.2fs (hard timeout %.1fs)",
        project.id,
        retry_start,
        audio_duration,
        DEEPGRAM_RETRY_TIMEOUT_SEC,
    )

    tail_data = _post_deepgram_retry(api_key, tail_path, lang_temp)
    if tail_data is None:
        logger.warning("Deepgram tail retry finished without a response; continuing project %s", project.id)
        return data

    recovered_count = _merge_deepgram_tail_retry(data, tail_data, retry_start)
    logger.warning(
        "Deepgram tail retry finished for project %s with %s recovered words; continuing",
        project.id,
        recovered_count,
    )
    return data


def build_sentences(data: dict[str, Any]) -> list[dict[str, Any]]:
    words = _sanitize_deepgram_word_timings(_extract_deepgram_words(data))
    if not words:
        return []

    sentences = []
    current_words: list[dict[str, Any]] = []

    for i, word in enumerate(words):
        current_words.append(word)

        next_word = words[i + 1] if i < len(words) - 1 else None
        start_time = float(current_words[0]["start"])
        end = float(word["end"])
        gap = float(next_word["start"]) - end if next_word else 0.0
        text_so_far = _normalize_text(" ".join(_word_text(w) for w in current_words))
        duration = end - start_time
        is_last_word = next_word is None
        has_sentence_punctuation = bool(SENTENCE_END_RE.search(text_so_far))
        enough_words = _word_count(text_so_far) >= MIN_SEGMENT_WORDS
        reached_hard_duration = duration >= MAX_CONTIGUOUS_SEGMENT_SECONDS

        should_split = (
            is_last_word
            or gap >= MIN_SENTENCE_GAP_SEC
            or reached_hard_duration
            or (has_sentence_punctuation and enough_words)
        )

        if should_split:
            if reached_hard_duration and gap < MIN_SENTENCE_GAP_SEC and not has_sentence_punctuation:
                split_index = _best_forced_split_index(current_words)
                split_words = current_words[: split_index + 1]
                remaining_words = current_words[split_index + 1 :]
                sentences.append(_segment_from_words(split_words))
                current_words = remaining_words
                continue

            segment = _segment_from_words(current_words)
            gap_from_previous = start_time - sentences[-1]["end"] if sentences else 0.0
            can_merge_with_previous = (
                sentences
                and not enough_words
                and gap_from_previous < MERGE_GAP_THRESHOLD_SEC
                and end - sentences[-1]["start"] <= MAX_CONTIGUOUS_SEGMENT_SECONDS
            )
            if can_merge_with_previous:
                sentences[-1]["end"] = segment["end"]
                sentences[-1]["text"] = _normalize_text(f"{sentences[-1]['text']} {segment['text']}")
            else:
                sentences.append(segment)

            current_words = []

    if current_words:
        sentences.append(_segment_from_words(current_words))

    return sentences


def _build_merged_segment(group: list[dict[str, Any]]) -> dict[str, Any]:
    actual_audio_size = round(sum(seg["end"] - seg["start"] for seg in group), 4)
    merged = {
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "text": _normalize_text(" ".join(seg["text"] for seg in group)),
        "trans": _normalize_text(" ".join(seg.get("trans", "") for seg in group)),
        "actual_audio_size": actual_audio_size,
    }
    confidences = [
        float(seg["segment_confidence"])
        for seg in group
        if isinstance(seg.get("segment_confidence"), (int, float))
    ]
    if confidences:
        merged["segment_confidence"] = round(sum(confidences) / len(confidences), 4)
    return merged


def merge_segments_by_gap(segments: list[dict[str, Any]], gap_threshold: float = MERGE_GAP_THRESHOLD_SEC) -> list[dict[str, Any]]:
    if not segments:
        return []

    merged = []
    group = [segments[0]]

    for curr in segments[1:]:
        prev = group[-1]
        gap = curr["start"] - prev["end"]
        combined_duration = curr["end"] - group[0]["start"]
        is_contiguous = gap <= CONTIGUOUS_GAP_TOLERANCE_SEC
        max_merge_duration = MAX_CONTIGUOUS_SEGMENT_SECONDS if is_contiguous else MAX_SEGMENT_SECONDS
        previous_has_sentence_end = bool(SENTENCE_END_RE.search(_normalize_text(prev.get("text", ""))))
        should_merge = gap < gap_threshold and combined_duration <= max_merge_duration and not previous_has_sentence_end

        if should_merge:
            group.append(curr)
        else:
            merged.append(_build_merged_segment(group))
            group = [curr]

    merged.append(_build_merged_segment(group))
    logger.info(
        "Merged %s raw segments into %s final segments using %.2fs threshold",
        len(segments),
        len(merged),
        gap_threshold,
    )
    return merged


def split_audio_on_silence_gaps(audio_path: str, min_silence_ms: int = MIN_SILENCE_MS, silence_thresh: int = -45):
    audio = AudioSegment.from_file(audio_path)
    return detect_silence(audio, min_silence_len=min_silence_ms, silence_thresh=silence_thresh)


def build_sentences_with_audio_gaps(
    data: dict[str, Any],
    audio_path: str,
    min_gap_ms: int = MIN_SILENCE_MS,
) -> list[dict[str, Any]]:
    with timed_stage("segmentation"):
        words = _sanitize_deepgram_word_timings(_extract_deepgram_words(data))
        sentences = build_sentences(data)
        if not sentences:
            return []

        audio_silences = split_audio_on_silence_gaps(audio_path, min_silence_ms=min_gap_ms)
        silence_points_sec = [(start_ms + end_ms) / 2000 for start_ms, end_ms in audio_silences]
        result = []

        for sent in sentences:
            sent_words = [
                word
                for word in words
                if sent["start"] <= (float(word["start"]) + float(word["end"])) / 2 <= sent["end"]
            ]
            inner_splits = [
                split
                for split in silence_points_sec
                if sent["start"] < split < sent["end"]
                and split - sent["start"] >= MIN_SENTENCE_GAP_SEC
                and sent["end"] - split >= MIN_SENTENCE_GAP_SEC
            ]

            if not inner_splits or sent["end"] - sent["start"] <= MAX_SEGMENT_SECONDS:
                result.append(sent)
                continue

            boundaries = [sent["start"], *inner_splits, sent["end"]]
            for index in range(len(boundaries) - 1):
                seg_start = boundaries[index]
                seg_end = boundaries[index + 1]
                seg_words = [
                    word
                    for word in sent_words
                    if seg_start <= (float(word["start"]) + float(word["end"])) / 2 < seg_end
                ]
                if seg_words:
                    segment = _segment_from_words(seg_words)
                    segment["start"] = seg_start
                    segment["end"] = seg_end
                    result.append(segment)

        result = _drop_non_speech_artifacts(result)
        merged = merge_segments_by_gap(result, MERGE_GAP_THRESHOLD_SEC)
        logger.info("Built %s natural segments from %s sentence candidates", len(merged), len(sentences))
        return merged


def get_text_from_deepgram(project, audio_path: str, lang_temp: str) -> str:
    api_key = getattr(settings, "DEEPGRAM_API_KEY", None) or os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured. Add it to .env or the process environment.")

    output_path = _media_path(f"{project.id}_deepgram_output.json")

    with timed_stage("transcription"):
        data = _post_deepgram_audio(api_key, audio_path, lang_temp)
        if not _extract_deepgram_words(data):
            transcript = _extract_deepgram_transcript(data)
            raise RuntimeError(f"Deepgram returned no timestamped words. Transcript: {transcript[:500]}")
        data = _retry_untranscribed_gaps(data, project, audio_path, api_key, lang_temp)
        data = _retry_untranscribed_tail(data, project, audio_path, api_key, lang_temp)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def _split_translation_chunks(text: str, max_bytes: int = TRANSLATION_MAX_BYTES) -> list[str]:
    text = _normalize_text(text)
    if len(text.encode("utf-8")) <= max_bytes:
        return [text] if text else []

    parts = re.split(r"(?<=[.?!।])\s+", text)
    chunks = []
    current = ""
    for part in parts:
        candidate = _normalize_text(f"{current} {part}") if current else part
        if len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(part.encode("utf-8")) <= max_bytes:
            current = part
            continue

        words = part.split()
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate.encode("utf-8")) <= max_bytes:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word

    if current:
        chunks.append(current)

    return chunks


def chunk_translate(main_txt: str, src_lang: str, tar_lang: str) -> str:
    translated_parts = []
    for chunk in _split_translation_chunks(main_txt):
        result = GoogleTranslator(source=src_lang, target=tar_lang).translate(chunk)
        if result:
            translated_parts.append(result)
    return _normalize_text(" ".join(translated_parts))


async def _translate_one_segment(
    index: int,
    segment: dict[str, Any],
    src_lang: str,
    target_lang: str,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> tuple[int, dict[str, Any], bool]:
    text = _normalize_text(segment.get("text", ""))
    updated = {**segment, "text": text, "trans": ""}
    if not text:
        return index, updated, False

    for attempt in range(3):
        try:
            async with semaphore:
                loop = asyncio.get_running_loop()
                translated = await loop.run_in_executor(executor, chunk_translate, text, src_lang, target_lang)
            updated["trans"] = _normalize_text(translated)
            return index, updated, bool(updated["trans"])
        except Exception as exc:
            logger.warning(
                "Translation failed for segment %s on attempt %s/3: %s",
                index,
                attempt + 1,
                exc,
            )
            if attempt < 2:
                await asyncio.sleep(0.4 * (2 ** attempt))

    updated["trans"] = text
    logger.error("Using source text as fallback for untranslated segment %s", index)
    return index, updated, False


async def translate_segments_async(
    segments: list[dict[str, Any]],
    src_lang: str,
    target_lang: str,
    concurrency: int = TRANSLATION_CONCURRENCY,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    semaphore = asyncio.Semaphore(concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        tasks = [
            _translate_one_segment(index, segment, src_lang, target_lang, semaphore, executor)
            for index, segment in enumerate(segments)
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda item: item[0])
    translated_segments = [item[1] for item in results]
    success_count = sum(1 for _, _, success in results if success)
    if success_count == 0 and any(_normalize_text(segment.get("text", "")) for segment in segments):
        raise RuntimeError("All segment translations failed")

    logger.info("Translated %s/%s segments", success_count, len(segments))
    return translated_segments


async def review_source_segments_async(
    segments: list[dict[str, Any]],
    project,
    src_lang: str,
) -> list[dict[str, Any]]:
    if not TRANSCRIPT_QA_ENABLED:
        return segments
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("Skipping source transcript QA because OPENAI_API_KEY is not configured")
        return segments

    try:
        reviewed = await apply_source_transcript_review(
            segments,
            source_language_code=src_lang,
            source_language_name=getattr(project.source_language, "name", ""),
            batch_size=TRANSCRIPT_QA_BATCH_SIZE,
        )
        fixed_count = sum(1 for segment in reviewed if segment.get("transcript_qa_status") == "fixed")
        logger.info("Source transcript QA reviewed %s segments and fixed %s", len(reviewed), fixed_count)
        return reviewed
    except Exception as exc:
        logger.warning("Source transcript QA failed; keeping existing transcript text: %s", exc)
        return segments


async def review_translated_segments_async(
    segments: list[dict[str, Any]],
    project,
    src_lang: str,
    target_lang: str,
) -> list[dict[str, Any]]:
    if not TRANSLATION_QA_ENABLED:
        return segments
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("Skipping translation QA because OPENAI_API_KEY is not configured")
        return segments

    try:
        reviewed = await apply_translation_review(
            segments,
            source_language_code=src_lang,
            target_language_code=target_lang,
            source_language_name=getattr(project.source_language, "name", ""),
            target_language_name=getattr(project.target_language, "name", ""),
            batch_size=TRANSLATION_QA_BATCH_SIZE,
        )
        fixed_count = sum(1 for segment in reviewed if segment.get("qa_status") == "fixed")
        logger.info("Translation QA reviewed %s segments and fixed %s", len(reviewed), fixed_count)
        return reviewed
    except Exception as exc:
        logger.warning("Translation QA failed; keeping existing translations: %s", exc)
        return segments


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Cannot run synchronous video pipeline helper while an event loop is already running")


def get_segments_translation(project, audio_path: str) -> str:
    lang_temp = project.source_language.lang_code
    tar_lang = project.target_language.lang_code
    output_path = get_text_from_deepgram(project, audio_path, lang_temp)

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = build_sentences_with_audio_gaps(data, audio_path, min_gap_ms=MIN_SILENCE_MS)

    with timed_stage("source transcript QA"):
        sentences = _run_async(review_source_segments_async(sentences, project, lang_temp))

    with timed_stage("translation"):
        translated = _run_async(translate_segments_async(sentences, lang_temp, tar_lang, TRANSLATION_CONCURRENCY))

    with timed_stage("translation QA"):
        translated = _run_async(review_translated_segments_async(translated, project, lang_temp, tar_lang))

    file_name = f"{project.id}_segments.json"
    file_path = _media_path(file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=4)

    project.segments = file_name
    project.save(update_fields=["segments"])
    return file_path


def load_segments(file_path: str) -> list[Lines]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Lines(**item) for item in data]


def export_dubbed_video(project) -> str:
    final_audio_path = _media_path(f"{project.id}_fitted_final.wav")
    output_video = _media_path(f"{project.id}_exported_dubbed.mp4")

    with timed_stage("video export"):
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i",
            project.input_video.path,
            "-i",
            final_audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_video,
        ]
        _run_subprocess(cmd, "FFmpeg dubbed video export failed")

    project.output_video = _relative_media_name(output_video)
    project.save(update_fields=["output_video"])
    return output_video


def sec_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def sec_to_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def _is_cjk_text(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _wrap_cjk_text(text: str, max_chars: int = SUBTITLE_MAX_CJK_CHARS_PER_LINE) -> str:
    if len(text) <= max_chars:
        return text

    lines = []
    current = ""
    break_chars = "，。！？；：、,.?!;:"
    for index, char in enumerate(text):
        current += char
        next_char = text[index + 1] if index + 1 < len(text) else ""
        inside_latin_word = char.isascii() and char.isalnum() and next_char.isascii() and next_char.isalnum()
        should_break = (
            len(current) >= max_chars
            and not inside_latin_word
            and (char in break_chars or len(current) >= max_chars + 6 or next_char.isspace())
        )
        if should_break:
            lines.append(current.strip())
            current = ""

    if current.strip():
        lines.append(current.strip())

    return "\n".join(lines)


def _wrap_word_text(text: str, max_words: int = SUBTITLE_MAX_WORDS_PER_LINE) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    lines = []
    for index in range(0, len(words), max_words):
        lines.append(" ".join(words[index : index + max_words]))
    return "\n".join(lines)


def _format_subtitle_text(text: str) -> str:
    text = _normalize_text(text).replace("\r", " ").replace("\n", " ")
    text = _normalize_text(text)
    if _is_cjk_text(text):
        return _wrap_cjk_text(text)
    return _wrap_word_text(text)


def _subtitle_font_name(project) -> str:
    lang_code = getattr(getattr(project, "target_language", None), "lang_code", "") or ""
    lang_code = lang_code.lower()
    if lang_code.startswith("zh"):
        if os.path.exists(_media_path("font/ResourceHanRoundedCN-Normal.ttf")):
            return "Resource Han Rounded CN Normal"
        return "Hiragino Sans GB"
    if lang_code.startswith("bn"):
        if os.path.exists(_media_path("font/NotoSansBengali-Regular.ttf")):
            return "Noto Sans Bengali"
        if os.path.exists(_media_path("font/Noto Sans Bengali.ttf")):
            return "Noto Sans Bengali"
        if os.path.exists(_media_path("font/kalpurush_ANSI.ttf")):
            return "Kalpurush"
        return "Kohinoor Bangla"
    if lang_code.startswith("ja"):
        return "Hiragino Sans"
    if lang_code.startswith("th"):
        return "Sukhumvit Set"
    return "Noto Sans"


def _ass_escape_text(text: str) -> str:
    text = text.replace("{", "(").replace("}", ")")
    return text.replace("\n", r"\N")


def _create_ass_subtitle(project, segments: list[Lines]) -> str:
    file_name = f"{project.id}_exported_dubbed.ass"
    output_file = _media_path(file_name)
    font_name = _subtitle_font_name(project)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n")
        f.write("PlayResX: 1280\n")
        f.write("PlayResY: 720\n\n")
        f.write("[V4+ Styles]\n")
        f.write(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )
        f.write(
            f"Style: Default,{font_name},38,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,"
            "0,0,0,0,100,100,0,0,1,3,1,2,55,55,42,1\n\n"
        )
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for line in segments:
            subtitle_text = _ass_escape_text(_format_subtitle_text(line.trans))
            f.write(
                f"Dialogue: 0,{sec_to_ass_time(line.start)},{sec_to_ass_time(line.end)},"
                f"Default,,0,0,0,,{subtitle_text}\n"
            )

    return output_file


def create_subtitle(project, segments_path: str) -> str:
    segments = load_segments(segments_path)
    file_name = f"{project.id}_exported_dubbed.srt"
    output_file = _media_path(file_name)
    legacy_file_name = f"{project.id}_subtitle.srt"
    legacy_output_file = _media_path(legacy_file_name)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, line in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_srt_time(line.start)} --> {sec_to_srt_time(line.end)}\n")
            f.write(f"{_format_subtitle_text(line.trans)}\n\n")

    project.subtitle = file_name
    project.save(update_fields=["subtitle"])

    if legacy_output_file != output_file:
        shutil.copyfile(output_file, legacy_output_file)

    ass_output_file = _create_ass_subtitle(project, segments)
    logger.info("Subtitle file created for project %s: %s", project.id, output_file)
    logger.info("ASS subtitle file created for project %s: %s", project.id, ass_output_file)
    return output_file


def burn_subtitle(project, segments_path: str) -> str:
    if not _ffmpeg_filter_exists("subtitles"):
        raise RuntimeError(
            "FFmpeg was found, but it does not include the 'subtitles' filter. "
            "Install an FFmpeg build with libass/subtitle support. In Docker, use apt-get install -y ffmpeg "
            "or a full FFmpeg image/package that includes libass."
        )

    final_audio_path = _media_path(f"{project.id}_fitted_final.wav")
    output_video = _media_path(f"{project.id}_output.mp4")
    ass_subtitle_file = _media_path(f"{project.id}_exported_dubbed.ass")
    subtitle_file = ass_subtitle_file if os.path.exists(ass_subtitle_file) else (
        project.subtitle.path if project.subtitle else _media_path(f"{project.id}_exported_dubbed.srt")
    )
    subtitle_name = os.path.basename(subtitle_file)

    with timed_stage("subtitle burn"):
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i",
            project.input_video.path,
            "-i",
            final_audio_path,
            "-vf",
            f"subtitles=filename='{subtitle_name}':fontsdir='font'",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *_subtitle_encoder_args(),
            "-c:a",
            "aac",
            "-shortest",
            output_video,
        ]
        _run_subprocess(cmd, "FFmpeg subtitle burn failed", cwd=settings.MEDIA_ROOT)

    project.output_video = _relative_media_name(output_video)
    project.save(update_fields=["output_video"])
    return output_video


def cleanup_intermediate_files(project_id=None) -> int:
    if project_id is None:
        raise ValueError("cleanup_intermediate_files requires a project_id")

    media_path = Path(settings.MEDIA_ROOT)
    patterns = [
        f"{project_id}_*_probe.mp3",
        f"{project_id}_*_probe_trimmed.mp3",
        f"{project_id}_*_seg.mp3",
        f"{project_id}_deepgram_tail_retry.wav",
    ]
    deleted_count = 0

    for pattern in patterns:
        for file_path in media_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                logger.info("Deleted project %s segment audio temp file: %s", project_id, file_path.name)
                deleted_count += 1

    return deleted_count


def _subtitle_encoder_args() -> list[str]:
    codec = SUBTITLE_VIDEO_CODEC
    if codec == "auto":
        codec = "h264_videotoolbox" if _ffmpeg_encoder_exists("h264_videotoolbox") else "libx264"
    if codec == "libx264":
        return ["-c:v", "libx264", "-preset", SUBTITLE_VIDEO_PRESET, "-crf", SUBTITLE_VIDEO_CRF]
    if codec == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", SUBTITLE_VIDEO_BITRATE]
    return ["-c:v", codec]


def _trim_silence(audio: AudioSegment, silence_thresh: int = -45, max_trim_ms: int = 250) -> AudioSegment:
    start_trim = min(detect_leading_silence(audio, silence_threshold=silence_thresh), max_trim_ms)
    end_trim = min(detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh), max_trim_ms)
    if start_trim + end_trim >= len(audio):
        return audio
    return audio[start_trim : len(audio) - end_trim]


def _build_atempo_filter(ratio: float) -> str:
    parts = []
    remaining = ratio
    while remaining > MAX_ATEMPO_RATIO:
        parts.append(f"atempo={MAX_ATEMPO_RATIO}")
        remaining /= MAX_ATEMPO_RATIO
    while remaining < MIN_ATEMPO_RATIO:
        parts.append(f"atempo={MIN_ATEMPO_RATIO}")
        remaining /= MIN_ATEMPO_RATIO
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def _time_stretch_audio(input_path: str, output_path: str, actual_ms: float, target_ms: float):
    if target_ms <= 0:
        raise ValueError(f"Invalid target duration: {target_ms}ms")

    ratio = actual_ms / target_ms
    if abs(actual_ms - target_ms) <= STRETCH_TOLERANCE_MS:
        shutil.copyfile(input_path, output_path)
        logger.info("Audio fit skipped for %s; %.0fms is within tolerance", output_path, abs(actual_ms - target_ms))
        return

    if ratio >= EXTREME_ATEMPO_RATIO or ratio <= 1 / EXTREME_ATEMPO_RATIO:
        logger.warning("Extreme audio stretch ratio %.2fx for %s", ratio, input_path)

    safe_ratio = max(MIN_ATEMPO_RATIO, min(ratio, MAX_ATEMPO_RATIO))
    if safe_ratio != ratio:
        logger.warning("Clamped audio stretch ratio from %.2fx to %.2fx for %s", ratio, safe_ratio, input_path)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i",
        input_path,
        "-filter:a",
        _build_atempo_filter(safe_ratio),
        output_path,
    ]
    _run_subprocess(cmd, "FFmpeg audio fitting failed")


def _short_audio_offset_ms(actual_ms: float, target_ms: float) -> int:
    spare_ms = max(0, int(round(target_ms - actual_ms)))
    return min(SHORT_AUDIO_DELAY_MS, spare_ms)


def _fit_or_place_natural_audio(
    input_path: str,
    output_path: str,
    actual_ms: float,
    target_ms: float,
) -> dict[str, Any]:
    if target_ms <= 0:
        raise ValueError(f"Invalid target duration: {target_ms}ms")

    ratio = actual_ms / target_ms
    if ratio < SHORT_AUDIO_STRETCH_THRESHOLD:
        offset_ms = _short_audio_offset_ms(actual_ms, target_ms)
        shutil.copyfile(input_path, output_path)
        logger.info(
            "Keeping short segment natural: target=%.0fms natural=%.0fms ratio=%.2fx offset=%sms",
            target_ms,
            actual_ms,
            ratio,
            offset_ms,
        )
        return {
            "audio_offset_ms": offset_ms,
            "natural_audio_ms": round(actual_ms, 2),
            "fitted_audio_ms": round(actual_ms, 2),
            "fit_mode": "natural_delayed",
        }

    fit_target_ms = target_ms
    fit_mode = "time_stretched"
    if ratio > MAX_NATURAL_SPEEDUP_RATIO:
        fit_target_ms = actual_ms / MAX_NATURAL_SPEEDUP_RATIO
        fit_mode = "speedup_limited"
        if fit_target_ms > target_ms + STRETCH_TOLERANCE_MS:
            fit_mode = "speedup_limited_unfitted"
        logger.info(
            "Limiting speed-up: target=%.0fms natural=%.0fms requested=%.2fx capped=%.2fx fitted=%.0fms",
            target_ms,
            actual_ms,
            ratio,
            MAX_NATURAL_SPEEDUP_RATIO,
            fit_target_ms,
        )
    elif ratio < MIN_NATURAL_SLOWDOWN_RATIO:
        fit_target_ms = actual_ms / MIN_NATURAL_SLOWDOWN_RATIO
        fit_mode = "slowdown_limited"
        logger.info(
            "Limiting slow-down: target=%.0fms natural=%.0fms requested=%.2fx capped=%.2fx fitted=%.0fms",
            target_ms,
            actual_ms,
            ratio,
            MIN_NATURAL_SLOWDOWN_RATIO,
            fit_target_ms,
        )

    _time_stretch_audio(input_path, output_path, actual_ms, fit_target_ms)
    fitted_ms = actual_ms if abs(actual_ms - fit_target_ms) <= STRETCH_TOLERANCE_MS else fit_target_ms
    metadata = {
        "audio_offset_ms": 0,
        "natural_audio_ms": round(actual_ms, 2),
        "fitted_audio_ms": round(fitted_ms, 2),
        "fit_mode": "within_tolerance" if abs(actual_ms - fit_target_ms) <= STRETCH_TOLERANCE_MS else fit_mode,
    }
    if fit_mode == "speedup_limited_unfitted":
        metadata["required_speedup_ratio"] = round(ratio, 4)
        metadata["max_speedup_ratio"] = MAX_NATURAL_SPEEDUP_RATIO
    return metadata


def _fits_after_max_speedup(actual_ms: float, target_ms: float) -> bool:
    return actual_ms <= (target_ms * MAX_NATURAL_SPEEDUP_RATIO) + STRETCH_TOLERANCE_MS


def _is_valid_audio_file(file_path: str) -> bool:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False

    try:
        return _duration_ms(file_path) > 0
    except Exception:
        return False


def _write_silent_probe(file_path: str, duration_ms: float) -> None:
    safe_duration = max(250, int(round(duration_ms)))
    AudioSegment.silent(duration=safe_duration).export(file_path, format="mp3")


async def _save_tts_with_retry(text: str, voice: str, output_path: str, index: int) -> None:
    last_error = None
    for attempt in range(TTS_RETRY_ATTEMPTS):
        try:
            if DEEPGRAM_TTS_MODEL:
                response = await asyncio.to_thread(
                    requests.post,
                    "https://api.deepgram.com/v1/speak",
                    headers={
                        "Authorization": f"Token {getattr(settings, 'DEEPGRAM_API_KEY', None) or os.environ.get('DEEPGRAM_API_KEY')}",
                        "Content-Type": "application/json",
                    },
                    params={"model": DEEPGRAM_TTS_MODEL, "encoding": "mp3"},
                    json={"text": text},
                    timeout=60,
                )
                if not response.ok:
                    raise RuntimeError(f"Deepgram TTS failed ({response.status_code}): {response.text[:300]}")
                await asyncio.to_thread(Path(output_path).write_bytes, response.content)
            else:
                await Communicate(text, voice).save(output_path)
            if _is_valid_audio_file(output_path):
                return
            raise RuntimeError("edge-tts created an empty or unreadable audio file")
        except Exception as exc:
            last_error = exc
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                logger.warning("Unable to remove failed TTS file %s", output_path)

            logger.warning(
                "TTS failed for segment %s on attempt %s/%s: %s",
                index,
                attempt + 1,
                TTS_RETRY_ATTEMPTS,
                exc,
            )
            if attempt < TTS_RETRY_ATTEMPTS - 1:
                await asyncio.sleep(TTS_RETRY_BASE_DELAY_SEC * (2 ** attempt))

    raise RuntimeError(f"TTS failed for segment {index} after {TTS_RETRY_ATTEMPTS} attempts: {last_error}")


async def _generate_tts_file(
    index: int,
    seg: Lines,
    voice: str,
    project_id: int,
    semaphore: asyncio.Semaphore,
    force: bool = False,
) -> str:
    if not seg.trans or not seg.trans.strip():
        raise ValueError(f"Segment {index} has empty translation")

    probe_path = _media_path(f"{project_id}_{index}_probe.mp3")
    if force and os.path.exists(probe_path):
        try:
            os.remove(probe_path)
        except OSError:
            logger.warning("Unable to remove stale TTS probe %s", probe_path)

    if _is_valid_audio_file(probe_path):
        logger.info("Reusing existing TTS probe for segment %s: %s", index, probe_path)
        return probe_path

    text = _normalize_text(seg.trans)
    target_ms = (seg.end - seg.start) * 1000
    async with semaphore:
        try:
            await _save_tts_with_retry(text, voice, probe_path, index)
        except Exception as exc:
            logger.error(
                "TTS permanently failed for segment %s; inserting silence so project can finish: %s",
                index,
                exc,
            )
            await asyncio.to_thread(_write_silent_probe, probe_path, target_ms)

    return probe_path


async def _generate_tts_text_file(
    index: int,
    text: str,
    voice: str,
    output_path: str,
    semaphore: asyncio.Semaphore,
    target_ms: float,
) -> str:
    async with semaphore:
        try:
            await _save_tts_with_retry(_normalize_text(text), voice, output_path, index)
        except Exception as exc:
            logger.error(
                "TTS permanently failed for segment %s rewrite candidate; inserting silence so project can finish: %s",
                index,
                exc,
            )
            await asyncio.to_thread(_write_silent_probe, output_path, target_ms)
    return output_path


def _raw_segment_duration_ms(seg: Lines) -> int:
    return max(1, int(round((seg.end - seg.start) * 1000)))


def _segment_target_ms(
    segments: list[Lines],
    index: int,
    video_duration_ms: int | None = None,
) -> int:
    seg = segments[index]
    target_ms = _raw_segment_duration_ms(seg)
    start_ms = int(round(seg.start * 1000))

    if index < len(segments) - 1:
        next_start_ms = int(round(segments[index + 1].start * 1000))
        next_window_ms = next_start_ms - start_ms - MIN_AUDIO_SEGMENT_GAP_MS
        if next_window_ms > 0:
            target_ms = min(target_ms, next_window_ms)

    if video_duration_ms is not None:
        remaining_ms = video_duration_ms - start_ms
        if remaining_ms > 0:
            target_ms = min(target_ms, remaining_ms)

    return max(1, target_ms)


async def _measure_probe_ms(probe_path: str) -> int:
    def measure() -> int:
        audio = AudioSegment.from_file(probe_path)
        return len(_trim_silence(audio))

    return await asyncio.to_thread(measure)


def _rewrite_candidate_is_acceptable(target_ms: int, best_actual_ms: int, candidate_actual_ms: int) -> bool:
    best_diff_ms = abs(target_ms - best_actual_ms)
    candidate_diff_ms = abs(target_ms - candidate_actual_ms)
    if candidate_diff_ms >= best_diff_ms - STRETCH_TOLERANCE_MS:
        return False

    if candidate_diff_ms <= REWRITE_GAP_THRESHOLD_MS:
        return True

    if best_actual_ms > target_ms and candidate_actual_ms < target_ms - REWRITE_GAP_THRESHOLD_MS:
        return False
    if best_actual_ms < target_ms and candidate_actual_ms > target_ms + REWRITE_GAP_THRESHOLD_MS:
        return False

    return True


def _save_segment_translation_metadata(segments_path: str, segments: list[Lines]) -> None:
    with open(segments_path, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    for index, seg in enumerate(segments):
        if index < len(raw_segments):
            raw_segments[index]["trans"] = seg.trans
            for key, value in seg.extra.items():
                if key.startswith("rewrite_"):
                    raw_segments[index][key] = value

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(raw_segments, f, ensure_ascii=False, indent=4)


async def _rewrite_segment_until_timing_is_close(
    index: int,
    seg: Lines,
    segments: list[Lines],
    project,
    probe_path: str,
    tts_semaphore: asyncio.Semaphore,
    rewrite_semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    video_duration_ms = int(round(project.original_duration * 1000))
    target_ms = _segment_target_ms(segments, index, video_duration_ms)
    max_natural_ms = int(round(target_ms * MAX_NATURAL_SPEEDUP_RATIO))
    best_path = probe_path
    best_actual_ms = await _measure_probe_ms(probe_path)
    best_diff_ms = abs(target_ms - best_actual_ms)
    best_too_long_to_fit = best_actual_ms > target_ms and not _fits_after_max_speedup(best_actual_ms, target_ms)

    if best_actual_ms > target_ms and not best_too_long_to_fit:
        return index, best_path

    if best_diff_ms <= REWRITE_GAP_THRESHOLD_MS and not best_too_long_to_fit:
        return index, best_path

    if not os.environ.get("OPENAI_API_KEY"):
        logger.info(
            "Skipping timing rewrite for segment %s because OPENAI_API_KEY is not configured; diff=%sms",
            index,
            best_diff_ms,
        )
        return index, best_path

    original_translation = seg.trans
    max_attempts = max(REWRITE_MAX_ATTEMPTS, HARD_FIT_REWRITE_MAX_ATTEMPTS) if best_too_long_to_fit else REWRITE_MAX_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        try:
            async with rewrite_semaphore:
                rewrite = await rewrite_translation_for_timing(
                    source_text=seg.text,
                    current_translation=seg.trans,
                    source_language_code=getattr(project.source_language, "lang_code", ""),
                    source_language_name=getattr(project.source_language, "name", ""),
                    target_language_code=getattr(project.target_language, "lang_code", ""),
                    target_language_name=getattr(project.target_language, "name", ""),
                    target_voice=getattr(project.target_language, "lang_model", ""),
                    target_ms=max_natural_ms if best_actual_ms > target_ms else target_ms,
                    actual_audio_ms=best_actual_ms,
                    attempt=attempt,
                    must_fit=best_too_long_to_fit,
                )
            rewritten_text = _normalize_text(rewrite.get("trans", ""))
            if not rewritten_text or rewritten_text == seg.trans:
                logger.info("Timing rewrite kept segment %s unchanged on attempt %s", index, attempt)
                break

            candidate_path = _media_path(f"{project.id}_{index}_rewrite_{attempt}_probe.mp3")
            candidate_path = await _generate_tts_text_file(
                index,
                rewritten_text,
                project.target_language.lang_model,
                candidate_path,
                tts_semaphore,
                target_ms,
            )
            candidate_actual_ms = await _measure_probe_ms(candidate_path)
            candidate_diff_ms = abs(target_ms - candidate_actual_ms)
            logger.info(
                "Timing rewrite segment %s attempt %s: target=%sms previous=%sms candidate=%sms",
                index,
                attempt,
                target_ms,
                best_actual_ms,
                candidate_actual_ms,
            )

            if not _rewrite_candidate_is_acceptable(target_ms, best_actual_ms, candidate_actual_ms):
                seg.extra.update(
                    {
                        "rewrite_rejected_action": rewrite.get("action", ""),
                        "rewrite_rejected_attempts": attempt,
                        "rewrite_rejected_trans": rewritten_text,
                        "rewrite_rejected_audio_diff_ms": candidate_diff_ms,
                        "rewrite_target_natural_audio_ms": max_natural_ms,
                    }
                )
                logger.info(
                    "Rejected timing rewrite for segment %s attempt %s: best_diff=%sms candidate_diff=%sms",
                    index,
                    attempt,
                    best_diff_ms,
                    candidate_diff_ms,
                )
                continue

            seg.trans = rewritten_text
            best_path = candidate_path
            best_actual_ms = candidate_actual_ms
            best_diff_ms = candidate_diff_ms
            best_too_long_to_fit = best_actual_ms > target_ms and not _fits_after_max_speedup(
                best_actual_ms,
                target_ms,
            )
            seg.extra.update(
                {
                    "rewrite_action": rewrite.get("action", ""),
                    "rewrite_attempts": attempt,
                    "rewrite_original_trans": original_translation,
                    "rewrite_audio_diff_ms": best_diff_ms,
                    "rewrite_target_natural_audio_ms": max_natural_ms,
                }
            )

            if best_actual_ms > target_ms and not best_too_long_to_fit:
                break
            if best_diff_ms <= REWRITE_GAP_THRESHOLD_MS and not best_too_long_to_fit:
                break
        except Exception as exc:
            logger.warning("Timing rewrite failed for segment %s on attempt %s: %s", index, attempt, exc)
            break

    if best_actual_ms > target_ms and not _fits_after_max_speedup(best_actual_ms, target_ms):
        seg.extra.update(
            {
                "rewrite_status": "timing_unfitted",
                "rewrite_target_natural_audio_ms": max_natural_ms,
                "rewrite_final_natural_audio_ms": best_actual_ms,
                "rewrite_required_speedup_ratio": round(best_actual_ms / target_ms, 4),
                "rewrite_max_speedup_ratio": MAX_NATURAL_SPEEDUP_RATIO,
            }
        )

    return index, best_path


def _prepare_tts_for_fitting(probe_path: str) -> tuple[str, float]:
    audio = AudioSegment.from_file(probe_path)
    trimmed = _trim_silence(audio)
    if len(trimmed) != len(audio):
        trimmed_path = probe_path.replace("_probe.mp3", "_probe_trimmed.mp3")
        trimmed.export(trimmed_path, format="mp3")
        return trimmed_path, float(len(trimmed))
    return probe_path, float(len(audio))


async def _fit_tts_file(
    index: int,
    seg: Lines,
    segments: list[Lines],
    video_duration_ms: int,
    project_id: int,
    probe_path: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, dict[str, Any]]:
    raw_target_ms = _raw_segment_duration_ms(seg)
    target_ms = _segment_target_ms(segments, index, video_duration_ms)
    force_target_fit = target_ms < raw_target_ms - STRETCH_TOLERANCE_MS
    final_path = _media_path(f"{project_id}_{index}_seg.mp3")

    async with semaphore:
        fit_input_path, actual_ms = await asyncio.to_thread(_prepare_tts_for_fitting, probe_path)
        logger.info(
            "Segment %s audio fit: target=%.0fms natural=%.0fms ratio=%.2fx",
            index,
            target_ms,
            actual_ms,
            actual_ms / target_ms if target_ms else 0,
        )
        fit_metadata = await asyncio.to_thread(
            _fit_or_place_natural_audio,
            fit_input_path,
            final_path,
            actual_ms,
            target_ms,
        )
        fit_metadata["target_audio_ms"] = round(target_ms, 2)
        if force_target_fit:
            fit_metadata["raw_segment_target_ms"] = round(raw_target_ms, 2)

    for path in {probe_path, fit_input_path}:
        try:
            if path != final_path and os.path.exists(path):
                os.remove(path)
        except OSError:
            logger.warning("Unable to remove temporary TTS file %s", path)

    return index, fit_metadata


def _save_segment_audio_metadata(segments_path: str, fit_results: list[tuple[int, dict[str, Any]]]) -> None:
    if not fit_results:
        return

    with open(segments_path, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    for index, metadata in fit_results:
        if index < len(raw_segments):
            raw_segments[index].update(metadata)

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(raw_segments, f, ensure_ascii=False, indent=4)


async def generate_audio_synced(project, segments_path: str):
    segments = load_segments(segments_path)
    voice = project.target_language.lang_model

    with timed_stage("TTS generation"):
        tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
        probe_paths = await asyncio.gather(
            *[_generate_tts_file(index, seg, voice, project.id, tts_semaphore) for index, seg in enumerate(segments)]
        )

    if REWRITE_GAP_THRESHOLD_MS > 0 and REWRITE_MAX_ATTEMPTS > 0:
        with timed_stage("translation timing rewrite"):
            video_duration_ms = int(round(project.original_duration * 1000))
            rewrite_semaphore = asyncio.Semaphore(REWRITE_CONCURRENCY)
            rewrite_results = await asyncio.gather(
                *[
                    _rewrite_segment_until_timing_is_close(
                        index,
                        seg,
                        segments,
                        project,
                        probe_paths[index],
                        tts_semaphore,
                        rewrite_semaphore,
                    )
                    for index, seg in enumerate(segments)
                ]
            )
            rewrite_results = sorted(rewrite_results, key=lambda item: item[0])
            probe_paths = [path for _, path in rewrite_results]
            _save_segment_translation_metadata(segments_path, segments)

    with timed_stage("audio fitting"):
        video_duration_ms = int(round(project.original_duration * 1000))
        fit_semaphore = asyncio.Semaphore(AUDIO_FIT_CONCURRENCY)
        fit_results = await asyncio.gather(
            *[
                _fit_tts_file(index, seg, segments, video_duration_ms, project.id, probe_paths[index], fit_semaphore)
                for index, seg in enumerate(segments)
            ]
        )
        _save_segment_audio_metadata(segments_path, list(fit_results))


def mix_final_audio(project, segments_path: str) -> str:
    segments = load_segments(segments_path)
    video_duration_ms = int(round(project.original_duration * 1000))
    final_audio = AudioSegment.silent(duration=video_duration_ms)
    schedule_metadata = []

    with timed_stage("audio mixing"):
        next_available_ms = 0
        for i, seg in enumerate(segments):
            file_path = _media_path(f"{project.id}_{i}_seg.mp3")
            if not os.path.exists(file_path):
                raise RuntimeError(f"Missing fitted segment audio: {file_path}")

            audio_seg = AudioSegment.from_file(file_path)
            max_allowed = max(0, int(round((seg.end - seg.start) * 1000)))
            requested_offset_ms = max(0, int(round(getattr(seg, "audio_offset_ms", 0) or 0)))
            allow_overhang = getattr(seg, "fit_mode", "") == "speedup_limited" and i < len(segments) - 1
            allowed_len_ms = len(audio_seg) if allow_overhang else max_allowed
            safe_offset_ms = min(requested_offset_ms, max(0, allowed_len_ms - len(audio_seg)))
            original_start_ms = int(round(seg.start * 1000)) + safe_offset_ms
            gap_ms = MIN_AUDIO_SEGMENT_GAP_MS if i > 0 and original_start_ms >= next_available_ms else 0
            required_start_ms = next_available_ms + gap_ms
            start_ms = max(original_start_ms, required_start_ms)

            if allow_overhang and i < len(segments) - 1:
                next_original_start_ms = int(round(segments[i + 1].start * 1000))
                if start_ms + len(audio_seg) + MIN_AUDIO_SEGMENT_GAP_MS > next_original_start_ms:
                    allow_overhang = False
                    allowed_len_ms = max_allowed
                    safe_offset_ms = min(requested_offset_ms, max(0, allowed_len_ms - len(audio_seg)))
                    original_start_ms = int(round(seg.start * 1000)) + safe_offset_ms
                    gap_ms = MIN_AUDIO_SEGMENT_GAP_MS if i > 0 and original_start_ms >= next_available_ms else 0
                    required_start_ms = next_available_ms + gap_ms
                    start_ms = max(original_start_ms, required_start_ms)

            if len(audio_seg) > max_allowed and not allow_overhang:
                logger.info("Trimming segment %s from %sms to %sms", i, len(audio_seg), max_allowed)
                audio_seg = audio_seg[:max_allowed]
            elif len(audio_seg) > max_allowed:
                logger.info(
                    "Allowing segment %s to overhang by %sms to preserve natural speech speed",
                    i,
                    len(audio_seg) - max_allowed,
                )

            if safe_offset_ms:
                logger.info("Segment %s placed %sms later to preserve natural TTS speed", i, safe_offset_ms)

            if start_ms > original_start_ms:
                logger.info(
                    "Segment %s delayed by %sms to keep a natural gap and prevent overlap",
                    i,
                    start_ms - original_start_ms,
                )

            if start_ms >= video_duration_ms:
                logger.warning("Skipping segment %s because its scheduled start is beyond video duration", i)
                schedule_metadata.append((i, {"scheduled_start_ms": start_ms, "scheduled_end_ms": start_ms}))
                continue

            if start_ms + len(audio_seg) > video_duration_ms:
                logger.info(
                    "Trimming segment %s at video end from %sms to %sms",
                    i,
                    len(audio_seg),
                    video_duration_ms - start_ms,
                )
                audio_seg = audio_seg[: max(0, video_duration_ms - start_ms)]

            final_audio = final_audio.overlay(audio_seg, position=start_ms)
            end_ms = start_ms + len(audio_seg)
            next_available_ms = end_ms
            schedule_metadata.append(
                (
                    i,
                    {
                        "audio_offset_ms": max(0, start_ms - int(round(seg.start * 1000))),
                        "audio_gap_before_ms": gap_ms,
                        "scheduled_start_ms": start_ms,
                        "scheduled_end_ms": end_ms,
                    },
                )
            )

        final_audio = final_audio[:video_duration_ms]
        if len(final_audio) < video_duration_ms:
            final_audio += AudioSegment.silent(duration=video_duration_ms - len(final_audio))

        final_audio_path = _media_path(f"{project.id}_fitted_final.wav")
        final_audio.export(final_audio_path, format="wav")
        _save_segment_audio_metadata(segments_path, schedule_metadata)

    return final_audio_path
