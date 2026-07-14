from pathlib import Path
from types import SimpleNamespace
from pydub import AudioSegment

from app.engine_adapter import load_engine


def test_slow_upload_is_retried_with_buffered_payload(tmp_path: Path, monkeypatch):
    engine = load_engine(tmp_path)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-test-audio")
    responses = [
        SimpleNamespace(ok=False, status_code=408, text='{"err_code":"SLOW_UPLOAD"}'),
        SimpleNamespace(ok=True, status_code=200, text="", json=lambda: {"results": {}}),
    ]
    payloads = []

    def post(*_, **kwargs):
        payloads.append(kwargs["data"])
        return responses.pop(0)

    monkeypatch.setattr(engine.requests, "post", post)
    monkeypatch.setattr(engine.time, "sleep", lambda _: None)
    monkeypatch.setattr(engine, "DEEPGRAM_UPLOAD_ATTEMPTS", 3)
    assert engine._post_deepgram_audio("key", str(audio), "en") == {"results": {}}
    assert payloads == [b"RIFF-test-audio", b"RIFF-test-audio"]


def test_large_wav_is_compressed_before_deepgram_upload(tmp_path: Path, monkeypatch):
    engine = load_engine(tmp_path)
    audio = tmp_path / "long.wav"
    AudioSegment.silent(duration=10_000, frame_rate=16_000).set_channels(1).export(audio, format="wav")
    original_size = audio.stat().st_size
    captured = {}

    def post(*_, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True, status_code=200, text="", json=lambda: {"results": {}})

    monkeypatch.setattr(engine.requests, "post", post)
    monkeypatch.setattr(engine, "DEEPGRAM_COMPRESS_UPLOAD_MIN_BYTES", 1)
    engine._post_deepgram_audio("key", str(audio), "en")
    assert captured["headers"]["Content-Type"] == "audio/mpeg"
    assert int(captured["headers"]["Content-Length"]) == len(captured["data"])
    assert len(captured["data"]) < original_size / 4
    assert not list(tmp_path.glob("*_deepgram_upload.mp3"))
