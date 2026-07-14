from app.engine_adapter import configure_engine, load_engine


def test_recording_fast_mode_keeps_timing_rewrites_and_increases_parallelism(tmp_path, monkeypatch):
    engine = load_engine(tmp_path)
    monkeypatch.setenv("VIDEO_RECORDING_FAST_MODE", "true")
    configure_engine(engine, "recording")
    assert engine.TTS_CONCURRENCY == 6
    assert engine.AUDIO_FIT_CONCURRENCY == 6
    assert engine.REWRITE_CONCURRENCY == 4
    assert engine.REWRITE_MAX_ATTEMPTS > 0
    assert engine.HARD_FIT_REWRITE_MAX_ATTEMPTS > 0
    assert engine.TRANSCRIPT_QA_ENABLED is False
    assert engine.TRANSLATION_QA_ENABLED is False

