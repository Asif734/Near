from pathlib import Path

from app.engine_adapter import ProjectAdapter, load_engine


def test_local_video_engine_is_loadable(tmp_path: Path):
    engine = load_engine(tmp_path)
    assert Path(engine.__file__).resolve().parent.name == "video_engine"
    assert all(
        hasattr(engine, name)
        for name in (
            "get_audio",
            "get_segments_translation",
            "generate_audio_synced",
            "mix_final_audio",
            "create_subtitle",
            "burn_subtitle",
        )
    )


def test_english_male_voice_routes_to_deepgram_odysseus():
    project = ProjectAdapter({
        "id": "voice-test", "input_path": "/tmp/input.mp4", "source_language": "bn",
        "target_language": "en", "voice_gender": "male",
    })
    assert project.deepgram_tts_model == "aura-2-odysseus-en"


def test_chinese_male_voice_uses_native_chinese_voice():
    project = ProjectAdapter({
        "id": "voice-test", "input_path": "/tmp/input.mp4", "source_language": "bn",
        "target_language": "zh-CN", "voice_gender": "male",
    })
    assert project.deepgram_tts_model is None
    assert project.target_language.lang_model == "zh-CN-YunxiNeural"
