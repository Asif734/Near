LANGUAGES = [
    {"code": "en", "name": "English (US)", "voice": "en-US-EmmaNeural", "female_voice": "en-US-EmmaNeural", "male_voice": "deepgram:aura-2-odysseus-en"},
    {"code": "bn", "name": "Bangla", "voice": "bn-BD-NabanitaNeural", "female_voice": "bn-BD-NabanitaNeural", "male_voice": "bn-BD-PradeepNeural"},
    {"code": "zh-CN", "name": "Chinese", "voice": "zh-CN-XiaoxiaoNeural", "female_voice": "zh-CN-XiaoxiaoNeural", "male_voice": "zh-CN-YunxiNeural"},
    {"code": "ja", "name": "Japanese", "voice": "ja-JP-NanamiNeural", "female_voice": "ja-JP-NanamiNeural", "male_voice": "ja-JP-KeitaNeural"},
    {"code": "ms", "name": "Malay", "voice": "ms-MY-YasminNeural", "female_voice": "ms-MY-YasminNeural", "male_voice": "ms-MY-OsmanNeural"},
    {"code": "th", "name": "Thai", "voice": "th-TH-PremwadeeNeural", "female_voice": "th-TH-PremwadeeNeural", "male_voice": "th-TH-NiwatNeural"},
    {"code": "id", "name": "Indonesian", "voice": "id-ID-GadisNeural", "female_voice": "id-ID-GadisNeural", "male_voice": "id-ID-ArdiNeural"},
    {"code": "tl", "name": "Filipino", "voice": "fil-PH-BlessicaNeural", "female_voice": "fil-PH-BlessicaNeural", "male_voice": "fil-PH-AngeloNeural"},
    {"code": "pt", "name": "Portuguese (Brazil)", "voice": "pt-BR-FranciscaNeural", "female_voice": "pt-BR-FranciscaNeural", "male_voice": "pt-BR-AntonioNeural"},
]

LANGUAGE_CODES = {item["code"] for item in LANGUAGES}
