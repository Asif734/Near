import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANGLA_WPM = 130
DEFAULT_ENGLISH_MS_PER_WORD = 360  # ~166 WPM, closer to many TTS voices
CANDIDATE_COUNT = 8
MAX_TEXT_ATTEMPTS = 2

openai_client: Optional[AsyncOpenAI] = None
TRANSLATE_MODEL = os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-5.4-mini")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_words(value: str, lang: str = "en") -> int:
    if not value or not isinstance(value, str):
        return 0
    if lang == "en":
        return len(re.findall(r"[A-Za-z0-9]+(?:[''\\-][A-Za-z0-9]+)?", value))
    return len([w for w in value.strip().split() if w])


def count_matches(value: str, pattern: str) -> int:
    return len(re.findall(pattern, value))


def estimate_bangla_reading_ms(value: str) -> int:
    clean = value.strip()
    word_count = count_words(clean, "bn")
    comma_count = count_matches(clean, r"[,،،]")
    sent_count = count_matches(clean, r"[.!?।]")

    word_ms = (word_count / BANGLA_WPM) * 60 * 1000
    comma_pause_ms = comma_count * 140
    sent_pause_ms = max(1, sent_count) * 350

    return round(word_ms + comma_pause_ms + sent_pause_ms)


def safe_json_parse(content: str) -> Optional[dict]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def extract_candidates(parsed: Optional[dict]) -> list:
    if not parsed or not isinstance(parsed.get("candidates"), list):
        return []
    result = []
    for item in parsed["candidates"]:
        if isinstance(item, str):
            result.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("trans"), str):
            result.append(item["trans"].strip())
    return [c for c in result if c]


def _get_openai_client() -> AsyncOpenAI:
    global openai_client
    if openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        openai_client = AsyncOpenAI(api_key=api_key)
    return openai_client


def _text_units(value: str) -> int:
    words = re.findall(r"[A-Za-z0-9]+(?:[''\\-][A-Za-z0-9]+)?", value)
    non_space_chars = len(re.sub(r"\s+", "", value or ""))
    return max(len(words), non_space_chars)


def _choose_timing_candidate(candidates: list[str], mode: str) -> str:
    candidates = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
    if not candidates:
        return ""
    if mode == "expand":
        return max(candidates, key=_text_units)
    if mode == "shorten":
        return min(candidates, key=_text_units)
    return candidates[0]


async def rewrite_translation_for_timing(
    *,
    source_text: str,
    current_translation: str,
    source_language_code: str,
    source_language_name: str,
    target_language_code: str,
    target_language_name: str,
    target_voice: str,
    target_ms: int,
    actual_audio_ms: int,
    attempt: int = 1,
    must_fit: bool = False,
) -> dict:
    if not source_text or not current_translation:
        return {"trans": current_translation, "action": "keep", "reason": "missing text"}

    diff_ms = actual_audio_ms - target_ms
    mode = "shorten" if diff_ms > 0 else "expand"
    action_text = "shorter and more compact" if mode == "shorten" else "longer and more detailed"
    current_units = _text_units(current_translation)
    timing_ratio = target_ms / actual_audio_ms if actual_audio_ms > 0 else 1
    target_units = current_units
    if mode == "shorten":
        target_units = max(3, math.floor(current_units * timing_ratio * 0.92))
    elif mode == "expand":
        target_units = max(current_units + 1, math.ceil(current_units * timing_ratio * 0.95))

    target_seconds = target_ms / 1000
    if mode == "shorten":
        rewrite_instruction = (
            f"Rewrite it in a short, precise manner while keeping the full meaning intact, "
            f"so the TTS can naturally speak it within about {target_seconds:.2f} seconds ({target_ms}ms)."
        )
    else:
        rewrite_instruction = (
            f"Rewrite or lengthen it naturally while keeping the same meaning, "
            f"so the TTS can fit about {target_seconds:.2f} seconds ({target_ms}ms) without sounding padded."
        )

    fit_instruction = ""
    if must_fit and mode == "shorten":
        fit_instruction = f"""
This rewrite is required because the current TTS cannot fit the segment within the allowed speed-up limit.
The next TTS probe must naturally speak within {target_seconds:.2f} seconds ({target_ms}ms), before any final speed-up.
Be short and precise: remove optional framing words, repeated phrasing, and verbose structure while preserving the meaning.
"""

    prompt = f"""
You are a strict voice-dubbing translation editor.

{rewrite_instruction}

Source language: {source_language_name} ({source_language_code})
Target language: {target_language_name} ({target_language_code})
Target TTS voice: {target_voice}

Timing:
- Target natural TTS duration: {target_ms}ms
- Current TTS duration: {actual_audio_ms}ms
- Difference: {abs(diff_ms)}ms
- Required action: make the target sentence {action_text}
- Current text length estimate: {current_units} units
- Target text length estimate for this rewrite: about {target_units} units
{fit_instruction}

Priority order:
1. Preserve confirmed source meaning.
2. Remove invented or unsupported meaning.
3. Make the wording natural for spoken dubbing.
4. Move TTS duration closer to target_ms.

Non-negotiable rules:
- Keep the same meaning as the source text.
- Use the current target translation only as a reference.
- Write only in the target language.
- Do not add unrelated facts.
- Do not remove important facts.
- Preserve every source fact, including names, places, dates, times, quantities, currencies, measurements, scores, percentages, ranges, and offers.
- If current_translation contains details not supported by source_text, remove them.
- If source_text is incomplete or ends mid-phrase, do not complete it from imagination.
- Write numbers in natural spoken words for the target language whenever they are meant to be spoken by TTS.
- Preserve culturally specific numbering units when they are natural in the target language or needed for meaning, such as lakh, crore, thousand, million, billion, or their target-language equivalents.
- Rewrite digit ranges into natural spoken ranges in the target language.
- Detect abbreviations, acronyms, initialisms, degrees, organization names, country names, technical terms, and dotted forms that the target TTS voice may pronounce with awkward pauses or as the wrong word.
- Rewrite abbreviations into the most natural spoken target-language form when needed for TTS. For unknown acronyms, prefer a TTS-friendly letter-by-letter form or a natural expansion only if the source/context clearly supports it. Do not invent an expansion.
- Remove punctuation inside abbreviations when that punctuation would cause unnatural TTS pauses, unless the abbreviation should be expanded into words.
- Do not mention timing, source language, target language, or rewriting.
- Keep it natural for spoken dubbing.
- If expanding, restore natural context/details from the source instead of adding filler.
- If shortening, compress wording without losing the main meaning. Aim near {target_units} text units.
- If shortening and the current TTS cannot fit the allowed timing window, be aggressive: remove filler, repeated phrasing, nonessential framing, unnecessary quotes, redundant transitions, and verbose explanations while preserving the core facts.
- If shortening, every candidate must be meaning-preserving but noticeably shorter than current_translation.
- If shortening, include at least three very compact candidates near {target_units} text units.
- If a segment ends mid-phrase, do not invent the next phrase just to make the sentence feel complete.
- Avoid excessive punctuation because TTS adds pauses.

Return exactly {CANDIDATE_COUNT} candidates as valid JSON:
{{
  "candidates": [
    {{ "trans": "candidate one" }}
  ]
}}
"""

    response = await _get_openai_client().chat.completions.create(
        model=os.environ.get("OPENAI_TRANSLATE_MODEL", TRANSLATE_MODEL),
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sourceText": source_text,
                        "currentTranslation": current_translation,
                        "mode": mode,
                        "targetMs": target_ms,
                        "actualAudioMs": actual_audio_ms,
                        "attempt": attempt,
                        "mustFit": must_fit,
                        "currentTextUnits": current_units,
                        "targetTextUnits": target_units,
                        "timingRatio": timing_ratio,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = (response.choices[0].message.content or "{}").strip()
    candidates = extract_candidates(safe_json_parse(raw))
    selected = _choose_timing_candidate(candidates, mode)
    return {
        "trans": selected or current_translation,
        "action": mode if selected else "keep",
        "candidates": candidates,
        "target_ms": target_ms,
        "actual_audio_ms": actual_audio_ms,
        "diff_ms": diff_ms,
    }


def get_estimated_english_ms(candidate: str, ms_per_word: int) -> int:
    clean = candidate.strip()
    words = count_words(clean, "en")
    comma_count = count_matches(clean, r"[,،،]")
    sent_count = count_matches(clean, r"[.!?]")

    return round(
        words * ms_per_word
        + comma_count * 120
        + max(1, sent_count) * 250
    )


@dataclass
class CandidateAnalysis:
    trans: str
    word_count: int
    estimated_ms: int
    estimated_diff_ms: int
    word_diff: int
    is_in_word_range: bool

    def to_dict(self):
        return {
            "trans": self.trans,
            "wordCount": self.word_count,
            "estimatedMs": self.estimated_ms,
            "estimatedDiffMs": self.estimated_diff_ms,
            "wordDiff": self.word_diff,
            "isInWordRange": self.is_in_word_range,
        }


def analyze_candidate(
        candidate: str,
        target_ms: int,
        target_words: int,
        min_words: int,
        max_words: int,
        ms_per_word: int,
) -> CandidateAnalysis:
    word_count = count_words(candidate, "en")
    estimated_ms = get_estimated_english_ms(candidate, ms_per_word)
    estimated_diff_ms = abs(estimated_ms - target_ms)
    word_diff = abs(word_count - target_words)
    is_in_word_range = min_words <= word_count <= max_words

    return CandidateAnalysis(
        trans=candidate,
        word_count=word_count,
        estimated_ms=estimated_ms,
        estimated_diff_ms=estimated_diff_ms,
        word_diff=word_diff,
        is_in_word_range=is_in_word_range,
    )


def choose_best_candidate(
        candidates: list,
        target_ms: int,
        target_words: int,
        min_words: int,
        max_words: int,
        ms_per_word: int,
        mode: str,
) -> tuple:
    analyzed = [
        analyze_candidate(c, target_ms, target_words, min_words, max_words, ms_per_word)
        for c in candidates
    ]
    in_range = [a for a in analyzed if a.is_in_word_range]
    pool = in_range if in_range else analyzed

    def sort_key(a: CandidateAnalysis):
        if mode == "expand":
            return (-a.word_count, a.word_diff, a.estimated_diff_ms)
        if mode == "shorten":
            return (a.word_count, a.word_diff, a.estimated_diff_ms)
        return (a.word_diff, a.estimated_diff_ms)

    pool.sort(key=sort_key)
    return (pool[0] if pool else None), analyzed


@dataclass
class WordPlan:
    ms_per_word: int
    target_words: int
    min_words: int
    max_words: int


def build_word_plan(
        mode: str,
        target_ms: int,
        actual_audio_ms: Optional[int],
        current_word_count: int,
        real_timing: Optional[dict],
) -> WordPlan:
    ms_per_word = DEFAULT_ENGLISH_MS_PER_WORD

    if actual_audio_ms and current_word_count > 0:
        ms_per_word = max(220, min(520, round(actual_audio_ms / current_word_count)))

    if not real_timing:
        # First generation: aim slightly high
        target_words = math.ceil(target_ms / ms_per_word) + 1
        return WordPlan(
            ms_per_word=ms_per_word,
            target_words=target_words,
            min_words=max(3, target_words),
            max_words=target_words + 3,
        )

    exact_words_needed = math.ceil(target_ms / ms_per_word)

    if real_timing["isTooShort"]:
        extra = math.ceil(real_timing["shortageMs"] / ms_per_word)
        target_words = max(exact_words_needed, current_word_count + extra)
        if real_timing["shortageMs"] >= 1500:
            target_words += 1
        if real_timing["shortageMs"] >= 3000:
            target_words += 2
        return WordPlan(
            ms_per_word=ms_per_word,
            target_words=target_words,
            min_words=max(3, target_words),
            max_words=target_words + 3,
        )

    if real_timing["isTooLong"]:
        words_to_remove = math.ceil(real_timing["overflowMs"] / ms_per_word)
        target_words = min(exact_words_needed, max(3, current_word_count - words_to_remove))
        return WordPlan(
            ms_per_word=ms_per_word,
            target_words=target_words,
            min_words=max(3, target_words - 2),
            max_words=max(3, target_words),
        )

    target_words = exact_words_needed
    return WordPlan(
        ms_per_word=ms_per_word,
        target_words=target_words,
        min_words=max(3, target_words - 1),
        max_words=target_words + 1,
    )


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

async def call_openai_for_candidates(
        *,
        mode: str,
        target_ms: int,
        target_seconds: float,
        target_words: int,
        min_words: int,
        max_words: int,
        real_timing: Optional[dict],
        attempt: int,
        # request context
        target_mode: str,
        tolerance_ms: int,
        trans: str,
        actual_audio_ms: Optional[int],
) -> list:
    mode_instruction = ""

    if mode == "initial":
        mode_instruction = f"""
Create a fresh English dubbing translation from the Bangla source.
Do not simply edit the old English translation.
The old English translation is only a rough meaning reference.
The result must be {min_words} to {max_words} words.
Aim for exactly {target_words} words.
"""
    elif mode == "expand":
        mode_instruction = f"""
The current English TTS audio is TOO SHORT.

Current English translation:
"{trans}"

Current real audio duration: {actual_audio_ms}ms
Target duration: {target_ms}ms
Short by: {real_timing["shortageMs"]}ms

Rewrite it LONGER while keeping the same meaning.
The result must be {min_words} to {max_words} words.
Aim for exactly {target_words} words.

Do not add random filler.
Expand only by restoring natural details from the Bangla source.
"""
    elif mode == "shorten":
        mode_instruction = f"""
The current English TTS audio is TOO LONG.

Current English translation:
"{trans}"

Current real audio duration: {actual_audio_ms}ms
Target duration: {target_ms}ms
Overflow by: {real_timing["overflowMs"]}ms

Rewrite it SHORTER while keeping the same meaning.
The result must be {min_words} to {max_words} words.
Aim for exactly {target_words} words.

Do not remove important meaning.
"""

    prompt = f"""
You are a professional Bangla to English voice-dubbing translation editor.

Goal:
Create an English translation that keeps the same meaning and reduces duration distance.

Target:
- Target mode: {target_mode}
- Target duration: {target_ms}ms
- Target seconds: {target_seconds}s
- Tolerance goal: ±{tolerance_ms}ms
- Required word count: {min_words} to {max_words} words
- Best word count: exactly {target_words} words

Strict rules:
- Translate from the Bangla source.
- Use the old/current English only as a reference.
- Keep the same meaning and expression.
- Do not add unrelated facts.
- Do not remove important facts.
- Preserve every source fact, including names, places, dates, times, quantities, currencies, measurements, scores, percentages, ranges, and offers.
- If the old/current English contains details not supported by the Bangla source, remove them.
- If the Bangla text is incomplete or ends mid-phrase, do not invent the missing ending.
- Write numbers in natural spoken English for TTS whenever they are meant to be spoken.
- Preserve culturally specific numbering units when they are natural in English or needed for meaning, such as lakh, crore, thousand, million, and billion.
- Rewrite digit ranges into natural spoken ranges, for example "10,000" as "ten thousand" and "8 to 10 lakh" as "eight lakh to ten lakh".
- Detect abbreviations, acronyms, initialisms, degrees, organization names, country names, technical terms, and dotted forms that TTS may pronounce with awkward pauses or as the wrong word.
- Rewrite abbreviations into the most natural spoken English form when needed for TTS. Examples: "U.S." as "United States" when it means the country, "U.N." as "United Nations", "A.I." as "AI", and "MBBS" as "M B B S" if the credential is normally spoken letter by letter.
- For unknown acronyms, prefer a TTS-friendly letter-by-letter form or a natural expansion only if the Bangla source/context clearly supports it. Do not invent an expansion.
- Remove punctuation inside abbreviations when that punctuation would cause unnatural TTS pauses, unless the abbreviation should be expanded into words.
- Natural spoken English for dubbing.
- If shortening, be compact and remove filler, repeated phrasing, nonessential framing, unnecessary quotes, redundant transitions, and verbose explanations while preserving core source facts.
- Do not return a candidate below {min_words} words.
- Do not return a candidate above {max_words} words.
- If the Bangla text is incomplete, translate only what exists.
- Avoid too many commas because TTS adds pauses.

Timing rules:
- Do not only make it fit under the slot.
- It must try to get close to the target duration.
- If current audio is too short, expand the sentence.
- If current audio is too long, shorten the sentence.

{mode_instruction}

Attempt number: {attempt}

Return exactly {CANDIDATE_COUNT} candidates.
Return only valid JSON.

JSON format:
{{
  "candidates": [
    {{ "trans": "candidate one" }},
    {{ "trans": "candidate two" }},
    {{ "trans": "candidate three" }},
    {{ "trans": "candidate four" }},
    {{ "trans": "candidate five" }},
    {{ "trans": "candidate six" }},
    {{ "trans": "candidate seven" }},
    {{ "trans": "candidate eight" }}
  ]
}}
"""

    response = await _get_openai_client().chat.completions.create(
        model=TRANSLATE_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps({
                    "targetMode": target_mode,
                    "targetMs": target_ms,
                    "targetSeconds": target_seconds,
                    "toleranceMs": tolerance_ms,
                    "mode": mode,
                    "actualAudioMs": actual_audio_ms,
                    "realTiming": real_timing,
                    "requiredWords": {
                        "min": min_words,
                        "max": max_words,
                        "target": target_words,
                    },
                }),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = (response.choices[0].message.content or "{}").strip()
    parsed = safe_json_parse(raw)
    return extract_candidates(parsed)


# ---------------------------------------------------------------------------
# Core async logic (reusable outside HTTP if needed)
# ---------------------------------------------------------------------------

async def _process(
        start: float,
        end: float,
        text: str,
        trans: str,
        target_mode: str,
        tolerance_ms: int,
        actual_audio_ms: Optional[int],
) -> dict:
    slot_ms = round((end - start) * 1000)
    segment_duration = round(slot_ms / 1000, 2)
    source_text_reading_ms = estimate_bangla_reading_ms(text)

    target_ms = source_text_reading_ms if target_mode == "sourceText" else slot_ms
    target_seconds = round(target_ms / 1000, 2)

    current_word_count = count_words(trans, "en")

    mode = "initial"
    real_timing = None

    if actual_audio_ms is not None:
        diff_ms = abs(actual_audio_ms - target_ms)
        real_timing = {
            "actualAudioMs": actual_audio_ms,
            "targetMs": target_ms,
            "diffMs": diff_ms,
            "shortageMs": max(0, target_ms - actual_audio_ms),
            "overflowMs": max(0, actual_audio_ms - target_ms),
            "isMatched": diff_ms <= tolerance_ms,
            "isTooShort": actual_audio_ms < target_ms - tolerance_ms,
            "isTooLong": actual_audio_ms > target_ms + tolerance_ms,
        }

        if real_timing["isMatched"]:
            return {
                "status": 200,
                "data": {
                    "start": start,
                    "end": end,
                    "text": text,
                    "oldTrans": trans,
                    "trans": trans,
                    "action": "keep",
                    "message": "Current audio already matches the target duration.",
                    "timing": {
                        "targetMode": target_mode,
                        "slotMs": slot_ms,
                        "segmentDuration": segment_duration,
                        "sourceTextReadingMs": source_text_reading_ms,
                        "targetMs": target_ms,
                        "targetSeconds": target_seconds,
                        "toleranceMs": tolerance_ms,
                        "currentWordCount": current_word_count,
                        **real_timing,
                        "needsAudioCheck": False,
                    },
                },
            }

        mode = "expand" if real_timing["isTooShort"] else "shorten"

    word_plan = build_word_plan(
        mode=mode,
        target_ms=target_ms,
        actual_audio_ms=actual_audio_ms,
        current_word_count=current_word_count,
        real_timing=real_timing,
    )

    best_overall: Optional[CandidateAnalysis] = None
    analyzed_overall: list = []

    for attempt in range(1, MAX_TEXT_ATTEMPTS + 1):
        candidates = await call_openai_for_candidates(
            mode=mode,
            target_ms=target_ms,
            target_seconds=target_seconds,
            target_words=word_plan.target_words,
            min_words=word_plan.min_words,
            max_words=word_plan.max_words,
            real_timing=real_timing,
            attempt=attempt,
            target_mode=target_mode,
            tolerance_ms=tolerance_ms,
            trans=trans,
            actual_audio_ms=actual_audio_ms,
        )

        if not candidates:
            continue

        best, analyzed = choose_best_candidate(
            candidates,
            target_ms,
            word_plan.target_words,
            word_plan.min_words,
            word_plan.max_words,
            word_plan.ms_per_word,
            mode,
        )

        analyzed_overall = analyzed

        if best and (best_overall is None or best.word_diff < best_overall.word_diff):
            best_overall = best

        if best and best.is_in_word_range:
            break

    if best_overall is None:
        return {
            "status": 422,
            "error": "Model returned no valid candidates.",
            "data": {
                "start": start,
                "end": end,
                "text": text,
                "oldTrans": trans,
                "targetMode": target_mode,
                "mode": mode,
                "slotMs": slot_ms,
                "targetMs": target_ms,
                "targetSeconds": target_seconds,
                "toleranceMs": tolerance_ms,
                "actualAudioMs": actual_audio_ms,
                "realTiming": real_timing,
                "wordPlan": {
                    "msPerWord": word_plan.ms_per_word,
                    "targetWords": word_plan.target_words,
                    "minWords": word_plan.min_words,
                    "maxWords": word_plan.max_words,
                },
            },
        }

    return {
        "status": 200,
        "data": {
            "start": start,
            "end": end,
            "text": text,
            "oldTrans": trans,
            "trans": best_overall.trans,
            "action": "generate" if mode == "initial" else mode,
            "message": (
                "Generated translation. Now generate TTS audio and measure actual duration."
                if mode == "initial"
                else "Generated corrected translation. Regenerate TTS audio and measure again."
            ),
            "timing": {
                "targetMode": target_mode,
                "slotMs": slot_ms,
                "segmentDuration": segment_duration,
                "sourceTextReadingMs": source_text_reading_ms,
                "targetMs": target_ms,
                "targetSeconds": target_seconds,
                "toleranceMs": tolerance_ms,
                "mode": mode,
                "actualAudioMs": actual_audio_ms,
                "realTiming": real_timing,
                "currentWordCount": current_word_count,
                "estimatedMsPerWordUsed": word_plan.ms_per_word,
                "targetEnglishWords": word_plan.target_words,
                "minWords": word_plan.min_words,
                "maxWords": word_plan.max_words,
                "selectedWordCount": best_overall.word_count,
                "selectedEstimatedMs": best_overall.estimated_ms,
                "selectedEstimatedDiffMs": best_overall.estimated_diff_ms,
                "selectedIsInWordRange": best_overall.is_in_word_range,
                "needsAudioCheck": True,
            },
            "candidates": [a.to_dict() for a in analyzed_overall],
        },
    }


# ---------------------------------------------------------------------------
# Django view
# ---------------------------------------------------------------------------

# async def sync_trans_reading_time(text, trans, start, end, actual_audio_ms):
#     start = start
#     end = end
#     text = text
#     trans = trans
#     target_mode = "segment"
#     tolerance_ms = 500
#     actual_audio_ms = actual_audio_ms
#
#     # --- Validation ---
#     if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
#         return ({"error": "start and end must be numbers."}, )
#
#     if not text or not trans:
#         return ({"error": "start, end, text, and trans are required."}, )
#
#     if end <= start:
#         return ({"error": "end time must be greater than start time."})
#
#     if target_mode not in ("segment", "sourceText"):
#         return (
#             {"error": 'targetMode must be either "segment" or "sourceText".'},
#
#         )
#
#     if actual_audio_ms is not None and (
#             not isinstance(actual_audio_ms, (int, float)) or actual_audio_ms <= 0
#     ):
#         return (
#             {"error": "actualAudioMs must be a positive number when provided."},
#
#         )
#
#     # --- Run async logic ---
#     try:
#         result = asyncio.run(
#             _process(
#                 start=start,
#                 end=end,
#                 text=text,
#                 trans=trans,
#                 target_mode=target_mode,
#                 tolerance_ms=tolerance_ms,
#                 actual_audio_ms=actual_audio_ms,
#             )
#         )
#     except Exception as e:
#         return ({"error": "Internal server error.", "message": str(e)}, )
#
#     status = result.pop("status")
#     print(result)
#     return result

async def sync_trans_reading_time(text, trans, start, end, actual_audio_ms):
    target_mode = "segment"
    tolerance_ms = 350

    # --- Validation ---
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return {"error": "start and end must be numbers."}

    if not text or not trans:
        return {"error": "start, end, text, and trans are required."}

    if end <= start:
        return {"error": "end time must be greater than start time."}

    if actual_audio_ms is not None and (
        not isinstance(actual_audio_ms, (int, float)) or actual_audio_ms <= 0
    ):
        return {"error": "actualAudioMs must be a positive number when provided."}

    # --- Run async logic ---
    try:
        print(
            f"""
        start={start}
        end={end}
        text={text}
        trans={trans}
        target_mode={target_mode}
        tolerance_ms={tolerance_ms}
        actual_audio_ms={actual_audio_ms}
        """
        )
        result = await _process(
            start=start,
            end=end,
            text=text,
            trans=trans,
            target_mode=target_mode,
            tolerance_ms=tolerance_ms,
            actual_audio_ms=actual_audio_ms,
        )
        print(result)
    except Exception as e:
        return {"error": "Internal server error.", "message": str(e)}

    result.pop("status", None)
    return result

# class Lines:
#     def __init__(self, start, end, text, trans):
#         self.start = start
#         self.end = end
#         self.text = text
#         self.trans = trans
#
#
# def load_segments(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     segments = [Lines(**item) for item in data]
#     return segments
#
#
#
# segments = load_segments("151_segments.json")
#
#
# new_audio = 5592
# for segment in segments:
#     ok = sync_trans_reading_time(segment.text, segment.trans, segment.start,segment.end,new_audio)
#     print('Text')
#     print(ok['data']['text'])
#     print("Old transcription")
#     print(ok['data']['oldTrans'])
#     print("New transcription")
#     print(ok['data']['trans'])
#     break


