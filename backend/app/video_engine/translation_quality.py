import asyncio
import json
import os
import re
from typing import Any, Optional

from openai import AsyncOpenAI


TRANSLATION_QA_MODEL = os.environ.get("OPENAI_TRANSLATION_QA_MODEL") or os.environ.get(
    "OPENAI_TRANSLATE_MODEL",
    "gpt-5.4-mini",
)

openai_client: Optional[AsyncOpenAI] = None


def _get_openai_client() -> AsyncOpenAI:
    global openai_client
    if openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        openai_client = AsyncOpenAI(api_key=api_key)
    return openai_client


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _word_parts(value: str) -> list[str]:
    return [part for part in _normalize_text(value).split() if part]


def _contains_word_sequence(value: str, sequence: list[str]) -> bool:
    if len(sequence) < 3:
        return False
    return " ".join(sequence) in _normalize_text(value)


def _is_safe_transcript_correction(
    *,
    original: str,
    corrected: str,
    previous_text: str,
    next_text: str,
) -> bool:
    original = _normalize_text(original)
    corrected = _normalize_text(corrected)
    if not original or not corrected:
        return False
    if corrected == original:
        return True

    original_words = _word_parts(original)
    corrected_words = _word_parts(corrected)
    if len(corrected_words) > len(original_words) + max(2, round(len(original_words) * 0.2)):
        return False

    previous_words = _word_parts(previous_text)
    next_words = _word_parts(next_text)
    if _contains_word_sequence(corrected, previous_words[-5:]) and not _contains_word_sequence(original, previous_words[-5:]):
        return False
    if _contains_word_sequence(corrected, next_words[:5]) and not _contains_word_sequence(original, next_words[:5]):
        return False

    return True


def _safe_json_parse(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content or "")
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _clean_review_item(item: Any, fallback_index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"index": fallback_index, "status": "keep", "corrected_trans": "", "issues": ["invalid QA item"]}

    try:
        index = int(item.get("index", fallback_index))
    except (TypeError, ValueError):
        index = fallback_index

    status = str(item.get("status", "keep")).strip().lower()
    if status not in {"keep", "fix"}:
        status = "keep"

    issues = item.get("issues", [])
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = []

    return {
        "index": index,
        "status": status,
        "corrected_trans": _normalize_text(str(item.get("corrected_trans", ""))),
        "issues": [_normalize_text(str(issue)) for issue in issues if _normalize_text(str(issue))],
        "confidence": item.get("confidence"),
    }


def _clean_transcript_review_item(item: Any, fallback_index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"index": fallback_index, "status": "keep", "corrected_text": "", "issues": ["invalid QA item"]}

    try:
        index = int(item.get("index", fallback_index))
    except (TypeError, ValueError):
        index = fallback_index

    status = str(item.get("status", "keep")).strip().lower()
    if status not in {"keep", "fix"}:
        status = "keep"

    issues = item.get("issues", [])
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = []

    return {
        "index": index,
        "status": status,
        "corrected_text": _normalize_text(str(item.get("corrected_text", ""))),
        "issues": [_normalize_text(str(issue)) for issue in issues if _normalize_text(str(issue))],
        "confidence": item.get("confidence"),
    }


def _review_item_from_segment(
    segments: list[dict[str, Any]],
    index: int,
    review_index: int,
) -> dict[str, str | int]:
    segment = segments[index]
    previous_segment = segments[index - 1] if index > 0 else {}
    next_segment = segments[index + 1] if index < len(segments) - 1 else {}
    return {
        "index": review_index,
        "previous_source_text": _normalize_text(str(previous_segment.get("text", ""))),
        "source_text": _normalize_text(str(segment.get("text", ""))),
        "next_source_text": _normalize_text(str(next_segment.get("text", ""))),
        "translation": _normalize_text(str(segment.get("trans", ""))),
    }


def _transcript_review_item_from_segment(
    segments: list[dict[str, Any]],
    index: int,
    review_index: int,
) -> dict[str, str | int]:
    segment = segments[index]
    previous_segment = segments[index - 1] if index > 0 else {}
    next_segment = segments[index + 1] if index < len(segments) - 1 else {}
    return {
        "index": review_index,
        "previous_text": _normalize_text(str(previous_segment.get("text", ""))),
        "text": _normalize_text(str(segment.get("text", ""))),
        "next_text": _normalize_text(str(next_segment.get("text", ""))),
    }


async def review_source_transcript_batch(
    segments: list[dict[str, Any]],
    *,
    source_language_code: str,
    source_language_name: str = "",
) -> list[dict[str, Any]]:
    items = []
    for index, segment in enumerate(segments):
        if segment.get("previous_text") or segment.get("next_text"):
            item = {
                "index": index,
                "previous_text": _normalize_text(str(segment.get("previous_text", ""))),
                "text": _normalize_text(str(segment.get("text", ""))),
                "next_text": _normalize_text(str(segment.get("next_text", ""))),
            }
        else:
            item = _transcript_review_item_from_segment(segments, index, index)
        if item["text"]:
            items.append(item)

    if not items:
        return []

    prompt = f"""
You are a strict source-transcript QA editor for a video dubbing pipeline.

Review and correct only clear speech-recognition mistakes before translation.

Source language: {source_language_name or source_language_code} ({source_language_code})

Rules:
- For every item, review text directly.
- Use previous_text and next_text only to understand context. Do not copy, move, append, prepend, or merge words from previous_text or next_text into corrected_text.
- Correct obvious spelling, punctuation, malformed words, broken numbers, wrong number values, and missing words only when the correction is strongly supported by the current or neighboring text.
- Pay special attention to numbers and quantities. For example, if the transcript says "18" but context clearly means "118", return a fix.
- Preserve the original source language. Do not translate.
- Preserve names, places, dates, times, quantities, currencies, measurements, scores, percentages, ranges, and offers.
- Do not invent facts, names, numbers, places, endings, or missing context.
- Do not rewrite for style. Keep the speaker's wording as much as possible.
- Keep corrected_text inside the current segment only. If the current text ends mid-phrase, do not complete it using the next segment.
- Do not combine one segment's sentence with another segment's sentence.
- Do not make corrected_text much longer than text.
- If the text is plausible or the correction is uncertain, return status "keep".
- If text is incomplete because the segment cuts off mid-phrase, do not invent the missing ending.
- If status is "fix", corrected_text must contain the complete corrected source transcript for this item.
- Do not mention QA, transcription, or these rules in corrected_text.

Number and quantity rules:
- Pay highest attention to numbers and quantities.
- Check integers, decimals, percentages, currencies, dates, years, measurements, scores, ranges, and quantities.
- Preserve names, places, dates, times, quantities, currencies, measurements, scores, percentages, ranges, and offers.
- Never change a number because it seems more realistic.
- Never normalize a number unless the correction is clearly supported.
- If the transcript says "18" but context clearly means "118", return a fix.
- If the intended number is uncertain, return "keep".

Return valid JSON only:
{{
  "items": [
    {{
      "index": 0,
      "status": "keep",
      "corrected_text": "",
      "issues": [],
      "confidence": 0.0
    }}
  ]
}}
"""

    response = await _get_openai_client().chat.completions.create(
        model=TRANSLATION_QA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Correct only clear ASR transcript mistakes before translation. Return keep when uncertain.",
                        "items": items,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = _safe_json_parse(response.choices[0].message.content or "{}")
    raw_items = parsed.get("items", [])
    if not isinstance(raw_items, list):
        return []
    return [_clean_transcript_review_item(item, fallback_index=i) for i, item in enumerate(raw_items)]


async def apply_source_transcript_review(
    segments: list[dict[str, Any]],
    *,
    source_language_code: str,
    source_language_name: str = "",
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    reviewed = [{**segment} for segment in segments]
    for start in range(0, len(reviewed), max(1, batch_size)):
        batch = reviewed[start : start + batch_size]
        contextual_batch = [
            _transcript_review_item_from_segment(reviewed, global_index, local_index)
            for local_index, global_index in enumerate(range(start, start + len(batch)))
        ]
        reviews = await review_source_transcript_batch(
            contextual_batch,
            source_language_code=source_language_code,
            source_language_name=source_language_name,
        )
        for review in reviews:
            local_index = review.get("index", -1)
            if not isinstance(local_index, int) or not 0 <= local_index < len(batch):
                continue
            segment = reviewed[start + local_index]
            corrected = _normalize_text(review.get("corrected_text", ""))
            if review.get("status") == "fix" and corrected:
                previous_segment = reviewed[start + local_index - 1] if start + local_index > 0 else {}
                next_segment = reviewed[start + local_index + 1] if start + local_index < len(reviewed) - 1 else {}
                is_safe = _is_safe_transcript_correction(
                    original=str(segment.get("text", "")),
                    corrected=corrected,
                    previous_text=str(previous_segment.get("text", "")),
                    next_text=str(next_segment.get("text", "")),
                )
                if is_safe:
                    segment["transcript_qa_original_text"] = segment.get("text", "")
                    segment["text"] = corrected
                    segment["transcript_qa_status"] = "fixed"
                else:
                    segment["transcript_qa_status"] = "rejected"
                    segment["transcript_qa_rejected_text"] = corrected
                    segment["transcript_qa_issues"] = [
                        "Rejected source transcript correction because it appears to move or merge neighboring segment text."
                    ]
            elif review.get("issues"):
                segment["transcript_qa_status"] = "needs_fix"
            else:
                segment["transcript_qa_status"] = "kept"
            if review.get("issues") and segment.get("transcript_qa_status") != "rejected":
                segment["transcript_qa_issues"] = review["issues"]
            if review.get("confidence") is not None:
                segment["transcript_qa_confidence"] = review["confidence"]
        await asyncio.sleep(0)

    return reviewed


async def review_translation_batch(
    segments: list[dict[str, Any]],
    *,
    source_language_code: str,
    target_language_code: str,
    source_language_name: str = "",
    target_language_name: str = "",
) -> list[dict[str, Any]]:
    items = []
    for index, segment in enumerate(segments):
        if segment.get("source_text") or segment.get("translation"):
            item = {
                "index": index,
                "previous_source_text": _normalize_text(str(segment.get("previous_source_text", ""))),
                "source_text": _normalize_text(str(segment.get("source_text", ""))),
                "next_source_text": _normalize_text(str(segment.get("next_source_text", ""))),
                "translation": _normalize_text(str(segment.get("translation", ""))),
            }
        else:
            item = _review_item_from_segment(segments, index, index)
        if item["source_text"] and item["translation"]:
            items.append(item)

    if not items:
        return []

    prompt = f"""
You are a strict bilingual translation auditor and dubbing editor.

Your job is to decide whether each target translation is faithful, natural for spoken dubbing, and ready for TTS.

Source language: {source_language_name or source_language_code} ({source_language_code})
Target language: {target_language_name or target_language_code} ({target_language_code})

Decision standard:
- Return "keep" only when the translation is fully faithful, has no invented content, preserves all important details, and is TTS-ready.
- Return "fix" for any missing meaning, extra meaning, wrong number, wrong entity, wrong unit, wrong named item, vague summary, awkward mistranslation, incomplete translation, or TTS-unfriendly wording.
- If status is "fix", corrected_trans must be the complete corrected target-language translation for the current item.

Source comparison rules:
- Compare source_text and translation directly before deciding.
- Check every sentence, clause, phrase, named item, number, unit, offer, date, time, place, score, percentage, range, and currency.
- Do not omit source meaning. Do not add meaning that is not supported by source_text.
- Treat fluent but inaccurate translations as failures.
- Treat vague summaries as failures when source_text contains concrete facts.
- When fixing, translate from source_text, not from the flawed translation.

Neighboring context rules:
- previous_source_text and next_source_text are context only.
- Use neighboring context to resolve pronouns, carryover words, split segment boundaries, and likely ASR/entity mistakes.
- Do not copy facts from neighbors into corrected_trans unless the current source_text clearly depends on that neighboring word, number, unit, object, or speaker role.
- If current source_text starts mid-phrase and previous_source_text contains the missing required number/unit/object, use it only to translate the current fragment correctly. Example: previous ends with "fifteen hundred" and current starts with "taka item"; current may say "a fifteen hundred taka item".
- Do not invent an ending when source_text cuts off mid-phrase. Translate only confirmed meaning and keep the ending conservative.
- Correct proper nouns clearly and accurately, especially place names, human names, organization names, brand names, and other named entities.

Entity and ASR caution:
- Do not promote unclear ASR words into proper names or places unless current or neighboring context confirms them.
- If a possible name/place appears only once, is phonetically suspicious, or makes the sentence less coherent, translate the meaning generically instead of inventing a proper noun.
- Preserve confirmed names, numbers, places, and event details.

TTS readiness rules:
- Numbers meant to be spoken must be written in natural spoken words in the target language, not raw digits.
- Preserve culturally specific numbering units when natural or needed, such as lakh, crore, thousand, million, billion, or target-language equivalents.
- Convert digit ranges into spoken ranges. Example for English: "10,000" -> "ten thousand"; "8 to 10 lakh" -> "eight lakh to ten lakh".
- Detect abbreviations, acronyms, initialisms, degrees, organization names, country names, technical terms, and dotted forms that TTS may pronounce with awkward pauses or as the wrong word.
- Rewrite abbreviations into the most natural spoken target-language form when needed. Examples for English: "U.S." -> "United States" when it means the country; "U.N." -> "United Nations"; "A.I." -> "AI"; "MBBS" -> "M B B S" if normally spoken letter by letter.
- For unknown acronyms, prefer a TTS-friendly letter-by-letter form or a natural expansion only if source/context clearly supports it. Do not invent an expansion.
- Remove punctuation inside abbreviations when it would cause unnatural TTS pauses, unless the abbreviation should be expanded into words.
- Avoid excessive punctuation that creates unnatural TTS pauses.

Style rules:
- Use natural spoken target-language wording for dubbing.
- Do not make corrected_trans more verbose than needed.
- Do not mention QA, transcription, source language, target language, or these rules in corrected_trans.

Return valid JSON only:
{{
  "items": [
    {{
      "index": 0,
      "status": "keep",
      "corrected_trans": "",
      "issues": [],
      "confidence": 0.0
    }}
  ]
}}
"""

    response = await _get_openai_client().chat.completions.create(
        model=TRANSLATION_QA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Compare each source_text with its translation using neighboring source context only for disambiguation. Return a fix whenever the translation misses, changes, summarizes, invents entities, invents endings, or adds meaning.",
                        "items": items,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = _safe_json_parse(response.choices[0].message.content or "{}")
    raw_items = parsed.get("items", [])
    if not isinstance(raw_items, list):
        return []
    return [_clean_review_item(item, fallback_index=i) for i, item in enumerate(raw_items)]


async def review_single_translation(
    *,
    source_text: str,
    translation: str,
    source_language_code: str,
    target_language_code: str,
    source_language_name: str = "",
    target_language_name: str = "",
) -> dict[str, Any]:
    reviews = await review_translation_batch(
        [{"text": source_text, "trans": translation}],
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        source_language_name=source_language_name,
        target_language_name=target_language_name,
    )
    return reviews[0] if reviews else {"index": 0, "status": "keep", "corrected_trans": "", "issues": []}


async def apply_translation_review(
    segments: list[dict[str, Any]],
    *,
    source_language_code: str,
    target_language_code: str,
    source_language_name: str = "",
    target_language_name: str = "",
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    reviewed = [{**segment} for segment in segments]
    for start in range(0, len(reviewed), max(1, batch_size)):
        batch = reviewed[start : start + batch_size]
        contextual_batch = [
            _review_item_from_segment(reviewed, global_index, local_index)
            for local_index, global_index in enumerate(range(start, start + len(batch)))
        ]
        reviews = await review_translation_batch(
            contextual_batch,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            source_language_name=source_language_name,
            target_language_name=target_language_name,
        )
        for review in reviews:
            local_index = review.get("index", -1)
            if not isinstance(local_index, int) or not 0 <= local_index < len(batch):
                continue
            segment = reviewed[start + local_index]
            corrected = _normalize_text(review.get("corrected_trans", ""))
            if review.get("status") == "fix" and corrected:
                segment["qa_original_trans"] = segment.get("trans", "")
                segment["trans"] = corrected
                segment["qa_status"] = "fixed"
            elif review.get("issues"):
                segment["qa_status"] = "needs_fix"
            else:
                segment["qa_status"] = "kept"
            if review.get("issues"):
                segment["qa_issues"] = review["issues"]
            if review.get("confidence") is not None:
                segment["qa_confidence"] = review["confidence"]
        await asyncio.sleep(0)

    return reviewed

