# multimodal/audio.py
#
# What this file does:
#   1. Takes an uploaded audio file (mp3/wav/m4a/ogg/webm)
#   2. Runs Whisper (base model) to transcribe speech → text
#   3. Converts math-specific phrases to proper math notation
#      e.g. "x squared" → "x**2", "square root of x" → "sqrt(x)"
#   4. Calculates confidence from Whisper's no_speech_prob
#   5. Returns transcript + confidence + HITL flag
#
# Used by: app.py when user selects "Audio" input mode
# HITL triggers when: confidence < ASR_CONFIDENCE_THRESHOLD (0.6)
#
# Note: Whisper model is loaded once and cached by app.py using
#       @st.cache_resource to avoid reloading on every audio upload

import re
import whisper

# Confidence threshold for audio — slightly lower than OCR (0.7)
# because Whisper is generally more reliable than OCR on math text
ASR_CONFIDENCE_THRESHOLD = 0.6

# ─────────────────────────────────────────────────────────────
# Math phrase → notation mapping
# Handles common spoken math phrases JEE students would say
# ─────────────────────────────────────────────────────────────

MATH_PHRASE_MAP = [

    # Powers and roots
    (r"x squared",              "x**2"),
    (r"x cubed",                "x**3"),
    (r"(\w+) squared",          r"\1**2"),
    (r"(\w+) cubed",            r"\1**3"),
    (r"(\w+) raised to (\w+)",  r"\1**\2"),
    (r"square root of (\w+)",   r"sqrt(\1)"),
    (r"cube root of (\w+)",     r"cbrt(\1)"),

    # Calculus
    (r"integral of",            "integrate"),
    (r"find the integral",      "integrate"),
    (r"find integral",          "integrate"),
    (r"derivative of",          "differentiate"),
    (r"d by d(\w+)",            r"d/d\1"),
    (r"limit as (\w+) tends to","limit \1 ->"),

    # Arithmetic spoken words
    (r"\bplus\b",               "+"),
    (r"\bminus\b",              "-"),
    (r"\btimes\b",              "*"),
    (r"\bdivided by\b",         "/"),
    (r"\bequals\b",             "="),
    (r"\bto the power of\b",    "**"),

    # Greek letters commonly used in JEE
    (r"\btheta\b",              "θ"),
    (r"\balpha\b",              "α"),
    (r"\bbeta\b",               "β"),
    (r"\bpi\b",                 "π"),
]


def clean_math_phrases(text: str) -> str:
    """
    Replace spoken math phrases with proper math notation.
    Applies all rules in MATH_PHRASE_MAP in order.
    """

    cleaned = text.lower().strip()

    for pattern, replacement in MATH_PHRASE_MAP:
        cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned


# ─────────────────────────────────────────────────────────────
# Main ASR function
# ─────────────────────────────────────────────────────────────

def run_asr(audio_path: str, model: whisper.Whisper) -> dict:
    """
    Transcribe audio file using Whisper and clean math phrases.

    Args:
        audio_path : path to the saved audio file
        model      : loaded Whisper model (passed in from app.py cache)

    Returns:
        {
            "transcript":      "cleaned transcription text",
            "raw_transcript":  "original whisper output before cleaning",
            "confidence":      0.85,
            "needs_hitl":      False,
            "language":        "en"
        }
    """

    try:
        # Transcribe — Whisper handles format detection automatically
        result = model.transcribe(audio_path)

        raw_transcript = result.get("text", "").strip()
        language       = result.get("language", "unknown")

        # No speech detected at all
        if not raw_transcript:
            return {
                "transcript"    : "",
                "raw_transcript": "",
                "confidence"    : 0.0,
                "needs_hitl"    : True,
                "language"      : language
            }

        # Calculate confidence from segments
        # no_speech_prob = probability audio is silence/unclear
        # confidence = 1 - average no_speech_prob across all segments
        segments = result.get("segments", [])

        if segments:
            avg_no_speech = sum(s["no_speech_prob"] for s in segments) / len(segments)
            confidence = round(1.0 - avg_no_speech, 2)
        else:
            # No segments means Whisper wasn't sure — treat as low confidence
            confidence = 0.5

        # Clean math phrases → proper notation
        cleaned_transcript = clean_math_phrases(raw_transcript)

        # Trigger HITL if confidence too low
        needs_hitl = confidence < ASR_CONFIDENCE_THRESHOLD

        return {
            "transcript"    : cleaned_transcript,
            "raw_transcript": raw_transcript,       # shown to user for confirmation
            "confidence"    : confidence,
            "needs_hitl"    : needs_hitl,
            "language"      : language
        }

    except Exception as e:
        return {
            "transcript"    : "",
            "raw_transcript": "",
            "confidence"    : 0.0,
            "needs_hitl"    : True,
            "language"      : "unknown",
            "error"         : str(e)
        }