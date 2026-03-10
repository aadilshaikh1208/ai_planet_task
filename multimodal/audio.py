import re
import whisper

# Trigger HITL when confidence drops below this
ASR_CONFIDENCE_THRESHOLD = 0.6


# Spoken math phrases mapped to proper notation
# Whisper transcribes "x squared" as text — we convert it to x**2
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
    cleaned = text.lower().strip()

    for pattern, replacement in MATH_PHRASE_MAP:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned


# ─────────────────────────────────────────────────────────────
# Main ASR function
# ─────────────────────────────────────────────────────────────

def run_asr(audio_path: str, model: whisper.Whisper) -> dict:
    """
    Transcribe audio using Whisper and convert spoken math phrases to notation.
    Whisper model is loaded once in app.py and passed in here.
    """

    try:
        result = model.transcribe(audio_path)

        raw_transcript = result.get("text", "").strip()
        language       = result.get("language", "unknown")

        if not raw_transcript:
            return {
                "transcript"    : "",
                "raw_transcript": "",
                "confidence"    : 0.0,
                "needs_hitl"    : True,
                "language"      : language
            }

       # Confidence = 1 - avg no_speech_prob across segments
        # no_speech_prob is Whisper's estimate of silence/unclear audio
        segments = result.get("segments", [])

        if segments:
            avg_no_speech = sum(s["no_speech_prob"] for s in segments) / len(segments)
            confidence = round(1.0 - avg_no_speech, 2)
        else:
            confidence = 0.5

        cleaned_transcript = clean_math_phrases(raw_transcript)

        needs_hitl = confidence < ASR_CONFIDENCE_THRESHOLD

        return {
            "transcript"    : cleaned_transcript,
            "raw_transcript": raw_transcript,     
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