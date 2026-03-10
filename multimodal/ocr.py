import easyocr
from PIL import Image
from utils.config import Config

# Initialized once at module level — loading it per request would be too slow
reader = easyocr.Reader(['en'], gpu=False)

# EasyOCR can produce zero dimensions on very large images, so we cap at this
MAX_IMAGE_SIZE = 1500


def prepare_image(image_path: str) -> str:
    """
    Validate, convert to RGB, and resize image to safe dimensions.
    Saves processed image back to same path.
    """

    img = Image.open(image_path)

    width, height = img.size
    if width == 0 or height == 0:
        raise ValueError("Image has zero dimensions — possibly corrupted.")

    # EasyOCR needs RGB — convert from RGBA or grayscale if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        ratio      = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height)
        new_width  = max(1, int(width  * ratio))
        new_height = max(1, int(height * ratio))
        img = img.resize((new_width, new_height), Image.LANCZOS)

    img.save(image_path)
    return image_path


def run_ocr(image_path: str) -> dict:
    """
    Run EasyOCR on an image and return extracted text with confidence.
    """

    try:
        prepare_image(image_path)

        # detail=1 returns [bbox, text, confidence] per block
        results = reader.readtext(image_path, detail=1)

        if not results:
            return {
                "text"      : "",
                "confidence": 0.0,
                "needs_hitl": True,
                "raw_blocks": []
            }

        extracted_lines   = []
        confidence_scores = []

        for (bbox, text, confidence) in results:
            extracted_lines.append(text)
            confidence_scores.append(confidence)

        full_text = " ".join(extracted_lines).strip()
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        needs_hitl = avg_confidence < Config.OCR_CONFIDENCE_THRESHOLD

        return {
            "text"      : full_text,
            "confidence": round(avg_confidence, 2),
            "needs_hitl": needs_hitl,
            "raw_blocks": results
        }

    except Exception as e:
        return {
            "text"      : "",
            "confidence": 0.0,
            "needs_hitl": True,
            "raw_blocks": [],
            "error"     : str(e)
        }