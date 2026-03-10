# multimodal/ocr.py
#
# What this file does:
#   1. Takes an image file (JPG/PNG)
#   2. Validates and resizes image to safe dimensions before OCR
#   3. Runs EasyOCR to extract text
#   4. Calculates average confidence score across all detected text blocks
#   5. Returns extracted text + confidence + HITL flag
#
# Used by: app.py when user selects "Image" input mode
# HITL triggers when: confidence < OCR_CONFIDENCE_THRESHOLD (0.7 in config)

import easyocr
from PIL import Image
from utils.config import Config

# Initialize EasyOCR reader once
# ['en'] = English only — sufficient for JEE math problems
# gpu=False = CPU mode, works on all machines without CUDA setup
reader = easyocr.Reader(['en'], gpu=False)

# Max dimension EasyOCR handles reliably
# Screenshots can be 2000-4000px which causes internal resize issues
MAX_IMAGE_SIZE = 1500


def prepare_image(image_path: str) -> str:
    """
    Validate, convert to RGB, and resize image to safe dimensions.
    Saves processed image back to same path.
    """

    img = Image.open(image_path)

    # Validate dimensions
    width, height = img.size
    if width == 0 or height == 0:
        raise ValueError("Image has zero dimensions — possibly corrupted.")

    # Convert to RGB — EasyOCR needs RGB not RGBA/grayscale
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if image is too large
    # EasyOCR internally resizes and can produce 0 dimensions on very large images
    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:

        # Calculate new size keeping aspect ratio
        ratio      = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height)
        new_width  = max(1, int(width  * ratio))
        new_height = max(1, int(height * ratio))

        img = img.resize((new_width, new_height), Image.LANCZOS)

    # Save processed image back to same temp path
    img.save(image_path)

    return image_path


def run_ocr(image_path: str) -> dict:
    """
    Run EasyOCR on an image and return extracted text with confidence.

    Args:
        image_path: path to the image file (JPG or PNG)

    Returns:
        {
            "text":        "extracted and cleaned text",
            "confidence":  0.85,
            "needs_hitl":  False,
            "raw_blocks":  [...]
        }
    """

    try:
        # Validate + resize image before passing to EasyOCR
        prepare_image(image_path)

        # Run OCR — detail=1 returns [bbox, text, confidence] per block
        results = reader.readtext(image_path, detail=1)

        # Nothing detected in image
        if not results:
            return {
                "text"      : "",
                "confidence": 0.0,
                "needs_hitl": True,
                "raw_blocks": []
            }

        # Each result is: (bbox, text, confidence)
        extracted_lines   = []
        confidence_scores = []

        for (bbox, text, confidence) in results:
            extracted_lines.append(text)
            confidence_scores.append(confidence)

        # Join all lines into one clean string
        full_text = " ".join(extracted_lines).strip()

        # Average confidence across all detected blocks
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # Trigger HITL if confidence below threshold
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