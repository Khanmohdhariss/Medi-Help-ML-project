"""
ocr_module.py — Patched High Accuracy Version
Hybrid OCR (Tesseract + TrOCR) → Spell Correction → LayoutLMv3 NER
"""

import io
import re
import cv2
import pytesseract
import torch
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# NLP imports
from nlp_module import run_layoutlmv3_on_image, extract_entities_full_pipeline

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_module")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load TrOCR Model (with fallback) ----------
try:
    TROCR_PATH = str(Path.home() / ".cache/huggingface/hub" / "models--microsoft--trocr-large-handwritten")
    try:
        processor = TrOCRProcessor.from_pretrained(TROCR_PATH, local_files_only=True)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_PATH, local_files_only=True).to(device)
        logger.info("✅ TrOCR loaded locally (large-handwritten).")
    except Exception:
        logger.warning("⚠️ Local TrOCR model not found — downloading...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)
except Exception as e:
    processor, trocr_model = None, None
    logger.error(f"❌ TrOCR load failed: {e}")

# ---------- Spell checker ----------
spell = SpellChecker()

# ---------- Image preprocessing ----------
def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enhance contrast, sharpen, and binarize for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 10
    )
    denoised = cv2.bilateralFilter(binary, 9, 75, 75)
    return denoised

# ---------- Tesseract ----------
def run_tesseract_ocr(image: np.ndarray) -> str:
    raw = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
    cleaned = [re.sub(r"[^a-zA-Z0-9\s.,:/()-]+", "", l).strip()
               for l in raw.splitlines() if len(l.strip()) >= 3]
    return " ".join(cleaned)

# ---------- TrOCR ----------
def run_trocr_ocr(image_bytes: bytes) -> str:
    if not (processor and trocr_model):
        return ""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Keep aspect ratio — pad to target size
        target_w, target_h = 600, 384
        pil_image.thumbnail((target_w, target_h), Image.LANCZOS)
        new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        new_img.paste(pil_image, ((target_w - pil_image.width) // 2, (target_h - pil_image.height) // 2))
        
        pixel_values = processor(images=new_img, return_tensors="pt").pixel_values.to(device)
        gen_ids = trocr_model.generate(pixel_values)
        return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        logger.warning(f"TrOCR fail: {e}")
        return ""

# ---------- Spell correction ----------
def correct_spelling(text: str) -> str:
    words = text.split()
    corrected = []
    for w in words:
        if w.isalpha() and w.lower() not in spell:
            suggestion = spell.correction(w)
            corrected.append(suggestion if suggestion else w)
        else:
            corrected.append(w)
    return " ".join(corrected)

# ---------- Main OCR + NLP pipeline ----------
import re

def extract_instructions(text: str) -> list:
    """Extract usage/storage instructions from OCR text."""
    patterns = [
        r"(take\s+\d+\s+times\s+(a|per)\s+day)",
        r"(take\s+once\s+(a|per)\s+day)",
        r"(every\s+\d+\s+hours?)",
        r"(after\s+meals?)",
        r"(before\s+meals?)",
        r"(store\s+below\s+\d+\s*°?c?)",
        r"(keep\s+out\s+of\s+reach\s+of\s+children)"
    ]
    matches = []
    for pat in patterns:
        matches.extend(re.findall(pat, text.lower()))
    # Flatten and clean matches
    clean_matches = list(set([" ".join(m).strip() if isinstance(m, tuple) else m for m in matches]))
    return clean_matches

def process_image_pipeline(image_bytes: bytes, drug_list=None) -> dict:
    from nlp_module import run_layoutlmv3_on_image, extract_entities_full_pipeline
    from drug_matcher import match_drug_with_context
    import numpy as np
    import cv2

    try:
        # Decode image
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decode failed.")

        # OCR
        preprocessed = preprocess_image_for_ocr(image)
        text_printed = run_tesseract_ocr(preprocessed)
        text_handwritten = run_trocr_ocr(image_bytes)
        full_text = f"{text_printed}\n{text_handwritten}".strip()

        # LayoutLMv3 hits
        try:
            layout_hits = run_layoutlmv3_on_image(preprocessed)
        except Exception as e:
            logger.warning(f"LayoutLMv3 failed: {e}")
            layout_hits = []

        # NLP entity extraction
        entities = extract_entities_full_pipeline(full_text, layout_hits=layout_hits)

        # Drug matching
        medicines = []
        for ent in entities:
            if ent["label"].lower() == "medication":
                match, dosages = match_drug_with_context(ent["text"], drug_list)
                medicines.append({
                    "name": match or ent["text"],
                    "dosages": dosages
                })

        # Instruction extraction
        instructions = extract_instructions(full_text)

        return {
            "printed_text": text_printed,
            "handwritten_text": text_handwritten,
            "medicines": medicines,
            "instructions": instructions
        }

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return {"status": "error", "message": str(e)}


    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return {"status": "error", "message": str(e)}

# ---------- Local test ----------
if __name__ == "__main__":
    sample_img = Path("sample_prescription.jpg")
    if sample_img.exists():
        out = process_image_pipeline(open(sample_img, "rb").read())
        import json; print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("⚠️ sample_prescription.jpg not found — skipping test.")
