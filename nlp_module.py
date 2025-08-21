"""
nlp_module.py — Hybrid NER loader with LayoutLMv3 integration.

Loads models from local Hugging Face cache (no network downloads).
Exposes:
 - run_layoutlmv3_on_image(image_np) -> list of (label, text, box)
 - extract_entities_full_pipeline(text, layout_hits=None) -> ensemble result dict
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from huggingface_hub import constants
HF_CACHE_DIR = constants.HF_HUB_CACHE

# Optional imports
try:
    from gliner import GLiNER
except Exception:
    GLiNER = None

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

# fuzzy matching
try:
    from rapidfuzz import process as rf_process
    _use_rapidfuzz = True
except Exception:
    try:
        from fuzzywuzzy import process as rf_process
        _use_rapidfuzz = False
    except Exception:
        rf_process = None
        _use_rapidfuzz = False

logger = logging.getLogger("nlp_module")
logging.basicConfig(level=logging.INFO)

# -------------------------
# REQUIRED MODELS (repo ids)
# -------------------------
REQUIRED_MODELS = {
    "gliner": "Ihor/gliner-biomed-small-v1.0",
    "trocr": "microsoft/trocr-large-handwritten",
    "layoutlmv3": "microsoft/layoutlmv3-large",
    "biomed_roberta": "allenai/biomed_roberta_base",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biogpt": "microsoft/BioGPT-large",
}

NER_MODEL_REPOS = {
    "biomed_roberta": REQUIRED_MODELS["biomed_roberta"],
    "clinicalbert": REQUIRED_MODELS["clinicalbert"],
}

# -------------------------
# Helpers: local cache paths
# -------------------------
def hf_repo_cache_folder(repo_id: str) -> Path:
    folder_name = "models--" + repo_id.replace("/", "--")
    return Path(HF_CACHE_DIR) / folder_name

def find_snapshot_for_repo(repo_cache_folder: Path) -> Optional[Path]:
    snaps = list((repo_cache_folder).glob("snapshots/*"))
    if not snaps:
        return None
    snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0]

def get_local_snapshot_path(repo_id: str) -> Path:
    repo_cache = hf_repo_cache_folder(repo_id)
    if not repo_cache.exists():
        raise FileNotFoundError(f"Model cache folder not found for '{repo_id}'. Expected: {repo_cache}")
    snap = find_snapshot_for_repo(repo_cache)
    if snap is None:
        raise FileNotFoundError(f"No snapshots found under {repo_cache}")
    return snap

# verify models exist (fail early)
def verify_all_models_exist():
    logger.info(f"Hugging Face cache dir: {HF_CACHE_DIR}")
    missing = []
    for k, repo in REQUIRED_MODELS.items():
        try:
            path = get_local_snapshot_path(repo)
            logger.info(f"✅ {k} at {path}")
        except Exception as e:
            logger.warning(f"❌ {k} missing: {e}")
            missing.append((k, repo))
    if missing:
        msg = "Missing local model caches: " + ", ".join(f"{k}({r})" for k, r in missing)
        logger.warning(msg)
        # do NOT raise — we allow partial start; functions are defensive
verify_all_models_exist()

# -------------------------
# Load GLiNER
# -------------------------
GLINER_OBJ = None
if GLI := GLI if False else None:  # harmless placeholder (avoid lint complaints)
    pass
# actual check
if GLI := None:
    pass  # unreachable - safe
if GLiNER is not None:
    try:
        gliner_repo = REQUIRED_MODELS["gliner"]
        gliner_path = str(get_local_snapshot_path(gliner_repo))
        logger.info(f"Loading GLiNER from {gliner_path}")
        GLiNER_OBJ = GLiNER.from_pretrained(gliner_path)
        logger.info("GLiNER loaded.")
    except Exception:
        logger.exception("Failed to load GLiNER; continuing without it.")
        GLiNER_OBJ = None
else:
    logger.info("GLiNER library not installed; skipping GLiNER.")

# -------------------------
# Load transformer NER pipelines
# -------------------------
NER_PIPELINES: Dict[str, Any] = {}
for name, repo_id in NER_MODEL_REPOS.items():
    try:
        snap = str(get_local_snapshot_path(repo_id))
        logger.info(f"Loading NER pipeline {name} from {snap} (local files only).")
        tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(snap, local_files_only=True)
        ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        NER_PIPELINES[name] = ner_pipe
        logger.info(f"Loaded NER pipeline: {name}")
    except Exception:
        logger.exception(f"Could not load NER pipeline {name}; skipping.")

if not NER_PIPELINES:
    logger.info("No transformer NER pipelines loaded; rely on GLiNER + regex + fuzzy matching.")

# -------------------------
# Load LayoutLMv3 (if possible)
# -------------------------
LAYOUT_PROCESSOR = None
LAYOUT_MODEL = None
LAYOUT_PIPELINE = None
try:
    layout_snap = str(get_local_snapshot_path(REQUIRED_MODELS["layoutlmv3"]))
    logger.info(f"Loading LayoutLMv3 from {layout_snap} (local files only)")
    LAYOUT_PROCESSOR = LayoutLMv3Processor.from_pretrained(layout_snap, local_files_only=True)
    LAYOUT_MODEL = LayoutLMv3ForTokenClassification.from_pretrained(layout_snap, local_files_only=True)
    # Attempt pipeline (some HF versions accept feature_extractor argument)
    try:
        LAYOUT_PIPELINE = pipeline("token-classification", model=LAYOUT_MODEL, tokenizer=LAYOUT_PROCESSOR.tokenizer, aggregation_strategy="simple")
    except Exception:
        # fallback: we'll process via processor + model directly
        LAYOUT_PIPELINE = None
    logger.info("LayoutLMv3 loaded.")
except Exception:
    logger.exception("Failed to load LayoutLMv3; layout-aware NER disabled.")
    LAYOUT_PROCESSOR = LAYOUT_MODEL = LAYOUT_PIPELINE = None

# -------------------------
# Drug names (fuzzy map)
# -------------------------
DRUG_JSON_PATHS = [
    Path.cwd() / "drug_names.json",
    Path.cwd() / "model_assets" / "drug_names.json",
]
DRUG_DOSAGES = {}
for p in DRUG_JSON_PATHS:
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "drug_dosages" in data:
                DRUG_DOSAGES = data["drug_dosages"]
            else:
                DRUG_DOSAGES = data
            logger.info(f"Loaded drug_names from {p} ({len(DRUG_DOSAGES)} entries).")
            break
        except Exception:
            logger.exception(f"Failed parsing {p}.")
DRUG_NAMES = [k.lower() for k in DRUG_DOSAGES.keys()]

def fuzzy_match_drug(name: str, cutoff: int = 75) -> Optional[str]:
    if not DRUG_NAMES or rf_process is None:
        return None
    n = re.sub(r"[^a-z0-9]", "", name.lower())
    try:
        res = rf_process.extractOne(n, DRUG_NAMES)
        if res and res[1] >= cutoff:
            return res[0]
    except Exception:
        logger.exception("Fuzzy match failed")
    return None

# -------------------------
# LayoutLMv3 runner: use pytesseract word boxes -> layout inputs
# -------------------------
import cv2
import pytesseract

def _normalize_box(box, width, height):
    # LayoutLMv3 expects boxes in 0-1000 range
    x, y, w, h = box
    x0 = int((x / width) * 1000)
    y0 = int((y / height) * 1000)
    x1 = int(((x + w) / width) * 1000)
    y1 = int(((y + h) / height) * 1000)
    # clamp
    return [max(0, min(1000, x0)), max(0, min(1000, y0)), max(0, min(1000, x1)), max(0, min(1000, y1))]

def run_layoutlmv3_on_image(image_np) -> List[Tuple[str, str, List[int]]]:
    """
    Run LayoutLMv3 token-classification on the image.
    Returns list of tuples: (label, word_text, box)
    Defensive: returns [] if model not loaded.
    """
    if LAYOUT_PROCESSOR is None or LAYOUT_MODEL is None:
        return []

    # Get pytesseract words with boxes
    h, w = image_np.shape[:2]
    data = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)
    words = []
    boxes = []
    for i, txt in enumerate(data.get("text", [])):
        t = str(txt).strip()
        if t == "":
            continue
        x = int(data["left"][i]); y = int(data["top"][i]); ww = int(data["width"][i]); hh = int(data["height"][i])
        words.append(t)
        boxes.append(_normalize_box((x, y, ww, hh), w, h))

    if not words:
        return []

    # Prepare inputs with processor
    try:
        encoding = LAYOUT_PROCESSOR(image_np, words, boxes=boxes, return_tensors="pt")
        # If pipeline is available, it may accept words/boxes directly; otherwise run model
        if LAYOUT_PIPELINE is not None:
            ents = LAYOUT_PIPELINE(encoding)
            # pipeline output formatting may vary; normalize to (label, text, box)
            results = []
            for e in ents:
                lab = e.get("entity_group") or e.get("entity") or e.get("label")
                txt = e.get("word") or e.get("text") or ""
                # pipeline doesn't return box easily; we will not rely on it here
                results.append((lab, txt, []))
            return results
        else:
            # raw model forward -> logits -> ids -> labels
            with __import__("torch").no_grad():
                outputs = LAYOUT_MODEL(**{k: v for k, v in encoding.items()})
                logits = outputs.logits  # (batch, seq_len, num_labels)
                preds = logits.argmax(-1).squeeze().tolist()
                # use tokenizer from processor
                tokens = LAYOUT_PROCESSOR.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
                id2label = getattr(LAYOUT_MODEL.config, "id2label", {})
                results = []
                # naive mapping tokens -> words: choose tokens that are not special
                for tok, pred in zip(tokens, preds):
                    if tok in ("[CLS]", "[SEP]", "[PAD]"):
                        continue
                    lab = id2label.get(pred, str(pred))
                    results.append((lab, tok.replace("##", ""), []))
                return results
    except Exception:
        logger.exception("LayoutLMv3 inference failed")
        return []

# -------------------------
# GLiNER wrapper (as before)
# -------------------------
DEFAULT_GLINER_LABELS = ["Drug", "Drug dosage", "Instruction", "Frequency", "medicine", "dosage", "instruction"]

def run_gliner_on_text(text: str) -> Dict[str, List[str]]:
    out = {"medicine": [], "dosage": [], "instruction": [], "frequency": []}
    if GLINER_OBJ is None:
        return out
    try:
        ents = GLINER_OBJ.predict_entities(text, labels=DEFAULT_GLINER_LABELS, threshold=0.45)
        if isinstance(ents, list) and len(ents) == 1 and isinstance(ents[0], list):
            ents = ents[0]
        for ent in ents:
            lab = ent.get("label", "").lower()
            txt = ent.get("text", "").strip()
            if not txt:
                continue
            if "drug" in lab or "medicine" in lab or "med" in lab:
                out["medicine"].append(txt)
            elif "dosage" in lab:
                out["dosage"].append(txt)
            elif "instruction" in lab:
                out["instruction"].append(txt)
            elif "frequency" in lab:
                out["frequency"].append(txt)
    except Exception:
        logger.exception("GLiNER failed")
    # dedupe
    for k in out:
        seen = set(); res = []
        for v in out[k]:
            if v.lower() not in seen:
                seen.add(v.lower()); res.append(v)
        out[k] = res
    return out

# -------------------------
# Transformer NER runner (as before)
# -------------------------
def run_transformer_ners(text: str) -> List[Tuple[str, str]]:
    results = []
    if not NER_PIPELINES:
        return results
    # chunk naive
    words = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
    for name, pipe in NER_PIPELINES.items():
        for ch in chunks:
            try:
                ents = pipe(ch)
                for e in ents:
                    label = e.get("entity_group") or e.get("entity") or e.get("label") or ""
                    txt = (e.get("word") or e.get("entity") or e.get("text") or "").strip()
                    if txt:
                        results.append((label.lower(), txt))
            except Exception:
                logger.exception(f"NER pipeline {name} failed on chunk.")
    return results

# -------------------------
# Aggregate / voting / final ensemble
# -------------------------
BATCH_REGEX = re.compile(r"\b(?:batch\s*no\.?|[A-Z0-9]{5,})\b", re.IGNORECASE)
DOSAGE_REGEX = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu|%|µg)\b", re.IGNORECASE)

def aggregate_votes(gl_out: Dict[str,List[str]], transformer_ents: List[Tuple[str,str]], layout_hits: List[Tuple[str,str,List[int]]] = None) -> Dict[str,List[str]]:
    aggregated = {"medicine": [], "dosage": [], "instruction": []}
    # start with gliner
    for m in gl_out.get("medicine", []):
        if not BATCH_REGEX.search(m):
            aggregated["medicine"].append(m)
    for d in gl_out.get("dosage", []):
        aggregated["dosage"].append(d)
    for ins in gl_out.get("instruction", []):
        aggregated["instruction"].append(ins)

    # layout hits (if any)
    if layout_hits:
        for lab, txt, box in layout_hits:
            l = lab.lower() if lab else ""
            if "drug" in l or "med" in l or "brand" in l:
                if not any(txt.lower() == ex.lower() for ex in aggregated["medicine"]):
                    if not BATCH_REGEX.search(txt):
                        aggregated["medicine"].append(txt)
            elif "dose" in l or DOSAGE_REGEX.search(txt):
                if not any(txt.lower() == ex.lower() for ex in aggregated["dosage"]):
                    aggregated["dosage"].append(txt)
            elif "instr" in l or "instruction" in l or "usage" in l:
                if not any(txt.lower() == ex.lower() for ex in aggregated["instruction"]):
                    aggregated["instruction"].append(txt)

    # transformer hits
    for lab, txt in transformer_ents:
        l = lab.lower()
        if "drug" in l or "chemical" in l or l.startswith("b-"):
            if not any(txt.lower() == ex.lower() for ex in aggregated["medicine"]):
                if not BATCH_REGEX.search(txt):
                    aggregated["medicine"].append(txt)
        elif "dose" in l or DOSAGE_REGEX.search(txt):
            if not any(txt.lower() == ex.lower() for ex in aggregated["dosage"]):
                aggregated["dosage"].append(txt)
        elif "instr" in l or "instruction" in l:
            if not any(txt.lower() == ex.lower() for ex in aggregated["instruction"]):
                aggregated["instruction"].append(txt)

    # final dedupe & cleanup
    for k in aggregated:
        seen = set(); out = []
        for v in aggregated[k]:
            key = v.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(v.strip())
        aggregated[k] = out
    return aggregated

def regex_extract_dosages(text: str):
    return list({m.group(0) for m in DOSAGE_REGEX.finditer(text)})

def dedup_list(items):
    seen = set(); out = []
    for x in items:
        k = x.lower().strip()
        if k and k not in seen:
            seen.add(k); out.append(x.strip())
    return out

def extract_entities_full_pipeline(text: str, layout_hits: List[Tuple[str,str,List[int]]] = None) -> Dict[str,Any]:
    """Main public API. Provide OCR text and optional layout hits to include layout model results."""
    if not text or not text.strip():
        return {"medicine": [], "dosage": [], "instructions": [], "mapped_medicines": [], "transformer_hits": []}

    gl = run_gliner_on_text(text)
    trans = run_transformer_ners(text)
    agg = aggregate_votes(gl, trans, layout_hits=layout_hits)

    # regex dosages merge
    agg["dosage"] = dedup_list(agg["dosage"] + regex_extract_dosages(text))

    # fuzzy map
    mapped = []
    for m in agg["medicine"]:
        matched = fuzzy_match_drug(m) if DRUG_NAMES else None
        mapped.append({"extracted": m, "matched": matched, "dosages": DRUG_DOSAGES.get(matched, []) if matched else []})

    # last-resort: scanning full text for known drug substrings
    found = {x.lower() for x in agg["medicine"]}
    for dn in DRUG_NAMES:
        if dn.lower() in text.lower() and dn.lower() not in found:
            agg["medicine"].append(dn)
            mapped.append({"extracted": dn, "matched": dn, "dosages": DRUG_DOSAGES.get(dn, [])})
            found.add(dn.lower())

    agg["medicine"] = dedup_list(agg["medicine"])
    agg["dosage"] = dedup_list(agg["dosage"])
    agg["instruction"] = dedup_list(agg.get("instruction", []))

    return {
        "medicine": agg["medicine"],
        "dosage": agg["dosage"],
        "instructions": agg["instruction"],
        "mapped_medicines": mapped,
        "transformer_hits": trans
    }

# small sanity test when run directly
if __name__ == "__main__":
    sample = """
    TELKO NOL  40 mg tab
    Batch no: AXZ213
    Mfd: 08/23 Exp: 07/25
    Take one tablet once daily after breakfast
    """
    print(json.dumps(extract_entities_full_pipeline(sample), indent=2, ensure_ascii=False))
