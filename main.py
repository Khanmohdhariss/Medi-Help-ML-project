# main.py (recommended FastAPI)
import json
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

from ocr_module import process_image_pipeline  # must accept drug_list param

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Medicine OCR API")

DRUG_FILE = Path("model_assets/drug_names.json")
drug_list = []  # loaded at startup

@app.on_event("startup")
async def startup():
    global drug_list
    try:
        if DRUG_FILE.exists():
            with DRUG_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Normalize to expected shape: list of names or dict depending on your matcher
            if isinstance(data, dict):
                # if saved as {name: [suggestions,...], ...}
                drug_list = data
            elif isinstance(data, list):
                drug_list = data
            else:
                logger.warning("drug_names.json has unexpected shape; using empty list.")
                drug_list = []
            logger.info(f"Loaded drug list with {len(drug_list)} entries.")
        else:
            logger.warning("Drug list not found at model_assets/drug_names.json; continuing without it.")
            drug_list = []
    except Exception as e:
        logger.exception("Failed to load drug list at startup.")
        drug_list = []

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate content type quickly (image/*)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        # Call heavy sync function in a thread so we don't block the event loop
        result = await run_in_threadpool(process_image_pipeline, image_bytes, drug_list)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))
