# warmup_models.py
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warmup")

# Models to download
REQUIRED_MODELS = [
    "microsoft/BioGPT-large",
    "allenai/biomed_roberta_base",
    "emilyalsentzer/Bio_ClinicalBERT",
    "Ihor/gliner-biomed-small-v1.0",  # GLiNER medical small
    "microsoft/trocr-large-handwritten",
    "microsoft/layoutlmv3-large",
]

# Hugging Face cache dir (defaults to ~/.cache/huggingface/hub)
from huggingface_hub import constants
HF_CACHE_DIR = Path(constants.HF_HUB_CACHE)
logger.info(f"Using Hugging Face cache: {HF_CACHE_DIR}")

def download_all():
    for model_id in REQUIRED_MODELS:
        logger.info(f"Downloading {model_id} ...")
        snapshot_download(
            repo_id=model_id,
            cache_dir=HF_CACHE_DIR,
            local_files_only=False,  # force check/download
            resume_download=True,    # continue if partial
        )
        logger.info(f"✅ Finished: {model_id}")

if __name__ == "__main__":
    download_all()
    logger.info("All required models downloaded successfully.")
