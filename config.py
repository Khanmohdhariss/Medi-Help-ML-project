import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
# Path to Tesseract executable; set TESSERACT_PATH environment variable to override
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "tesseract")

# Path to drug names and dosages mapping file
DRUG_NAMES_FILE = os.getenv("DRUG_NAMES_FILE", "model_assets/drug_names.json")

# Path to a "plain" drug name list if needed (optional fallback)
# DRUG_NAMES_LIST_FILE = os.getenv("DRUG_NAMES_LIST_FILE", "model_assets/drug_names_list.json")

# RxNorm API base URL, used for U.S. drug name normalization/lookup
RXNORM_BASE_URL = os.getenv("RXNORM_BASE_URL", "https://rxnav.nlm.nih.gov/REST")

# SpaCy/scispaCy model to load for biomedical NER
SPACY_NER_MODEL = os.getenv("SPACY_NER_MODEL", "en_ner_bc5cdr_md")

# Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Thread/process pool size for parallel OCR/NLP (tune for deployment scale)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))

# Add more as needed (e.g., API keys, auxiliary asset paths, debug flags, etc.)