import json
import requests
from fuzzywuzzy import fuzz
from rapidfuzz import process  # For faster matching
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load drug info at import time
try:
    with open('model_assets/drug_names.json', 'r', encoding='utf-8') as f:
        djson = json.load(f)
    DRUG_DOSAGES = djson["drug_dosages"]
    BRAND_TO_GENERIC = djson["brand_to_generic"]
    ALL_GENERICS = set(DRUG_DOSAGES.keys())
    ALL_BRANDS = set(BRAND_TO_GENERIC.keys())
    LOOKUP_SET = ALL_GENERICS.union(ALL_BRANDS)
    LOOKUP_LIST = list(LOOKUP_SET)
except Exception as e:
    logger.error(f"Failed to load drug_names.json: {e}")
    DRUG_DOSAGES, BRAND_TO_GENERIC, LOOKUP_LIST = {}, {}, []

# Function to match drug names (RxNorm API, Local DB, Fuzzy matching)
def match_drug(term: str):
    """Match extracted drug name using RxNorm API and local drug list (with fuzzy matching)."""
    if not term:
        logger.warning(f"Empty term provided for drug matching: {term}")
        return None, []

    norm = term.strip().lower()

    # 1. RxNorm first (API-based)
    api_url = f"https://rxnav.nlm.nih.gov/REST/approximateTerm?term={requests.utils.quote(term)}"
    try:
        resp = requests.get(api_url, timeout=7)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            rxnorm_name = candidates[0].get("term", "").strip()
            rxnorm_lower = rxnorm_name.lower()
            # If RxNorm returns a name in your generics/brands: map it
            if rxnorm_lower in (g.lower() for g in ALL_GENERICS):
                for g in ALL_GENERICS:
                    if g.lower() == rxnorm_lower:
                        return {
                            "source": "rxnorm+local-generic",
                            "name": g,
                            "dosages": DRUG_DOSAGES[g]
                        }, []
            elif rxnorm_lower in (b.lower() for b in ALL_BRANDS):
                for b in ALL_BRANDS:
                    if b.lower() == rxnorm_lower:
                        gen = BRAND_TO_GENERIC[b]
                        return {
                            "source": "rxnorm+local-brand",
                            "name": gen,
                            "dosages": DRUG_DOSAGES.get(gen, [])
                        }, []
            return {
                "source": "rxnorm",
                "name": rxnorm_name,
                "dosages": []
            }, []
    except Exception as e:
        logger.warning(f"RxNorm API error or skip: {e}")

    # Continue with local database matching...


    # 2. Local database (only if RxNorm fails)
    # Exact match (generic)
    for generic in ALL_GENERICS:
        if generic.lower() == norm:
            return {
                "source": "local-generic",
                "name": generic,
                "dosages": DRUG_DOSAGES[generic]
            }, []

    # Exact match (brand)
    for brand in ALL_BRANDS:
        if brand.lower() == norm:
            gen = BRAND_TO_GENERIC[brand]
            return {
                "source": "local-brand",
                "name": gen,
                "dosages": DRUG_DOSAGES.get(gen, [])
            }, []

    # Fuzzy matching as last-resort
    if not LOOKUP_LIST:
        return None, []

    results = process.extract(term, LOOKUP_LIST, scorer=fuzz.token_sort_ratio, limit=3, score_cutoff=45)
    if results:
        best_name, score, _ = results[0]
        mapped_generic = BRAND_TO_GENERIC.get(best_name, best_name)
        best_src = "local-brand" if best_name in BRAND_TO_GENERIC else "local-generic"
        best = {
            "source": f"fuzzy-{best_src}",
            "name": mapped_generic,
            "dosages": DRUG_DOSAGES.get(mapped_generic, [])
        }
        suggestions = []
        for sug_name, sug_score, _ in results[1:]:
            sug_generic = BRAND_TO_GENERIC.get(sug_name, sug_name)
            sug_src = "local-brand" if sug_name in BRAND_TO_GENERIC else "local-generic"
            suggestions.append({
                "source": f"fuzzy-{sug_src}",
                "name": sug_generic,
                "dosages": DRUG_DOSAGES.get(sug_generic, [])
            })
        return best, suggestions

    return None, []

# Function to match drug names with context (fuzzy matching, RxNorm API, local DB)
def match_drug_with_context(extracted_text: str, drug_list: dict):
    """Match extracted drug name with the most similar drug name in the drug list."""
    if not extracted_text:
        logger.warning(f"Received empty or invalid text for matching: {extracted_text}")
        return None, []

    norm = extracted_text.strip().lower()

    # 1. RxNorm first (API-based)
    api_url = f"https://rxnav.nlm.nih.gov/REST/approximateTerm?term={requests.utils.quote(extracted_text)}"
    try:
        resp = requests.get(api_url, timeout=7)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            rxnorm_name = candidates[0].get("term", "").strip()
            rxnorm_lower = rxnorm_name.lower()
            # If RxNorm returns a name in your generics/brands: map it
            if rxnorm_lower in (g.lower() for g in ALL_GENERICS):
                for g in ALL_GENERICS:
                    if g.lower() == rxnorm_lower:
                        return {
                            "source": "rxnorm+local-generic",
                            "name": g,
                            "dosages": DRUG_DOSAGES[g]
                        }, []
            elif rxnorm_lower in (b.lower() for b in ALL_BRANDS):
                for b in ALL_BRANDS:
                    if b.lower() == rxnorm_lower:
                        gen = BRAND_TO_GENERIC[b]
                        return {
                            "source": "rxnorm+local-brand",
                            "name": gen,
                            "dosages": DRUG_DOSAGES.get(gen, [])
                        }, []
            # RxNorm returns a valid name but not in your local dict
            return {
                "source": "rxnorm",
                "name": rxnorm_name,
                "dosages": []
            }, []
    except Exception as e:
        logger.warning(f"RxNorm API error or skip: {e}")

    # 2. Local database (only if RxNorm fails)
    # Exact match (generic)
    for generic in ALL_GENERICS:
        if generic.lower() == norm:
            return {
                "source": "local-generic",
                "name": generic,
                "dosages": DRUG_DOSAGES[generic]
            }, []

    # Exact match (brand)
    for brand in ALL_BRANDS:
        if brand.lower() == norm:
            gen = BRAND_TO_GENERIC[brand]
            return {
                "source": "local-brand",
                "name": gen,
                "dosages": DRUG_DOSAGES.get(gen, [])
            }, []

    # Fuzzy matching as last-resort
    if not LOOKUP_LIST:
        return None, []

    results = process.extract(extracted_text, LOOKUP_LIST, scorer=fuzz.token_sort_ratio, limit=3, score_cutoff=45)
    if results:
        best_name, score, _ = results[0]
        mapped_generic = BRAND_TO_GENERIC.get(best_name, best_name)
        best_src = "local-brand" if best_name in BRAND_TO_GENERIC else "local-generic"
        best = {
            "source": f"fuzzy-{best_src}",
            "name": mapped_generic,
            "dosages": DRUG_DOSAGES.get(mapped_generic, [])
        }
        suggestions = []
        for sug_name, sug_score, _ in results[1:]:
            sug_generic = BRAND_TO_GENERIC.get(sug_name, sug_name)
            sug_src = "local-brand"
