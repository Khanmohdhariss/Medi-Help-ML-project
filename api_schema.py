from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class MedicineEntity(BaseModel):
    name: str                     # normalized/generic name
    source: str                   # 'generic', 'brand', 'fuzzy-brand', etc
    dosages: List[str]            # all mapped dosages
    suggestions: Optional[List[Dict[str, Any]]] = []

class OCRResponse(BaseModel):
    medicine: List[MedicineEntity]
    dosage: List[str]
    instructions: List[str]

class PredictionResponse(BaseModel):
    status: str
    data: dict

class ErrorResponse(BaseModel):
    status: str
    message: str
