from pydantic import BaseModel
from typing import List, Optional


class SupplierOut(BaseModel):
    supplier_id: str
    supplier_name: str
    country: str
    category: str
    annual_spend_usd: int
    on_time_delivery_rate: float
    quality_defect_ppm: float
    financial_health_score: float
    single_source_flag: int
    geopolitical_risk_index: float
    predicted_risk_score: float
    predicted_risk_category: str
    prediction_confidence: float


class InvestigationReport(BaseModel):
    risk_summary: str
    key_risk_drivers: List[str]
    recommendation: str
    recommended_actions: List[str]
    from_cache: bool = False


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    matched_supplier_ids: Optional[List[str]] = None
