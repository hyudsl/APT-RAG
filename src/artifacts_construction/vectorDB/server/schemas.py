from typing import List, Dict, Any
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    k: int = 20


class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    time_metrics: Dict[str, float]
    total_documents: int
