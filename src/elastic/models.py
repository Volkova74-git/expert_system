from datetime import datetime

from pydantic import BaseModel, Field
from typing import Optional, List


class SearchParams(BaseModel):
    size: int = Field(default=5, ge=1, le=100)
    filter_date_from: Optional[datetime] = None
    filter_date_to: Optional[datetime] = None


class PositionModel(BaseModel):
    text: str
    start: int
    end: int


class ChunkModel(BaseModel):
    id: str
    document_id: str
    document_name: str
    file_name: str
    chunk_text: str
    total_chunks: int
    chunk_index: int
    reg_number: str
    reg_date: datetime
    vector:  Optional[List[float]] = None
    score: Optional[float] = None

    rerank_score: Optional[float] = None   # наш итоговый rerank score
    keyword_positions: Optional[List[PositionModel]] = None


class SearchResponse(BaseModel):
    chunks: List[ChunkModel]
    total_hits: int
    search_time: float