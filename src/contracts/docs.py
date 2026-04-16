from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    size: int = Field(10, ge=1, le=100)


class DeleteDocsRequest(BaseModel):
    ids: list[str]