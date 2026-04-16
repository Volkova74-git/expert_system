from datetime import datetime

from pydantic import BaseModel
from typing import List


class FileSchema(BaseModel):
    Name: str
    Text: str


class DocumentSchema(BaseModel):
    ID: str
    Name: str
    RegNumber: str
    RegDate: datetime
    Files: List[FileSchema]


class ChunkSchema(BaseModel):
    document_id: str
    document_name: str
    file_name: str
    chunk_text: str
    total_chunks: int
    chunk_index: int
    reg_number: str
    reg_date: datetime
    vector: List[float] | None = None


class BatchSchema(BaseModel):
    batch_id: str
    size: int
    chunks: List[ChunkSchema]