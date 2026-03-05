from enum import Enum

from pydantic import BaseModel, ConfigDict
from uuid import UUID
from src.db.models import FileStatus
from typing import Optional, List

class TranscribeResponse(BaseModel):
    file_id: UUID
    status: FileStatus
    message: str

class SegmentResponse(BaseModel):
    id: UUID
    start_timestamp: float
    end_timestamp: float
    transcription_text: str
    # FUTURE PROOFING: Field ini disiapkan dari sekarang.
    # Nanti kalau modul MT jalan, field ini akan terisi string.
    translation_text: Optional[str] = None 
    
    model_config = ConfigDict(from_attributes=True)
    
class ExportType(str, Enum):
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"

class FormatType(str, Enum):
    SRT = "srt"
    VTT = "vtt"