from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import List, Optional
from src.db.models import FileStatus

class FileStatusResponse(BaseModel):
    """
    Response for polling the status of a file.
    """
    id: UUID
    status: FileStatus
    file_name: str
    created_at: datetime

class SpeakerSummary(BaseModel):
    id: UUID
    name: str

class FileResponse(BaseModel):
    id: UUID
    file_name: str
    status: FileStatus
    duration_seconds: Optional[float] = 0.0
    created_at: datetime
    file_size: int
    mime_type: str
    speaker: Optional[SpeakerSummary] = None

    model_config = ConfigDict(from_attributes=True)

class FileDownloadResponse(BaseModel):
    file_id: UUID
    download_url: str
    expires_in_seconds: int
    
class FileListResponseWrapper(BaseModel):
    message: str
    data: List[FileResponse]
    metadata: dict
    
class FileResponseWrapper(BaseModel):
    message: str
    data: FileResponse