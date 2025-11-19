from pydantic import BaseModel
from uuid import UUID
from src.db.models import FileStatus

class TranscribeResponse(BaseModel):
    file_id: UUID
    status: FileStatus
    message: str