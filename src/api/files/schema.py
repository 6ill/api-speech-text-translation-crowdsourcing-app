from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from src.db.models import FileStatus

class FileStatusResponse(BaseModel):
    """
    Response for polling the status of a file.
    """
    id: UUID
    status: FileStatus
    file_name: str
    created_at: datetime