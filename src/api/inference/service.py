from fastapi import UploadFile, HTTPException, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Any
from uuid import UUID, uuid4
import mimetypes

from src.core.config import Config
from src.core.errors import FileNotFound, InsufficientPermission
from src.core.logging import get_logger
from src.core.storage import StorageClient
from src.db.models import File, Role, Segment, FileStatus, User
from .schema import TranscribeResponse

from src.workers.inference_tasks import run_transcription_task

logger = get_logger("InferenceService")

class InferenceService:
    @staticmethod
    async def get_segments_by_file_id(
        session: AsyncSession, 
        user: User, 
        file_id: UUID
    ):
        file_record = await session.get(File, file_id)
        
        if not file_record:
            raise FileNotFound()
            
        is_owner = (file_record.user_id == user.id)
        is_admin = (user.role == Role.ADMIN)
        
        if not (is_owner or is_admin):
            raise InsufficientPermission()
            
        if file_record.status not in [FileStatus.TRANSCRIBED, FileStatus.TRANSLATED]:
            return []

        statement = (
            select(Segment)
            .where(Segment.file_id == file_id)
            .order_by(Segment.start_timestamp)
        )
        
        result = await session.exec(statement)
        segments = result.all()
        
        return segments