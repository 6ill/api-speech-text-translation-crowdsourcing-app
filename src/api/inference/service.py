from fastapi import UploadFile, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Any
from uuid import UUID, uuid4
import mimetypes

from src.core.config import Config
from src.core.logging import get_logger
from src.core.storage import StorageClient
from src.db.models import File, Segment, FileStatus, User
from .schema import TranscribeResponse

from src.workers.inference_tasks import run_transcription_task

logger = get_logger("InferenceService")

class InferenceService:
    """
    Handles the transcription and translation logic.
    """

    @staticmethod
    async def handle_upload_and_dispatch(
        file: UploadFile, 
        user_id: UUID,
        db: AsyncSession,
    ) -> TranscribeResponse:
        """
        Handles the initial upload and dispatches the background task.
        """
        if not file.filename:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No file name provided.")

        file_extension = file.filename.split(".")[-1]
        new_file = File(
            id=uuid4(),
            user_id=user_id,
            status=FileStatus.UPLOADING,
            file_name=file.filename,
            storage_bucket=Config.STORAGE_BUCKET_AUDIO,
            storage_key="",
            mime_type=file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream",
            file_size=file.size or 0,
            duration_seconds=0.0 
        )
        db.add(new_file)
        await db.commit()
        await db.refresh(new_file)
        
        storage_key = f"audio/{user_id}/{new_file.id}.{file_extension}"
        
        success = StorageClient.upload_file_obj(
            file.file, 
            storage_key, 
            new_file.mime_type
        )
        
        if not success:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to upload file to storage.")

        new_file.status = FileStatus.UPLOADED
        new_file.storage_key = storage_key
        await db.commit()
        
        logger.info(f"Dispatching Celery task for file_id: {new_file.id}")
        run_transcription_task.delay(
            file_id=str(new_file.id),
            storage_key=new_file.storage_key
        )
        
        return TranscribeResponse(
            file_id=new_file.id,
            status=new_file.status,
            message="File uploaded. Transcription has been queued."
        )