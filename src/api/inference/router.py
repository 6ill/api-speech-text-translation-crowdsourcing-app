from fastapi import (
    APIRouter, 
    Depends, 
    UploadFile, 
    File as FastAPIFile, 
    HTTPException,
    status
)
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Annotated, Any
from uuid import UUID

from src.api.auth.dependency import get_current_user
from src.db.main import get_session
from src.db.models import User 
from src.api.inference.service import InferenceService
from src.api.inference.schema import TranscribeResponse

router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]

@router.get("/{file_id}", status_code=status.HTTP_200_OK)
async def get_transcription_result(
    file_id: UUID,
    session: SessionDep,
    user: CurrentUser
):
    """
    Get all segments for a specific file.
    Used by frontend Audio Player to sync text with audio.
    Future-ready for Translation display.
    """
    segments = await InferenceService.get_segments_by_file_id(session, user, file_id)
    
    return {
        "message": "Transcription segments retrieved successfully",
        "data": {
            "file_id": file_id,
            "total_segments": len(segments),
            "segments": segments
        }
    }
    
@router.post("/{file_id}/translate", status_code=status.HTTP_201_CREATED)
async def translate_file(
    file_id: UUID,
    session: SessionDep,
    user: CurrentUser
):
    """
    Trigger Machine Translation for a file.
    User decides when to proceed to translation phase.
    """
    file_record = await InferenceService.trigger_translation(session, user, file_id)
    
    return {
        "message": "Translation task started",
        "data": {
            "file_id": file_record.id,
            "status": file_record.status
        }
    }