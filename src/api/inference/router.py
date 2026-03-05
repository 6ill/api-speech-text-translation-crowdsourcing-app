from fastapi import (
    APIRouter, 
    Depends,
    Query,
    Response, 
    UploadFile, 
    File as FastAPIFile, 
    HTTPException,
    status
)
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Annotated, Any
from uuid import UUID

import urllib

from src.api.auth.dependency import get_current_user
from src.db.main import get_session
from src.db.models import User 
from src.api.inference.service import InferenceService
from src.api.inference.schema import ExportType, FormatType, TranscribeResponse

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
    
@router.get("/{file_id}/export")
async def export_subtitle_file(
    file_id: UUID,
    user: CurrentUser,
    session: SessionDep,
    export_type: ExportType = Query(description="Select output task: transcription or translation"),
    format: FormatType = Query(default=FormatType.SRT, description="Select subtitle format: srt or vtt"),
):
    """
    Download segments as a Subtitle file (.srt or .vtt).
    Browser will automatically download this response as a file.
    """
    content, filename = await InferenceService.export_subtitles(
        session=session,
        user=user,
        file_id=file_id,
        export_type=export_type.value,
        format_type=format.value
    )

    encoded_filename = urllib.parse.quote(filename)

    return Response(
        content=content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )
    
@router.get("/{file_id}/full-text", status_code=status.HTTP_200_OK)
async def get_inference_full_text(
    file_id: UUID,
    session: SessionDep,
    user: CurrentUser,
    task_type: ExportType = Query(description="Select the task type: transcription or translation"),
):
    """
    Get the full concatenated text of a file's transcription or translation.
    Returns a single string combining all segments.
    """
    full_text = await InferenceService.get_full_text(
        session=session,
        user=user,
        file_id=file_id,
        task_type=task_type.value
    )
    
    return {
        "message": f"Full {task_type.value} text retrieved successfully",
        "data": {
            "file_id": file_id,
            "task_type": task_type.value,
            "full_text": full_text
        }
    }