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
from typing import Any
from uuid import UUID

from src.db.main import get_session
from src.db.models import User 
from src.api.inference.service import InferenceService
from src.api.inference.schema import TranscribeResponse

router = APIRouter()

async def get_hardcoded_user_id(db: AsyncSession = Depends(get_session)) -> UUID:
    user = await db.exec(select(User).limit(1))
    first_user = user.first()
    if not first_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No users in database. Please create a user first."
        )
    return first_user.id


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = FastAPIFile(...),
    db: AsyncSession = Depends(get_session),
    user_id: UUID = Depends(get_hardcoded_user_id)
):
    """
    Uploads an audio file, queues transcription, and returns immediately.
    """
    return await InferenceService.handle_upload_and_dispatch(
        file=file,
        user_id=user_id,
        db=db
    )