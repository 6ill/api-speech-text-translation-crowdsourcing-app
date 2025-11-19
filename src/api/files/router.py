# src/api/files/router.py

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID

from src.db.main import get_session
from .service import FileService
from .schema import FileStatusResponse

router = APIRouter()

@router.get("/{file_id}/status", response_model=FileStatusResponse)
async def get_status_for_file(
    file_id: UUID,
    db: AsyncSession = Depends(get_session)
):
    """
    Poll this endpoint to get the current status of the file processing.
    """
    file_record = await FileService.get_file_status(file_id, db)
    return file_record