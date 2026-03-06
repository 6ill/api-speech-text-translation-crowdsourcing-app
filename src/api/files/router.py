# src/api/files/router.py

from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Form, Query, UploadFile, status, File as FastAPIFile
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID

from src.api.auth.dependency import RoleChecker, get_current_user
from src.core.storage import StorageClient
from src.db.main import get_session
from src.db.models import Role, User
from .service import FileService
from .schema import FileResponseWrapper, FileStatusResponse, FileListResponseWrapper, FileUpdate

SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]

router = APIRouter()

@router.get("/{file_id}/status")
async def get_status_for_file(
    file_id: UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_session), 
):
    """
    Poll this endpoint to get the current status of the file processing.
    """
    file_record = await FileService.get_file_status(file_id, db, current_user)
    
    return {
        "data": FileStatusResponse.model_validate(file_record)
    }



@router.get("/", status_code=status.HTTP_200_OK, response_model=FileListResponseWrapper)
async def list_files(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    """
    List files. User sees own files. Admin sees all files.
    """
    skip = (page - 1) * limit
    files = await FileService.get_all_files(session, user, skip, limit)
    
    return {
        "message": "Files retrieved successfully",
        "data": files,
        "metadata": {
            "page": page,
            "limit": limit
        }
    }

@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_file(
    session: SessionDep,
    user: CurrentUser,
    file: UploadFile = FastAPIFile(...),
    speaker_id: Optional[UUID] = Form(None) # Metadata dari Form-Data
):
    """
    Upload audio file. Automatically triggers transcription pipeline.
    """
    new_file = await FileService.upload_audio(session, user, file, speaker_id)
    
    return {
        "message": "File uploaded and transcription started",
        "data": new_file
    }

@router.get("/{file_id}", status_code=status.HTTP_200_OK, response_model=FileResponseWrapper)
async def get_file_detail(
    file_id: UUID,
    session: SessionDep,
    user: CurrentUser
):
    """
    Get file detail. Includes speaker info and status.
    """
    file_record = await FileService.get_file_by_id(file_id, session, user)
    
    return {
        "message": "File detail retrieved",
        "data": file_record
    }

@router.get("/{file_id}/url", status_code=status.HTTP_200_OK)
async def get_file_download_url(
    file_id: UUID,
    session: SessionDep,
    user: CurrentUser
):
    """
    Get a temporary Presigned URL to play/download the audio.
    Valid for 1 hour.
    """
    file_record = await FileService.get_file_by_id(file_id, session, user)
    
    # Generate URL
    url = StorageClient.generate_presigned_url(file_record.storage_key)
    
    if not url:
        return {
            "message": "Failed to generate URL",
            "data": None
        }
        
    return {
        "message": "Download URL generated",
        "data": {
            "file_id": file_record.id,
            "download_url": url,
            "expires_in_seconds": 3600
        }
    }

@router.delete("/{file_id}") 
async def delete_file(
    file_id: UUID,
    session: SessionDep,
    user: CurrentUser
):
    """
    Delete file. Users can delete their own files. Admins can delete any file.
    """
    await FileService.delete_file(file_id, session, user)
    
    return {
        "message": "File deleted successfully",
        "data": None
    }
    
@router.patch("/{file_id}", status_code=status.HTTP_200_OK, response_model=FileResponseWrapper)
async def update_file(
    file_id: UUID,
    update_data: FileUpdate,
    session: SessionDep,
    user: CurrentUser
):
    """
    Update file metadata (e.g., file_name, speaker_id).
    Only the fields provided in the request body will be updated.
    """
    updated_file = await FileService.update_file_metadata(
        file_id=file_id, 
        update_data=update_data, 
        session=session, 
        user=user
    )
    
    return {
        "message": "File metadata updated successfully",
        "data": updated_file
    }