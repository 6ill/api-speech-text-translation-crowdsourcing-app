from fastapi import APIRouter, Depends, status, HTTPException
from typing import List, Annotated, Optional
from uuid import UUID
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.main import get_session
from src.db.models import CorrectionStatus, Role, User
from src.api.auth.dependency import RoleChecker, get_current_user, AccessTokenBearer

from .schema import CorrectionReview, CorrectionSubmit, CorrectionResponse, TaskType
from .service import CorrectionService

router = APIRouter()
SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]
allow_admin_only = Depends(RoleChecker([Role.ADMIN]))

@router.post("/{task_type}", status_code=status.HTTP_200_OK)
async def submit_corrections(
    task_type: TaskType,
    payload: List[CorrectionSubmit],
    session: SessionDep,
    user: CurrentUser
):
    """
    Submit corrections (Batch).
    Task Type: 'transcription' or 'translation'.
    """
    if task_type == TaskType.TRANSCRIPTION:
        result = await CorrectionService.submit_transcription_corrections(session, user, payload)
    elif task_type == TaskType.TRANSLATION:
        result = await CorrectionService.submit_translation_corrections(session, user, payload)
    else:
        raise HTTPException(status_code=400, detail="Invalid task type")

    return {
        "message": f"{task_type.capitalize()} corrections submitted successfully",
        "data": result
    }
    

@router.get("/{task_type}", response_model=dict, dependencies=[allow_admin_only])
async def get_corrections_list(
    task_type: TaskType,
    session: SessionDep,
    file_id: Optional[UUID] = None,
    status: Optional[CorrectionStatus] = None
):
    """
    Get list of corrections for review.
    No pagination (as requested).
    Includes timestamps for audio player integration.
    """
    data = await CorrectionService.get_corrections(
        session, task_type, file_id, status
    )
    
    return {
        "message": "Corrections retrieved successfully",
        "data": data,
        "metadata": {
            "total": len(data),
            "filters": {
                "file_id": str(file_id) if file_id else None,
                "status": status
            }
        }
    }

# 2. REVIEW Corrections (Admin Only)
@router.post("/{task_type}/review", dependencies=[allow_admin_only])
async def review_corrections(
    task_type: TaskType,
    payload: CorrectionReview,
    session: SessionDep
):
    """
    Approve or Reject corrections.
    Can be used to re-evaluate previously reviewed items.
    """
    count = await CorrectionService.review_batch(session, task_type, payload)
    
    return {
        "message": f"Successfully reviewed {count} corrections",
        "data": {
            "updated_count": count,
            "action": payload.action
        }
    }