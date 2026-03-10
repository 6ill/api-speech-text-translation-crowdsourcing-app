from typing import Annotated
from fastapi import APIRouter, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.auth.dependency import RoleChecker
from src.db.main import get_session
from src.db.models import Role
from .service import AdminService

router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]
allow_admin_only = Depends(RoleChecker([Role.ADMIN]))


@router.get("/stats", status_code=status.HTTP_200_OK, dependencies=[allow_admin_only])
async def get_admin_stats(session: SessionDep):
    """
    Get aggregated platform statistics for the admin dashboard.
    Returns counts for users, files, speakers, and pending corrections.
    """
    stats = await AdminService.get_stats(session)

    return {
        "message": "Admin stats retrieved successfully",
        "data": stats,
    }