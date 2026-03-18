from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel.ext.asyncio.session import AsyncSession
from src.api.auth.dependency import RoleChecker
from src.workers.pipeline_tasks import run_cl_pipeline
from src.db.main import get_session
from src.db.models import PipelineTaskType, Role
from typing import Annotated, Optional

from .schema import PipelineConfigUpdate, PipelineTriggerResponse
from .service import PipelineService

router = APIRouter()

SessionDep     = Annotated[AsyncSession, Depends(get_session)]
allow_admin_only = Depends(RoleChecker([Role.ADMIN]))

@router.post("/trigger/{task_type}", response_model=PipelineTriggerResponse, dependencies=[allow_admin_only])
async def trigger_pipeline_manual(task_type: PipelineTaskType):
    """
    Manually triggers the Continual Learning Pipeline for a specific task (asr/mt).
    This is useful for testing or forced updates.
    """
    try:
        task = run_cl_pipeline.delay(task_type_str=task_type.value)
        
        return PipelineTriggerResponse(
            task_id=task.id,
            message=f"Pipeline for {task_type.value} triggered successfully.",
            task_type=task_type.value
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger pipeline: {str(e)}"
        )
    
@router.get("/config", dependencies=[allow_admin_only])
async def get_all_configs(session: SessionDep):
    """Return configs for all pipeline task types (asr + mt)."""
    configs = await PipelineService.get_all_configs(session)
    return {"message": "Pipeline configs retrieved", "data": configs}
 
 
@router.patch("/config/{task_type}", dependencies=[allow_admin_only])
async def update_config(
    task_type: PipelineTaskType,
    payload: PipelineConfigUpdate,
    session: SessionDep,
):
    """Partially update the pipeline config for a given task type."""
    updated = await PipelineService.update_config(task_type, payload, session)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pipeline config found for task type '{task_type}'.",
        )
    return {"message": "Config updated successfully", "data": updated}
 
 

@router.get("/runs", dependencies=[allow_admin_only])
async def get_run_logs(
    session: SessionDep,
    task_type: Optional[PipelineTaskType] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Return pipeline run history, newest first. Filter by task_type optionally."""
    logs = await PipelineService.get_run_logs(session, task_type, limit)
    return {
        "message": "Run logs retrieved",
        "data": logs,
        "metadata": {"total": len(logs), "limit": limit},
    }