from fastapi import APIRouter, HTTPException, status
from src.workers.pipeline_tasks import run_cl_pipeline
from src.db.models import PipelineTaskType
from .schema import PipelineTriggerResponse

router = APIRouter()

@router.post("/trigger/{task_type}", response_model=PipelineTriggerResponse)
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