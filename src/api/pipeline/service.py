from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlmodel import select, col, desc
from sqlmodel.ext.asyncio.session import AsyncSession

from src.core.errors import PipelineIsNotActive
from src.db.models import PipelineConfig, PipelineRunLog, PipelineRunStatus, PipelineTaskType
from src.workers.pipeline_tasks import run_cl_pipeline
from .schema import PipelineConfigUpdate, PipelineConfigResponse, PipelineRunLogResponse


class PipelineService:

    @staticmethod
    async def get_all_configs(session: AsyncSession) -> List[PipelineConfigResponse]:
        result = await session.exec(select(PipelineConfig))
        configs = result.all()
        return [PipelineConfigResponse.model_validate(c) for c in configs]

    @staticmethod
    async def get_config_by_task_type(
        task_type: PipelineTaskType, session: AsyncSession
    ) -> Optional[PipelineConfig]:
        result = await session.exec(
            select(PipelineConfig).where(PipelineConfig.task_type == task_type)
        )
        return result.first()

    @staticmethod
    async def update_config(
        task_type: PipelineTaskType,
        update_data: PipelineConfigUpdate,
        session: AsyncSession,
    ) -> Optional[PipelineConfigResponse]:
        config = await PipelineService.get_config_by_task_type(task_type, session)
        if not config:
            return None

        patch = update_data.model_dump(exclude_unset=True)
        config.sqlmodel_update(patch)
        session.add(config)
        await session.commit()
        await session.refresh(config)
        return PipelineConfigResponse.model_validate(config)

    @staticmethod
    async def get_run_logs(
        session: AsyncSession,
        task_type: Optional[PipelineTaskType] = None,
        limit: int = 50,
    ) -> List[PipelineRunLogResponse]:
        """
        Return run logs newest-first. Joins config to resolve task_type string.
        Optional task_type filter.
        """
        stmt = (
            select(PipelineRunLog, PipelineConfig)
            .join(PipelineConfig, PipelineRunLog.config_id == PipelineConfig.id)
            .order_by(desc(PipelineRunLog.start_time))
            .limit(limit)
        )
        if task_type:
            stmt = stmt.where(PipelineConfig.task_type == task_type)

        result = await session.exec(stmt)
        rows = result.all()

        out = []
        for run_log, config in rows:
            resp = PipelineRunLogResponse(
                id=run_log.id,
                config_id=run_log.config_id,
                task_type=config.task_type.value,
                mlflow_run_id=run_log.mlflow_run_id,
                status=run_log.status.value,
                data_samples_used=run_log.data_samples_used,
                metrics_baseline=run_log.metrics_baseline,
                metrics_new_model=run_log.metrics_new_model,
                message=run_log.message,
                start_time=run_log.start_time,
                end_time=run_log.end_time,
            )
            out.append(resp)
        return out
    
    @staticmethod
    async def get_config_by_task_type(
        task_type: PipelineTaskType, session: AsyncSession
    ) -> Optional[PipelineConfig]:
        result = await session.exec(
            select(PipelineConfig).where(PipelineConfig.task_type == task_type)
        )
        return result.first()
 
    @staticmethod
    async def is_pipeline_running(config_id, session: AsyncSession) -> bool:
        """
        Check if there is already a RUNNING job for this pipeline config.
        This is the guard that prevents duplicate triggers.
        """
        result = await session.exec(
            select(PipelineRunLog).where(
                PipelineRunLog.config_id == config_id,
                PipelineRunLog.status == PipelineRunStatus.RUNNING,
            )
        )
        return result.first() is not None
 
    @staticmethod
    async def trigger_pipeline(
        task_type: PipelineTaskType, session: AsyncSession
    ) -> str:
        """
        Validates config + checks for in-progress run, then dispatches the
        Celery task. Returns the Celery task ID.
        """
        config = await PipelineService.get_config_by_task_type(task_type, session)
 
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No pipeline config found for task type '{task_type.value}'.",
            )
 
        if not config.is_active:
            raise PipelineIsNotActive()
 
        already_running = await PipelineService.is_pipeline_running(config.id, session)
        if already_running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A '{task_type.value}' pipeline run is already in progress. Wait for it to finish.",
            )
 
        task = run_cl_pipeline.delay(task_type_str=task_type.value)
        return task.id
 