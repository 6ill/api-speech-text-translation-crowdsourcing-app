from typing import List, Optional
from uuid import UUID
from sqlmodel import select, col, desc
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.models import PipelineConfig, PipelineRunLog, PipelineTaskType
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