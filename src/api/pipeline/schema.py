from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime

class PipelineTriggerResponse(BaseModel):
    task_id: str
    message: str
    task_type: str

class PipelineConfigResponse(BaseModel):
    id: UUID
    task_type: str
    is_active: bool
    cron_schedule: str
    min_samples_required: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    evaluation_dataset_storage_key: str
    updated_at: datetime
 
    model_config = {"from_attributes": True}
 
 
class PipelineConfigUpdate(BaseModel):
    is_active: Optional[bool] = None
    cron_schedule: Optional[str] = Field(default=None, max_length=50)
    min_samples_required: Optional[int] = Field(default=None, ge=1)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    num_epochs: Optional[int] = Field(default=None, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    evaluation_dataset_storage_key: Optional[str] = Field(default=None, max_length=255)
 
 
class PipelineRunLogResponse(BaseModel):
    id: UUID
    config_id: UUID
    task_type: str
    mlflow_run_id: Optional[str]
    status: str
    data_samples_used: int
    metrics_baseline: Optional[dict]
    metrics_new_model: Optional[dict]
    message: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
 
    model_config = {"from_attributes": True}