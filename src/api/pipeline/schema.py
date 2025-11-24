from pydantic import BaseModel
from uuid import UUID

class PipelineTriggerResponse(BaseModel):
    task_id: str
    message: str
    task_type: str