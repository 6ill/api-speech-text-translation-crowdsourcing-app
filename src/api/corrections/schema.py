from datetime import datetime
from pydantic import BaseModel, ConfigDict
from uuid import UUID
from enum import Enum
from typing import List, Optional

class TaskType(str, Enum):
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"

class CorrectionSubmit(BaseModel):
    segment_id: UUID
    corrected_text: str

class ReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    RESET = "reset"

class CorrectionResponse(BaseModel):
    id: UUID
    segment_id: UUID
    original_text: str
    corrected_text: str
    status: str
    start_timestamp: float
    end_timestamp: float
    file_id: UUID

    model_config = ConfigDict(from_attributes=True)

class CorrectionReview(BaseModel):
    correction_ids: List[UUID]
    action: ReviewAction