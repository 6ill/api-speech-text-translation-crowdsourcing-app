from pydantic import BaseModel


class AdminStatsResponse(BaseModel):
    total_users: int
    total_files: int
    total_speakers: int
    pending_corrections: int