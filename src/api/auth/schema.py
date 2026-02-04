from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
from uuid import UUID

from src.api.files.schema import FileStatusResponse

class UserCreateModel(BaseModel):
    email:str = Field(max_length=255)
    password: str = Field(min_length=6)

class UserModel(BaseModel):
    id: UUID
    email: str
    created_at: datetime
    password_hash: str = Field(exclude=True)
    files: List[FileStatusResponse]