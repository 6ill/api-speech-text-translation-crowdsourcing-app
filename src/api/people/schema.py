from pydantic import BaseModel, Field, EmailStr
from uuid import UUID
from typing import Optional

class PeopleBase(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    email: EmailStr

class PeopleCreate(PeopleBase):
    pass

class PeopleUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    email: Optional[EmailStr] = Field(default=None, max_length=255)

class PeopleResponse(PeopleBase):
    id: UUID