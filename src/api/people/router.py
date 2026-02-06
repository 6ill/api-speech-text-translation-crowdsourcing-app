from fastapi import APIRouter, Depends, status, HTTPException
from typing import List, Annotated
from uuid import UUID
from sqlmodel.ext.asyncio.session import AsyncSession

from src.core.logging import get_logger
from src.db.main import get_session
from src.db.models import Role
from src.api.auth.dependency import RoleChecker
from .schema import PeopleCreate, PeopleResponse, PeopleUpdate
from .service import PeopleService

people_service = PeopleService()
SessionDep = Annotated[AsyncSession, Depends(get_session)]

allow_admin_only = Depends(RoleChecker([Role.ADMIN]))
allow_user_or_admin = Depends(RoleChecker([Role.USER, Role.ADMIN]))

router = APIRouter()
logger = get_logger("People_Router")

@router.get("/", dependencies=[allow_user_or_admin])
async def get_all_people(session: SessionDep):
    result =  await people_service.get_all_people(session)
    return {
        "data": result
    }

@router.get("/{person_id}", dependencies=[allow_user_or_admin])
async def get_person(person_id: UUID, session: SessionDep):
    person = await people_service.get_person_by_id(person_id, session)
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    return {
        "data": person
    }

@router.post("/", status_code=status.HTTP_201_CREATED, dependencies=[allow_admin_only])
async def create_person(person_data: PeopleCreate, session: SessionDep):
    logger.info(f"data orang: {person_data}")
    existing_person = await people_service.get_person_by_email(person_data.email, session)
    if existing_person:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Person with this email already exists")
    
    result =  await people_service.create_person(person_data, session)

    return {
        "message": "Successfully inserted new person!",
        "data": result        
    }

@router.patch("/{person_id}", dependencies=[allow_admin_only])
async def update_person(person_id: UUID, person_data: PeopleUpdate, session: SessionDep):
    updated_person = await people_service.update_person(person_id, person_data, session)
    if not updated_person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    
    return {
        "message": "Successfully updated!",
        "data": updated_person  
    }

@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[allow_admin_only])
async def delete_person(person_id: UUID, session: SessionDep):
    success = await people_service.delete_person(person_id, session)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
    return None