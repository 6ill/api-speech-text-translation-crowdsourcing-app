from uuid import UUID
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List, Optional

from src.db.models import People
from .schema import PeopleCreate, PeopleUpdate

class PeopleService:
    async def get_all_people(self, session: AsyncSession) -> List[People]:
        statement = select(People).order_by(People.name)
        result = await session.exec(statement)
        return result.all()

    async def get_person_by_id(self, person_id: UUID, session: AsyncSession) -> Optional[People]:
        statement = select(People).where(People.id == person_id)
        result = await session.exec(statement)
        return result.first()
    
    async def get_person_by_email(self, email: str, session: AsyncSession) -> Optional[People]:
        statement = select(People).where(People.email == email)
        result = await session.exec(statement)
        return result.first()

    async def create_person(self, person_data: PeopleCreate, session: AsyncSession) -> People:
        data_dict = person_data.model_dump()
        new_person = People(**data_dict)
        
        session.add(new_person)
        await session.commit()
        await session.refresh(new_person)
        return new_person

    async def update_person(self, person_id: UUID, update_data: PeopleUpdate, session: AsyncSession) -> Optional[People]:
        person_db = await self.get_person_by_id(person_id, session)
        if not person_db:
            return None
        
        person_data = update_data.model_dump(exclude_unset=True)
        
        person_db.sqlmodel_update(person_data)
            
        session.add(person_db)
        await session.commit()
        await session.refresh(person_db)
        return person_db

    async def delete_person(self, person_id: UUID, session: AsyncSession) -> bool:
        person_db = await self.get_person_by_id(person_id, session)
        if not person_db:
            return False
            
        await session.delete(person_db)
        await session.commit()
        return True