from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .schema import UserCreateModel
from .utils import generate_password_hash
from src.db.models import User 


class UserService:
    async def get_user_by_email(self, email: str, session: AsyncSession):
        statement = select(User).where(User.email == email)

        result = await session.exec(statement)

        user = result.first()

        return user
    
    async def user_exists(self, email, session: AsyncSession):
        user = await self.get_user_by_email(email, session)

        return user is not None

    async def create_user(self, user_data: UserCreateModel, session: AsyncSession):
        data_dict = user_data.model_dump()
        new_user = User(**data_dict)

        new_user.password_hash = generate_password_hash(data_dict["password"])

        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        return new_user