from typing import AsyncGenerator
from sqlmodel import create_engine, SQLModel, text
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from src.core.config import Config


engine = AsyncEngine(create_engine(url=Config.DATABASE_URL, echo=False))


async def init_db():
    """create a connection to our db"""

    async with engine.begin() as conn:
        
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to provide the session object"""
    async_session = async_sessionmaker(
        bind=engine, 
        class_=AsyncSession, 
        expire_on_commit=False # allow us to use session object after committing
    )

    async with async_session() as session:
        yield session

