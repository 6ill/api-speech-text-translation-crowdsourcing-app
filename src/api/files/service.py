from fastapi import HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID
from src.db.models import File

class FileService:
    
    @staticmethod
    async def get_file_status(file_id: UUID, db: AsyncSession) -> File:
        """
        Fetches the file record from the DB.
        """
        file_record = await db.get(File, file_id)
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        return file_record