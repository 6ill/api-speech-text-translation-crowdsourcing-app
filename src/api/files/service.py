import os
from fastapi import HTTPException, UploadFile, status
from sqlmodel import desc, select
from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID, uuid4

from .schema import FileUpdate
from src.core.config import Config
from src.core.logging import get_logger
from src.core.storage import StorageClient
from src.db.models import File, FileStatus, Role, User
from src.core.errors import FileNotFound
from src.workers.inference_tasks import run_transcription_task

logger = get_logger("File_Service")

class FileService:
    
    @staticmethod
    async def get_file_status(file_id: UUID, session: AsyncSession, user: User) -> File:
        """
        Fetches the file record from the DB.
        """
        query = select(File).where(File.id == file_id)
        if user.role != Role.ADMIN:
            query = query.where(File.user_id == user.id)
                
        result = await session.exec(query)
        file_record = result.first()

        if not file_record:
            raise FileNotFound()
        
        return file_record
    
    @staticmethod
    async def get_all_files(session: AsyncSession, user: User, skip: int = 0, limit: int = 10):
        """
        Fetches files with strict ownership rules.
        Admin: Sees all.
        User: Sees only their own.
        """
        query = select(File).options(selectinload(File.speaker))

        if user.role != Role.ADMIN:
            query = query.where(File.user_id == user.id)

        query = query.order_by(desc(File.created_at)).offset(skip).limit(limit)

        result = await session.exec(query)
        
        return result.all()
    
    @staticmethod
    async def get_file_by_id(file_id: UUID, session: AsyncSession, user: User) -> File:
        query = select(File).options(selectinload(File.speaker)).where(File.id == file_id)

        if user.role != Role.ADMIN:
            query = query.where(File.user_id == user.id)

        result = await session.exec(query)
        file_record = result.first()

        if not file_record:
            raise FileNotFound()
    
        return file_record
    

    @staticmethod
    async def upload_audio(
        session: AsyncSession,
        user: User,
        file: UploadFile,
        speaker_id: UUID | None
    ) -> File:
        """
        Handle upload + saving to storage + trigger transcriptions
        """
        allowed_extensions = {".mp3", ".wav", ".m4a", ".ogg"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Invalid audio format. Allowed: mp3, wav, m4a, ogg")

        new_file_uuid = uuid4() 
        storage_key = f"audio/{user.id}/{new_file_uuid}{file_ext}" 
        
        new_file = File(
            id=new_file_uuid,
            user_id=user.id,
            speaker_id=speaker_id,
            file_name=file.filename,
            storage_bucket=Config.STORAGE_BUCKET_AUDIO,
            storage_key=storage_key,
            duration_seconds=0.0,
            mime_type=file.content_type,
            file_size=file.size if file.size else 0,
            status=FileStatus.UPLOADING
        )
        
        session.add(new_file)
        await session.commit()
        await session.refresh(new_file)
        
        try:
            success = StorageClient.upload_file_obj(
                file.file, 
                storage_key, 
                file.content_type
            )
            
            if not success:
                raise Exception("S3 Upload failed")
                
        except Exception as e:
            await session.delete(new_file)
            await session.commit()
            logger.error(f"Upload failed, DB record deleted: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload file to storage")

        # Update Status & AUTO TRIGGER CELERY
        new_file.status = FileStatus.UPLOADED
        session.add(new_file)
        await session.commit()
        
        run_transcription_task.delay(
            file_id=str(new_file.id), 
            storage_key=storage_key
        )
        
        return new_file

    @staticmethod
    async def delete_file(file_id: UUID, session: AsyncSession, user: User):
        file_record = await session.get(File, file_id)
        if not file_record:
            raise FileNotFound()
            
        is_owner = (file_record.user_id == user.id)
        is_admin = user.role == Role.ADMIN

        if not(is_owner or is_admin):
            raise FileNotFound()
        
        storage_key_to_delete = file_record.storage_key

        try:
            await session.delete(file_record)
            await session.commit()
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Database deletion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete file record: {str(e)}")

        StorageClient.delete_file(storage_key_to_delete)
        
        return True
    
    @staticmethod
    async def update_file_metadata(
        file_id: UUID, 
        update_data: FileUpdate, 
        session: AsyncSession, 
        user: User
    ) -> File:
        file_record = await FileService.get_file_by_id(file_id, session, user)

        update_dict = update_data.model_dump(exclude_unset=True)
        
        if not update_dict:
            return file_record

        file_record.sqlmodel_update(update_dict)
        
        session.add(file_record)
        await session.commit()
        
        return await FileService.get_file_by_id(file_id, session, user)
