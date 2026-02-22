from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID

from src.core.errors import FileNotFound, FileNotTranscribed, TranslationInProgress
from src.core.logging import get_logger
from src.db.models import File, Role, Segment, FileStatus, User
from src.workers.inference_tasks import run_translation_task


logger = get_logger("InferenceService")


class InferenceService:
    @staticmethod
    async def get_segments_by_file_id(session: AsyncSession, user: User, file_id: UUID):
        file_record = await session.get(File, file_id)

        if not file_record:
            raise FileNotFound()

        is_owner = file_record.user_id == user.id
        is_admin = user.role == Role.ADMIN

        if not (is_owner or is_admin):
            raise FileNotFound()

        if file_record.status not in [FileStatus.TRANSCRIBED, FileStatus.TRANSLATED]:
            return []

        statement = (
            select(Segment)
            .where(Segment.file_id == file_id)
            .order_by(Segment.start_timestamp)
        )

        result = await session.exec(statement)
        segments = result.all()

        return segments
    
    @staticmethod
    async def trigger_translation(
        session: AsyncSession,
        user: User,
        file_id: UUID
    ):
        file_record = await session.get(File, file_id)
        if not file_record:
            raise FileNotFound()
            
        is_owner = (file_record.user_id == user.id)
        
        if not is_owner:
            raise FileNotFound()

        if file_record.status == FileStatus.TRANSLATING:
            raise TranslationInProgress(
                detail="Translation is already in progress. Please wait."
            )
        
        if file_record.status not in [FileStatus.TRANSCRIBED, FileStatus.TRANSLATED]:
             raise FileNotTranscribed()

        file_record.status = FileStatus.TRANSLATING
        session.add(file_record)
        await session.commit()
        await session.refresh(file_record)

        run_translation_task.delay(file_id=str(file_id))

        return file_record
