from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID

from src.core.errors import FileNotFound, FileNotTranscribed, FileNotTranslated, TranslationInProgress
from src.core.logging import get_logger
from src.db.models import File, Role, Segment, FileStatus, User
from src.utils.subtitle import generate_subtitle_content
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
    
    @staticmethod
    async def export_subtitles(
        session: AsyncSession,
        user: User,
        file_id: UUID,
        export_type: str, # "transcription" or "translation"
        format_type: str  # "srt" or "vtt"
    ) -> tuple[str, str]:
        """
        Return: (content_string, filename)
        """
        file_record = await session.get(File, file_id)
        if not file_record or file_record.user_id != user.id:
            raise FileNotFound()
            
        is_translation = (export_type == "translation")
        
        if is_translation and file_record.status not in [FileStatus.TRANSLATED]:
            raise FileNotTranslated()
            
        if not is_translation and file_record.status not in [FileStatus.TRANSCRIBED, FileStatus.TRANSLATED, FileStatus.TRANSLATING]:
            raise FileNotTranscribed()

        segments = await InferenceService.get_segments_by_file_id(session, user, file_id)

        is_vtt = (format_type == "vtt")
        content = generate_subtitle_content(segments, is_translation=is_translation, is_vtt=is_vtt)

        safe_title = "".join([c if c.isalnum() else "_" for c in file_record.file_name])
        lang_code = "en" if is_translation else "id"
        ext = "vtt" if is_vtt else "srt"
        
        filename = f"{safe_title}_{lang_code}.{ext}"

        return content, filename
    
    @staticmethod
    async def get_full_text(
        session: AsyncSession,
        user: User,
        file_id: UUID,
        task_type: str # "transcription" or "translation"
    ) -> str:
        file_record = await session.get(File, file_id)
        if not file_record or file_record.user_id != user.id:
            raise FileNotFound()
            
        is_translation = (task_type == "translation")
        
        if is_translation:
            if file_record.status != FileStatus.TRANSLATED:
                raise FileNotTranslated()
        else:
            if file_record.status not in [FileStatus.TRANSCRIBED, FileStatus.TRANSLATING, FileStatus.TRANSLATED]:
                raise FileNotTranscribed()

        segments = await InferenceService.get_segments_by_file_id(session, user, file_id)

        text_parts = []
        for seg in segments:
            text = seg.translation_text if is_translation else seg.transcription_text
            
            if text and text.strip():
                text_parts.append(text.strip())

        full_text = " ".join(text_parts)
        
        return full_text
