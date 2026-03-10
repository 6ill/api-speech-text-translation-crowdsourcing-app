from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.models import (
    CorrectionStatus,
    File,
    People,
    TranscriptionCorrection,
    TranslationCorrection,
    User,
)
from .schema import AdminStatsResponse


class AdminService:
    @staticmethod
    async def get_stats(session: AsyncSession) -> AdminStatsResponse:
        total_users = (await session.exec(select(func.count(User.id)))).one()
        total_files = (await session.exec(select(func.count(File.id)))).one()
        total_speakers = (await session.exec(select(func.count(People.id)))).one()

        pending_transcription = (
            await session.exec(
                select(func.count(TranscriptionCorrection.id)).where(
                    TranscriptionCorrection.status == CorrectionStatus.PENDING
                )
            )
        ).one()

        pending_translation = (
            await session.exec(
                select(func.count(TranslationCorrection.id)).where(
                    TranslationCorrection.status == CorrectionStatus.PENDING
                )
            )
        ).one()

        return AdminStatsResponse(
            total_users=total_users,
            total_files=total_files,
            total_speakers=total_speakers,
            pending_corrections=pending_transcription + pending_translation,
        )