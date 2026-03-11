from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List, Optional, Type
from uuid import UUID

from src.db.models import (
    File, Segment, User, Role,
    TranscriptionCorrection, TranslationCorrection,
    CorrectionStatus
)
from src.core.errors import InsufficientPermission, FileNotFound
from .schema import CorrectionResponse, CorrectionReview, CorrectionSubmit, ReviewAction, TaskType

class CorrectionService:
    @staticmethod
    async def submit_transcription_corrections(session: AsyncSession, user: User, payload: List[CorrectionSubmit]):
        return await CorrectionService._process_batch(
            session=session,
            user=user,
            payload=payload,
            CorrectionModel=TranscriptionCorrection,
            target_field="transcription_text"
        )

    @staticmethod
    async def submit_translation_corrections(session: AsyncSession, user: User, payload: List[CorrectionSubmit]):
        return await CorrectionService._process_batch(
            session=session,
            user=user,
            payload=payload,
            CorrectionModel=TranslationCorrection,
            target_field="translation_text"
        )
    
    @staticmethod
    async def _process_batch(
        session: AsyncSession,
        user: User,
        payload: List[CorrectionSubmit],
        CorrectionModel: Type,
        target_field: str
    ):
        results = []
        
        for item in payload:
            segment = await session.get(Segment, item.segment_id)
            if not segment:
                continue
            
            file = await session.get(File, segment.file_id)
            
            if file.user_id != user.id and user.role != Role.ADMIN:
                raise FileNotFound()

            current_text_in_segment = getattr(segment, target_field)
            
            stmt_exist = select(CorrectionModel).where(CorrectionModel.segment_id == item.segment_id)
            existing_correction = (await session.exec(stmt_exist)).first()
            
            if existing_correction:
                existing_correction.corrected_text = item.corrected_text
                existing_correction.status = CorrectionStatus.PENDING
                existing_correction.used_for_training = False
                
                session.add(existing_correction)
                results.append(existing_correction)
            else:
                new_correction = CorrectionModel(
                    segment_id=item.segment_id,
                    original_text=current_text_in_segment if current_text_in_segment else "",
                    corrected_text=item.corrected_text,
                    status=CorrectionStatus.PENDING
                )
                session.add(new_correction)
                results.append(new_correction)
            
            setattr(segment, target_field, item.corrected_text)
            session.add(segment)
            
        await session.commit()
        return results
    
    @staticmethod
    async def get_corrections(
        session: AsyncSession,
        task_type: TaskType,
        file_id: Optional[UUID] = None,
        status_filter: Optional[CorrectionStatus] = None
    ):
        if task_type == TaskType.TRANSCRIPTION:
            ModelClass = TranscriptionCorrection
        else:
            ModelClass = TranslationCorrection

        query = (
            select(ModelClass, Segment)
            .join(Segment, ModelClass.segment_id == Segment.id)
        )

        if file_id:
            query = query.where(Segment.file_id == file_id)
        
        if status_filter:
            query = query.where(ModelClass.status == status_filter)
        
        query = query.order_by(Segment.start_timestamp)
        
        results = await session.exec(query)
        
        # Result tuple: (CorrectionObj, SegmentObj)
        response_data = []
        for correction, segment in results.all():
            resp = CorrectionResponse(
                id=correction.id,
                segment_id=correction.segment_id,
                original_text=correction.original_text,
                corrected_text=correction.corrected_text,
                status=correction.status,
                start_timestamp=segment.start_timestamp,
                end_timestamp=segment.end_timestamp,
                file_id=segment.file_id
            )
            response_data.append(resp)
            
        return response_data

    @staticmethod
    async def review_batch(
        session: AsyncSession,
        task_type: TaskType,
        payload: CorrectionReview
    ):
        if task_type == TaskType.TRANSCRIPTION:
            ModelClass = TranscriptionCorrection
        else:
            ModelClass = TranslationCorrection
        
        stmt = select(ModelClass).where(col(ModelClass.id).in_(payload.correction_ids))
        results = await session.exec(stmt)
        corrections = results.all()
        
        updated_count = 0
        
        if payload.action == ReviewAction.APPROVE:
            target_status = CorrectionStatus.APPROVED
        elif payload.action == ReviewAction.REJECT:
            target_status = CorrectionStatus.REJECTED
        elif payload.action == ReviewAction.RESET:
            target_status = CorrectionStatus.PENDING
            
        is_training_ready = (payload.action == ReviewAction.APPROVE)
        
        for correction in corrections:
            correction.status = target_status
            correction.used_for_training = is_training_ready
            session.add(correction)
            updated_count += 1
            
        await session.commit()
        return updated_count