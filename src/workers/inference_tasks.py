from asgiref.sync import async_to_sync
from contextlib import contextmanager
from sqlmodel import select
from typing import Any
from uuid import UUID

from src.celery_app import celery_app
from src.core.config import Config
from src.core.logging import get_logger
from src.core.mlflow_client import load_model_pipeline
from src.core.storage import StorageClient
from src.db.main import get_sync_session
from src.db.models import File, Segment, FileStatus

logger = get_logger("InferenceWorker")

logger.info("Worker starting... Loading ASR Model...")
ASR_PIPELINE: Any = load_model_pipeline(
    Config.ASR_MODEL_NAME, 
    "automatic-speech-recognition", 
    "production"
)

if ASR_PIPELINE is None:
    logger.critical("WORKER FAILED TO START: Could not load ASR model.")


@contextmanager
def db_session_scope():
    session_gen = get_sync_session()
    session = next(session_gen)
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database transaction failed: {e}", exc_info=True)
        session.rollback()
        raise
    finally:
        session.close()

@celery_app.task(name="tasks.run_transcription_task", queue="inference_queue")
def run_transcription_task(file_id: str, storage_key: str):
    """
    Synchronous Celery task.
    """
    logger.info(f"[Task ID: {file_id}] Celery task started. Processing transcription...")
    
    if ASR_PIPELINE is None:
        logger.error(f"[Task ID: {file_id}] ASR_PIPELINE is not loaded. Aborting.")
        raise RuntimeError("ASR_PIPELINE is not loaded in worker.")

    file_uuid = UUID(file_id)

    try:
        with db_session_scope() as session:
            statement = select(File).where(File.id == file_uuid)
            result = session.exec(statement)
            file_record = result.first()
            if not file_record:
                logger.error(f"[Task ID: {file_id}] File not found in DB.")
                return
            
            file_record.status = FileStatus.TRANSCRIBING
            session.commit()
            
            logger.info(f"[Task ID: {file_id}] Downloading from S3: {storage_key}")
            audio_bytes = StorageClient.download_file_obj(storage_key)
            if audio_bytes is None:
                raise Exception("Failed to download file from S3.")

            logger.info(f"[Task ID: {file_id}] Starting ML inference...")
            result = ASR_PIPELINE(
                audio_bytes, 
                return_timestamps=True,
                language="id",
            )
            
            segments = []
            for chunk in result.get("chunks", []):
                start, end = chunk["timestamp"]
                segments.append(
                    Segment(
                        file_id=file_uuid,
                        start_timestamp=start or 0.0,
                        end_timestamp=end or start or 0.0,
                        transcription_text=chunk["text"].strip()
                    )
                )
            
            if not segments:
                logger.warning(f"[Task ID: {file_id}] Transcription returned no segments.")

            session.add_all(segments)
            
            file_record.status = FileStatus.TRANSCRIBED
            file_record.duration_seconds = segments[-1].end_timestamp if segments else 0.0
        
        logger.info(f"[Task ID: {file_id}] Transcription complete. Status: TRANSCRIBED.")

    except Exception as e:
        logger.error(f"[Task ID: {file_id}] Transcription failed: {e}", exc_info=True)
        pass