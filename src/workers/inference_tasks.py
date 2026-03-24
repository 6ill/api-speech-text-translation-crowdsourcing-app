from contextlib import contextmanager
import gc
import mlflow
from pathlib import Path
from sqlmodel import select
import torch
from transformers import pipeline
from typing import Any
from uuid import UUID

from src.celery_app import celery_app
from src.core.config import Config
from src.core.logging import get_logger
from src.core.mlflow_client import load_model_from_registry
from src.core.storage import StorageClient
from src.db.main import get_sync_session
from src.db.models import File, Segment, FileStatus

logger = get_logger("InferenceWorker")

_GLOBAL_ASR_PIPELINE = load_model_from_registry(
    model_name=Config.ASR_MODEL_NAME,
    alias="production",
)
_GLOBAL_MT_PIPELINE = None

def get_or_load_asr_pipeline():
    """
    Lazy loader for the ASR Pipeline.
    This ensures the model is NOT loaded when FastAPI imports this file.
    It is loaded only when the Celery Worker executes the first task.
    """
    global _GLOBAL_ASR_PIPELINE
    
    # If model is already loaded in this process, return it immediately
    if _GLOBAL_ASR_PIPELINE is not None:
        return _GLOBAL_ASR_PIPELINE

    logger.info("Initializing ASR Model (Lazy Load)...")

    # Define absolute path to the specific folder
    LOCAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "whisper_production" 

    try:
        if LOCAL_MODEL_PATH.exists() and any(LOCAL_MODEL_PATH.iterdir()):
            
            # 1. Detect Hardware
            is_gpu = torch.cuda.is_available()
            device_arg = 0 if is_gpu else -1
            torch_dtype = torch.float16 if is_gpu else torch.float32
            
            logger.info(f"Hardware Check -> CUDA Available: {is_gpu}. Using device index: {device_arg}")
            
            # 2. Load Native Transformers Pipeline
            _GLOBAL_ASR_PIPELINE = pipeline(
                task="automatic-speech-recognition",
                model=str(LOCAL_MODEL_PATH),
                tokenizer=str(LOCAL_MODEL_PATH),
                device=device_arg,
                torch_dtype=torch_dtype,
                chunk_length_s=30 
            )
            
            logger.info("Model loaded successfully into RAM.")
            return _GLOBAL_ASR_PIPELINE
        else:
            logger.critical(f"Local model files not found at {LOCAL_MODEL_PATH}.")
            return None
            
    except Exception as e:
        logger.critical(f"Failed to load local model: {e}", exc_info=True)
        return None


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
    
    asr_pipeline = get_or_load_asr_pipeline()
    
    if asr_pipeline is None:
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
            result = asr_pipeline(
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
    

def get_translation_pipeline():
    """
    Lazy loading model pipeline to save VRAM when idle.
    """
    global _GLOBAL_MT_PIPELINE
    if _GLOBAL_MT_PIPELINE is not None:
        return _GLOBAL_MT_PIPELINE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Translation Model: {Config.MT_MODEL_NAME} on {device}...")
    
    _GLOBAL_MT_PIPELINE = pipeline(
        "text-generation",
        model=Config.MT_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=256,
    )
    
    return _GLOBAL_MT_PIPELINE

@celery_app.task(name="tasks.run_translation_task", queue="inference_queue")
def run_translation_task(file_id: str):
    logger.info(f"Starting translation task for File ID: {file_id}")
    
    try:
        llm_pipe = get_translation_pipeline()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
        
    with db_session_scope() as session:
        file_record = session.get(File, file_id)
        if not file_record:
            logger.error("File not found.")
            return

        segments = session.exec(
            select(Segment).where(Segment.file_id == file_id).order_by(Segment.start_timestamp)
        ).all()

        total_segments = len(segments)
        logger.info(f"Translating {total_segments} segments...")

        
        for index, seg in enumerate(segments):
            original_text = seg.transcription_text
            
            if not original_text or len(original_text.strip()) == 0:
                continue

            messages = [
                {"role": "system", "content": "You are a professional translator. Translate the following Indonesian text into English accurately. Do not add any explanations, notes, or conversational filler. Output only the translation."},
                {"role": "user", "content": original_text},
            ]
            
            # Generate
            try:
                outputs = llm_pipe(
                    messages, 
                    temperature=0.1
                )
                
                generated_text = outputs[0]["generated_text"][-1]["content"]
                
                translated_text = generated_text.strip()
                
                seg.translation_text = translated_text
                
                if index % 10 == 0:
                    logger.info(f"Translated {index}/{total_segments}")

            except Exception as e:
                logger.error(f"Error translating segment {seg.id}: {e}")
                continue

        # Update File Status & Commit
        file_record.status = FileStatus.TRANSLATED
        session.add(file_record)
        session.commit()
        
        logger.info(f"Translation completed for File ID: {file_id}")

    global _GLOBAL_MT_PIPELINE
    _GLOBAL_MT_PIPELINE = None
    del llm_pipe              
    
    gc.collect()              
    torch.cuda.empty_cache()  
    
    logger.info(f"Translation model unloaded to free VRAM.")