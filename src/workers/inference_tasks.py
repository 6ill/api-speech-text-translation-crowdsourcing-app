from contextlib import contextmanager
import gc
import os
import subprocess
import tempfile
import mlflow
from pathlib import Path
from peft import PeftModel
from sqlmodel import select
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, pipeline
from typing import Any
from uuid import UUID

from src.celery_app import celery_app
from src.core.config import Config
from src.core.logging import get_logger
from src.core.mlflow_client import fetch_adapter_from_registry, load_model_from_registry
from src.core.storage import StorageClient
from src.db.main import get_sync_session
from src.db.models import File, Segment, FileStatus

logger = get_logger("InferenceWorker")

_GLOBAL_VAD_MODEL = None
_GLOBAL_VAD_UTILS = None

_GLOBAL_ASR_PIPELINE = None

_GLOBAL_MT_MODEL = None
_GLOBAL_MT_TOKENIZER = None

def get_vad_model():
    """Lazy loader for Silero VAD to save memory."""
    global _GLOBAL_VAD_MODEL, _GLOBAL_VAD_UTILS
    
    if _GLOBAL_VAD_MODEL is None:
        logger.info("Initializing Silero VAD Model...")
        # Trust repo is required for torch hub
        _GLOBAL_VAD_MODEL, _GLOBAL_VAD_UTILS = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
    return _GLOBAL_VAD_MODEL, _GLOBAL_VAD_UTILS

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

    is_gpu = torch.cuda.is_available()
    
    base_model_id = Config.ASR_BASE_MODEL_ID
    
    logger.info(f"Fetching ASR adapter for {Config.ASR_MODEL_NAME} from MLflow...")
    adapter_path = fetch_adapter_from_registry(Config.ASR_MODEL_NAME, "production")

    logger.info(f"Loading Base Model ASR ({base_model_id})...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    if adapter_path:
        logger.info(f"Attaching LoRA Adapter to ASR from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        logger.warning("ASR Adapter not found. Using raw Base Model as fallback.")
        model = base_model

    _GLOBAL_ASR_PIPELINE = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )
    
    logger.info("ASR Pipeline is READY.")
    return _GLOBAL_ASR_PIPELINE


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
            with tempfile.TemporaryDirectory() as tmpdir:
                file_ext = os.path.splitext(storage_key)[1].lower()
                local_media_path = os.path.join(tmpdir, f"downloaded_media{file_ext}")
                
                logger.info(f"[Task ID: {file_id}] Downloading raw media from S3...")
                
                raw_media_bytes = StorageClient.download_file_obj(storage_key)
                
                if not raw_media_bytes:
                    logger.error(f"[Task ID: {file_id}] Failed to download media from S3.")
                    session.commit()
                    return
                    
                # Write the bytes to a physical temp file so FFmpeg can read it
                with open(local_media_path, "wb") as f:
                    f.write(raw_media_bytes)
                
                final_audio_path = local_media_path
                
                allowed_video = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
                if file_ext in allowed_video:
                    logger.info(f"[Task ID: {file_id}] Video detected. Extracting audio...")
                    final_audio_path = os.path.join(tmpdir, "extracted_audio.mp3")
                    
                    subprocess.run([
                        "ffmpeg", "-i", local_media_path,
                        "-q:a", "0", "-map", "a", 
                        final_audio_path, "-y"
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    new_storage_key = storage_key.rsplit(".", 1)[0] + ".mp3"
                    with open(final_audio_path, "rb") as extracted_file:
                        StorageClient.upload_file_obj(
                            file_obj=extracted_file, 
                            object_name=new_storage_key, 
                            content_type="audio/mpeg"
                        )
                    
                    # Delete the massive raw video from S3 to save bucket space
                    StorageClient.delete_file(storage_key)
                    
                    # Update the database to reflect the new extracted audio file
                    file_record.storage_key = new_storage_key
                    file_record.mime_type = "audio/mpeg"
                    file_record.file_size = os.path.getsize(final_audio_path)
                    session.commit()
                    logger.info(f"[Task ID: {file_id}] Extraction complete. Raw video deleted from S3.")
                
                logger.info(f"[Task ID: {file_id}] Running Silero VAD for smart chunking...")
                
                vad_model, vad_utils = get_vad_model()
                get_speech_timestamps, _, read_audio, _, _ = vad_utils
                
                # 1. Load audio. Silero requires exactly 16000Hz.
                wav = read_audio(final_audio_path, sampling_rate=16000)
                
                # 2. Extract speech timestamps
                # max_speech_duration_s forces natural cuts under 10 seconds
                # min_speech_duration_ms ignores throat clears, coughs, and mic bumps
                speech_timestamps = get_speech_timestamps(
                    wav, 
                    vad_model,
                    threshold=0.4, 
                    sampling_rate=16000, 
                    max_speech_duration_s=7.0, 
                    min_speech_duration_ms=350,
                    min_silence_duration_ms=800,
                    min_silence_at_max_speech=300,
                    speech_pad_ms=100,   
                )

                if not speech_timestamps:
                    logger.warning(f"[Task ID: {file_id}] VAD detected no speech. Skipping inference.")
                    file_record.status = FileStatus.TRANSCRIBED
                    session.commit()
                    return

                # 3. Slice the audio into pure-speech arrays for the pipeline
                chunk_inputs = []
                offsets = []
                
                for ts in speech_timestamps:
                    start_sample = ts['start']
                    end_sample = ts['end']
                    
                    # Hugging Face pipeline accepts raw 1D numpy arrays natively
                    chunk_inputs.append(wav[start_sample:end_sample].numpy())
                    offsets.append(start_sample / 16000.0) # Save the real start time in seconds

                logger.info(f"[Task ID: {file_id}] Passing {len(chunk_inputs)} speech chunks to ML inference...")
                
                # 4. Process all chunks efficiently using batch_size
                results = asr_pipeline(
                    chunk_inputs, 
                    return_timestamps=True,
                    generate_kwargs={
                        "language": "indonesian", 
                        "task": "transcribe"
                    },
                    batch_size=4
                )
                
                # Hugging Face returns a dict if there's 1 input, or a list of dicts for multiple
                if not isinstance(results, list):
                    results = [results]
                
                # 5. Map the isolated chunk timestamps back to the original audio's timeline
                segments = []
                for idx, (res, offset_sec) in enumerate(zip(results, offsets)):
                    chunk_duration = len(chunk_inputs[idx]) / 16000.0
                    
                    # If model fails to return timestamps, fallback to the chunk's VAD boundaries
                    chunks = res.get("chunks", [{"timestamp": (0.0, chunk_duration), "text": res.get("text", "")}])
                    
                    for chunk in chunks:
                        c_start, c_end = chunk["timestamp"]
                        
                        # Add the VAD offset back to the pipeline's internal timestamp
                        abs_start = (c_start or 0.0) + offset_sec
                        abs_end = (c_end or chunk_duration) + offset_sec
                        
                        text = chunk.get("text", "").strip()
                        if text:
                            segments.append(
                                Segment(
                                    file_id=file_uuid,
                                    start_timestamp=abs_start,
                                    end_timestamp=abs_end,
                                    transcription_text=text
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
    

def get_translation_model_and_tokenizer():
    # Avoid GPU not detected in backend service container
    from unsloth import FastLanguageModel

    global _GLOBAL_MT_MODEL, _GLOBAL_MT_TOKENIZER
    if _GLOBAL_MT_MODEL is not None:
        return _GLOBAL_MT_MODEL, _GLOBAL_MT_TOKENIZER
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing MT Model on {device}...")
    
    base_model_id = Config.MT_BASE_MODEL_ID
    adapter_path = fetch_adapter_from_registry(Config.MT_MODEL_NAME, "production")
    
    # Unsloth automatically handles the base model resolution if provided an adapter path
    model_source = adapter_path if adapter_path else base_model_id
    
    logger.info(f"Loading MT Model from {model_source} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
        token=Config.HF_TOKEN
    )
    
    # Crucial for doubling inference speed
    FastLanguageModel.for_inference(model)
    
    _GLOBAL_MT_MODEL = model
    _GLOBAL_MT_TOKENIZER = tokenizer
    
    logger.info("MT Model is READY.")
    return _GLOBAL_MT_MODEL, _GLOBAL_MT_TOKENIZER


@celery_app.task(name="tasks.run_translation_task", queue="inference_queue")
def run_translation_task(file_id: str):
    logger.info(f"Starting translation task for File ID: {file_id}")
    
    try:
        model, tokenizer = get_translation_model_and_tokenizer()
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
            
            try:
                # Format using the tokenizer's chat template
                text_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
                
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    use_cache=True, 
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                input_length = inputs['input_ids'].shape[1]
                translated_text = tokenizer.decode(
                    outputs[0][input_length:], 
                    skip_special_tokens=True
                ).strip()
                
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

    global _GLOBAL_MT_MODEL, _GLOBAL_MT_TOKENIZER
    _GLOBAL_MT_MODEL = None
    _GLOBAL_MT_TOKENIZER = None
    
    del model, tokenizer
    gc.collect()              
    torch.cuda.empty_cache()
    logger.info(f"Translation model unloaded to free VRAM.")