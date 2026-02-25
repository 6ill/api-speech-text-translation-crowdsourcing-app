from uuid import UUID

from datasets import Dataset, Audio
import json
from pathlib import Path
import soundfile
from sqlalchemy.orm import Session
from sqlmodel import select, func
import torchaudio
from typing import List, Any, Tuple

from src.core.logging import get_logger
from src.core.storage import StorageClient
from src.db.models import (
    TranscriptionCorrection, 
    Segment, 
    File, 
    CorrectionStatus,
    TranslationCorrection
)

logger = get_logger("Dataset_Builder")


class ASRDatasetBuilder:
    """
    Utility class to build ASR datasets for fine-tuning.
    Handles: Data fetching, Experience Replay, Audio Downloading & Cropping.
    """

    def __init__(self, db_session: Session, local_cache_dir: str = "./training_cache"):
        self.db = db_session
        self.cache_dir = Path(local_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = 16000  # Standard for Whisper/SeamlessM4T

    def fetch_training_data(
        self, min_samples: int, replay_ratio: float = 0.2
    ) -> Tuple[List[dict], List[Any]]:
        """
        Orchestrates the data collection process.
        Returns:
            1. A list of dictionaries ready for Hugging Face Dataset.
            2. A list of correction IDs (UUIDs) used (to update DB later).
        """
        # 1. Fetch New Data (Approved & Unused)
        new_data_rows = self._fetch_new_corrections(limit=1000)  # Safe limit per batch
        count_new = len(new_data_rows)

        if count_new < min_samples:
            logger.info(
                f"Not enough new data. Found {count_new}, required {min_samples}."
            )
            return [], []

        # 2. Calculate Replay Data needed
        # If ratio is 0.2, meant 20% of TOTAL data should be old data.
        # Formula: replay_count = (new_count * ratio) / (1 - ratio)
        if replay_ratio > 0:
            count_replay = int((count_new * replay_ratio) / (1 - replay_ratio))
            replay_data_rows = self._fetch_replay_corrections(limit=count_replay)
            logger.info(
                f"Data Composition: {count_new} New + {len(replay_data_rows)} Replay"
            )
            combined_rows = new_data_rows + replay_data_rows
        else:
            combined_rows = new_data_rows

        # 3. Process Audio (Download & Crop)
        dataset_dicts = []
        used_correction_ids = []

        logger.info("Processing audio segments...")
        for row in combined_rows:
            correction, segment, file_record = row

            try:
                # Process audio returns the path to the local cropped wav file
                audio_path = self._process_audio_segment(
                    file_record=file_record, segment=segment
                )

                if audio_path:
                    dataset_dicts.append(
                        {"audio": audio_path, "sentence": correction.corrected_text}
                    )

                    # Only mark NEW data as used (Replay data is already True)
                    if not correction.used_for_training:
                        used_correction_ids.append(correction.id)

            except Exception as e:
                logger.error(f"Skipping segment {segment.id}: {e}")
                continue

        return dataset_dicts, used_correction_ids

    def convert_to_hf_dataset(self, data_dicts: List[dict]) -> Dataset:
        """
        Converts list of dicts to Hugging Face Dataset object
        and casts the 'audio' column to Audio feature.
        """
        if not data_dicts:
            return Dataset.from_dict({})

        ds = Dataset.from_list(data_dicts)
        # This automatically handles loading audio files when accessed
        ds = ds.cast_column("audio", Audio(sampling_rate=self.target_sr))
        return ds

    # --- PRIVATE HELPER METHODS ---

    def _fetch_new_corrections(self, limit: int):
        """Query for Approved and Unused corrections."""
        statement = (
            select(TranscriptionCorrection, Segment, File)
            .join(Segment, TranscriptionCorrection.segment_id == Segment.id)
            .join(File, Segment.file_id == File.id)
            .where(TranscriptionCorrection.status == CorrectionStatus.APPROVED)
            .where(TranscriptionCorrection.used_for_training == False)
            .limit(limit)
        )
        return self.db.exec(statement).all()

    def _fetch_replay_corrections(self, limit: int):
        """Query for OLD (Used) corrections randomly for Experience Replay."""
        statement = (
            select(TranscriptionCorrection, Segment, File)
            .join(Segment, TranscriptionCorrection.segment_id == Segment.id)
            .join(File, Segment.file_id == File.id)
            .where(TranscriptionCorrection.status == CorrectionStatus.APPROVED)
            .where(TranscriptionCorrection.used_for_training == True)
            .order_by(func.random())  # Random sampling
            .limit(limit)
        )
        return self.db.exec(statement).all()

    def _process_audio_segment(self, file_record: File, segment: Segment) -> str:
        """
        1. Downloads full audio from S3 to temp.
        2. Loads and crops audio based on timestamps.
        3. Resamples to 16kHz.
        4. Saves cropped audio to cache dir.
        Returns local path to cropped audio.
        """
        # 1. Download Full File
        temp_file_path = (
            self.cache_dir / f"temp_{file_record.id}"
        )

        # Check if raw file already exists in cache to save bandwidth (optional optimization)
        if not temp_file_path.exists():
            raw_bytes = StorageClient.download_file_obj(file_record.storage_key)
            if not raw_bytes:
                raise ValueError(f"Could not download {file_record.storage_key}")

            with open(temp_file_path, "wb") as f:
                f.write(raw_bytes)

        # 2. Load & Crop (Using Torchaudio)
        # Torchaudio load returns: (waveform, sample_rate)
        waveform, sr = torchaudio.load(temp_file_path)

        # Calculate frames for cropping
        start_frame = int(segment.start_timestamp * sr)
        end_frame = int(segment.end_timestamp * sr)

        # Safety check for end_frame
        if end_frame > waveform.shape[1]:
            end_frame = waveform.shape[1]

        cropped_waveform = waveform[:, start_frame:end_frame]

        # 3. Resample if necessary (Whisper needs 16000)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
            cropped_waveform = resampler(cropped_waveform)

        # 4. Save Cropped Segment
        output_filename = f"{segment.id}.wav"
        output_path = self.cache_dir / output_filename

        torchaudio.save(output_path, cropped_waveform, self.target_sr)

        # Optional: Delete temp full file to save space,
        # or keep it if you expect multiple segments from same file.
        # os.remove(temp_file_path)

        return str(output_path)

    def cleanup_cache(self):
        """Utility to clean up the local audio cache."""
        for p in self.cache_dir.glob("*"):
            p.unlink()

    def load_static_test_set(self, s3_key: str) -> Dataset:
        """
        Loads the static evaluation dataset from S3.
        1. Downloads the JSON manifest.
        2. Downloads the referenced audio files to cache.
        3. Builds the HF Dataset.
        """
        logger.info(f"Loading static test set from S3 key: {s3_key}")

        # 1. Download JSON Manifest
        json_bytes = StorageClient.download_file_obj(s3_key)
        if not json_bytes:
            raise ValueError(f"Static dataset manifest not found at {s3_key}")

        manifest_data = json.loads(json_bytes.decode("utf-8"))
        logger.info(f"Manifest loaded. Found {len(manifest_data)} samples.")

        dataset_dicts = []

        # 2. Process each entry
        for item in manifest_data:
            storage_key = item["storage_key"]
            sentence = item["sentence"]

            # Determine local path
            filename = Path(storage_key).name
            local_audio_path = self.cache_dir / f"static_{filename}"

            # Download audio if not cached
            if not local_audio_path.exists():
                audio_bytes = StorageClient.download_file_obj(storage_key)
                if not audio_bytes:
                    logger.warning(
                        f"Missing audio for static set: {storage_key}. Skipping."
                    )
                    continue

                with open(local_audio_path, "wb") as f:
                    f.write(audio_bytes)

            # Optional: Validate audio file integrity
            try:
                # Just try to load metadata to ensure it's valid audio
                torchaudio.info(str(local_audio_path))

                dataset_dicts.append(
                    {"audio": str(local_audio_path), "sentence": sentence}
                )
            except Exception as e:
                logger.error(f"Corrupt audio in static set {storage_key}: {e}")
                continue

        # 3. Convert to HF Dataset
        return self.convert_to_hf_dataset(dataset_dicts)

class MTDatasetBuilder:
    def __init__(self, db_session: Session):
        self.db = db_session
        
    def convert_to_hf_dataset(self, data_dicts: List[dict]) -> Dataset:
        """
        Converts list of dicts to Hugging Face Dataset object
        """
        if not data_dicts:
            return Dataset.from_dict({})

        ds = Dataset.from_dict({
            "source_text": [d["source_text"] for d in data_dicts],
            "target_text": [d["target_text"] for d in data_dicts]
        })
        
        return ds

    def fetch_training_data(self, min_samples: int, replay_ratio: float = 0.2) -> Tuple[Dataset, List[UUID]]:
        """
        Fetches approved new translation data + old data (replay).
        Returns HF Dataset and list of used IDs.
        """
        stmt_new = select(TranslationCorrection).where(
            TranslationCorrection.status == CorrectionStatus.APPROVED,
            TranslationCorrection.used_for_training == False
        ).limit(1000)
        new_data = self.db.exec(stmt_new).all()
        
        count_new = len(new_data)
        if count_new < min_samples:
            logger.info(f"Not enough MT data. Found {count_new}, required {min_samples}.")
            return None, []

        if replay_ratio > 0:
            count_replay = int((count_new * replay_ratio) / (1 - replay_ratio))
            stmt_replay = select(TranslationCorrection).where(
                TranslationCorrection.status == CorrectionStatus.APPROVED,
                TranslationCorrection.used_for_training == True
            ).order_by(func.random()).limit(count_replay)
            replay_data = self.db.exec(stmt_replay).all()
            
            combined_data = new_data + replay_data
        else:
            combined_data = new_data

        dataset_dicts = []
        used_correction_ids = []
        
        for record in combined_data:
            dataset_dicts.append({
                "source_text": record.original_text,
                "target_text": record.corrected_text
            })
            if not record.used_for_training:
                used_correction_ids.append(record.id)
                
        
        return dataset_dicts, used_correction_ids