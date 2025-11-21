import sys
import os
from pathlib import Path

# ADD PROJECT ROOT TO SYSTEM PATH ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import shutil
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Import Storage Client proyek Anda untuk upload otomatis
# Pastikan environment variable sudah di-set di terminal sebelum menjalankan script ini!
from src.core.storage import StorageClient 
from src.core.config import Config

# Konfigurasi
DATASET_ID = "SpeechColab/GigaSpeech2"
LANGUAGE_SUBSET = "id"
TARGET_SAMPLES = 100 # Jumlah sampel untuk test set statis
OUTPUT_DIR = "static_test_dataset_asr"
ZIP_FILENAME = "static_test_dataset_asr.zip"
S3_KEY = "datasets/static_test_dataset_asr.zip"

def prepare_static_dataset():
    print("1. Downloading TEST metadata (Human Annotated)...")
    # Kita gunakan split 'test' karena kualitasnya lebih baik untuk evaluasi
    metadata_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=f"data/{LANGUAGE_SUBSET}/test.tsv", 
        repo_type="dataset"
    )

    print("2. Loading transcripts map...")
    # Format GigaSpeech: segment_id <tab> text
    df = pd.read_csv(metadata_path, sep="\t", header=None, names=["segment_id", "text"])
    transcript_map = dict(zip(df["segment_id"].astype(str), df["text"]))
    
    print(f"3. Streaming GigaSpeech2 ({LANGUAGE_SUBSET}) TEST split...")
    # Streaming mode agar tidak perlu download 300GB data
    ds_full = load_dataset(
        DATASET_ID, 
        "default", 
        data_dir=f"data/{LANGUAGE_SUBSET}", 
        split="train", 
        streaming=False, 
    )

    print("4. Mapping text and filtering samples...")
    # This step joins the audio file path with the text from the TSV
    def map_audio_path_to_text(sample):
        # Extract segment_id from the audio file path
        filename = os.path.basename(sample["wav"]["path"])
        segment_id = os.path.splitext(filename)[0]
        
        text = transcript_map.get(segment_id)
        
        if text:
            # Add the text and ID for downstream processing
            sample["sentence"] = text
            sample["segment_id"] = segment_id
            return sample
        return None # Return None if text is missing
        
    ds_processed = ds_full.map(map_audio_path_to_text, num_proc=os.cpu_count())
    
    # Filter out samples that couldn't be matched
    ds_processed = ds_processed.filter(lambda x: x.get("sentence") is not None)

    # 5. Take the target number of samples (10)
    ds_final = ds_processed.select(range(min(TARGET_SAMPLES, len(ds_processed))))
    
    # --- CRITICAL STEP: Trigger audio array loading one last time ---
    collected_samples = []
    print(f"6. Accessing audio arrays and preparing final list ({len(ds_final)} samples)...")
    
    for sample in tqdm(ds_final):
        # Accessing the "array" key forces the audio decoding, 
        # but since we loaded the data non-streaming, this should be stable.
        audio_array = sample["wav"]["array"]
        sampling_rate = sample["wav"]["sampling_rate"]

        collected_samples.append({
            "audio": {
                "array": audio_array,
                "sampling_rate": sampling_rate
            },
            "sentence": sample["sentence"],
            "segment_id": sample["segment_id"]
        })
        
    print(f"Collected {len(collected_samples)} samples.")

    print("5. Creating Hugging Face Dataset...")
    # Buat object Dataset dari list
    hf_dataset = Dataset.from_list(collected_samples)
    
    # Cast kolom audio agar konsisten 16kHz (PENTING untuk Whisper)
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"6. Saving to disk: {OUTPUT_DIR}...")
    hf_dataset.save_to_disk(OUTPUT_DIR)
    
    print("7. Zipping dataset...")
    # Membuat file zip dari folder dataset
    shutil.make_archive(base_name=OUTPUT_DIR, format='zip', root_dir=OUTPUT_DIR)
    
    print(f"8. Uploading {ZIP_FILENAME} to S3 ({Config.STORAGE_BUCKET_TEST})...")
    # Baca file zip sebagai binary
    with open(ZIP_FILENAME, "rb") as f:
        success = StorageClient.upload_file_obj(
            f, 
            S3_KEY, 
            "application/zip",
            Config.STORAGE_BUCKET_TEST
        )
    
    if success:
        print(f"SUCCESS! Dataset uploaded to: {S3_KEY}")
        print("Update 'evaluation_dataset_s3_key' in your PipelineConfig DB with this key.")
    else:
        print("ERROR: Upload failed.")

    # Cleanup local files
    # shutil.rmtree(OUTPUT_DIR)
    # os.remove(ZIP_FILENAME)

if __name__ == "__main__":
    prepare_static_dataset()