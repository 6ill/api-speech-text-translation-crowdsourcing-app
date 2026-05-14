import os
import pandas as pd
from datasets import Dataset, Audio
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# Path to your main dataset folder that contains 'wavs/' and 'metadata.csv'
DATASET_DIR = Path("/home/cit/Tugas-Akhir/TABillGraceHizkia/asr-khotbah/khotbah_dataset") 
CSV_PATH = DATASET_DIR / "metadata.csv"
N_ROW = 1000

# The name of the output folder and zip file
OUTPUT_DIR = "static_test_dataset_asr_new"

def build_hf_dataset():
    print(f"1. Reading metadata CSV from {CSV_PATH}...")
    
    if not CSV_PATH.exists():
        print(f"ERROR: Metadata CSV not found at {CSV_PATH}")
        return

    # Read the CSV. Expected columns: path, text, duration, original_source
    df = pd.read_csv(CSV_PATH)
    print(f"   Loaded {len(df)} rows from CSV.")

    data_samples = []
    missing_files = 0
    
    print("2. Verifying audio files and matching with transcripts...")
    for index, row in df.iterrows():
        # The 'path' column contains relative paths like "wavs/audio1.wav"
        relative_audio_path = row["path"]
        transcript_text = row["text"]
        
        # Combine the base directory with the relative path to get the exact location
        absolute_audio_path = DATASET_DIR / relative_audio_path
        
        # Verify the file actually exists on the hard drive
        if absolute_audio_path.exists():
            data_samples.append({
                "audio": str(absolute_audio_path.resolve()), 
                "sentence": transcript_text  # The ASR trainer expects the key 'sentence'
            })
        else:
            print(f"   [WARNING] Audio file missing: {absolute_audio_path}")
            missing_files += 1

        if len(data_samples) == N_ROW:
            break
        

    print(f"   Successfully matched: {len(data_samples)} files.")
    if missing_files > 0:
        print(f"   Missing audio files: {missing_files}")

    if len(data_samples) == 0:
        print("ERROR: No valid audio-transcript pairs found! Check your paths.")
        return
    
    print(f"3. Creating Hugging Face Dataset with {len(data_samples)} samples...")
    ds = Dataset.from_list(data_samples)
    
    print("4. Casting audio column to 16kHz (Required for Whisper/ML pipeline)...")
    # This ensures that when the dataset is loaded later, it resamples to 16kHz on the fly
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"5. Saving dataset to disk at '{OUTPUT_DIR}'...")
    ds.save_to_disk(OUTPUT_DIR)
    
    print("6. Compressing the dataset into a ZIP file...")
    # This creates 'static_test_dataset_asr.zip'
    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
    
    # Optional: Clean up the unzipped folder after zipping to save space
    shutil.rmtree(OUTPUT_DIR)
    
    print(f"\nSUCCESS! The dataset has been packaged.")
    print(f"File ready for S3 upload: {os.path.abspath(OUTPUT_DIR + '.zip')}")

if __name__ == "__main__":
    build_hf_dataset()