import os
import pandas as pd
from datasets import Dataset, Audio
import shutil
from pathlib import Path

TSV_PATH = r"D:\TABillGraceHizkia\test-static-dataset\test.tsv"
AUDIO_DIR = r"D:\TABillGraceHizkia\test-static-dataset\test"
OUTPUT_DIR = r"D:\TABillGraceHizkia\static_test_dataset_asr" 

def convert_to_hf_dataset():
    print("1. Membaca TSV...")
    # Format GigaSpeech: segment_id <tab> text
    # Pastikan nama kolom sesuai dengan isi TSV
    df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["segment_id", "text"])
    
    # Buat dictionary untuk lookup cepat: segment_id -> text
    # Ini penting agar kita tidak perlu looping dataframe berkali-kali
    transcript_map = dict(zip(df["segment_id"].astype(str), df["text"]))
    print(f"   Loaded {len(transcript_map)} transcripts mapping.")

    print("2. Memindai semua file audio (recursive)...")
    # Cari semua file .wav di dalam AUDIO_DIR dan sub-foldernya
    audio_files = list(Path(AUDIO_DIR).rglob("*.wav"))
    print(f"   Ditemukan {len(audio_files)} file audio total di folder.")

    data_samples = []
    missing_transcript_count = 0
    
    print("3. Mencocokkan Audio dengan Transkrip...")
    
    for audio_path in audio_files:
        # Ambil nama file tanpa ekstensi sebagai segment_id
        # Contoh: "path/to/0/0-1.wav" -> "0-1"
        segment_id = audio_path.stem 
        
        # Cari teks berdasarkan segment_id
        text = transcript_map.get(segment_id)
        
        if text:
            data_samples.append({
                "audio": str(audio_path), # Path absolut ke file
                "sentence": text
            })
        else:
            # Audio ada, tapi tidak ada di TSV (jarang terjadi, tapi mungkin)
            missing_transcript_count += 1

    print(f"   Berhasil dicocokkan: {len(data_samples)}")
    print(f"   Audio tanpa transkrip: {missing_transcript_count}")

    if len(data_samples) == 0:
        print("ERROR: Tidak ada data yang cocok! Cek kembali path dan format ID.")
        return

    # Ambil sampel secukupnya untuk test set (misal 100 agar evaluasi cepat)
    # Hapus slicing [:100] jika ingin memproses SEMUA data
    final_samples = data_samples[:100] 
    
    print(f"4. Memproses {len(final_samples)} sampel menjadi Dataset Hugging Face...")
    ds = Dataset.from_list(final_samples)
    
    # PENTING: Cast ke Audio 16kHz agar cocok dengan Whisper
    # Saat ini terjadi, library 'datasets' akan membaca file audio dari path, 
    # resample, dan menyimpannya dalam format internal Arrow.
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"5. Menyimpan dataset ke disk: {OUTPUT_DIR}...")
    ds.save_to_disk(OUTPUT_DIR)
    
    print("6. Membuat file ZIP...")
    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
    
    print(f"\nSELESAI! File '{OUTPUT_DIR}.zip' siap di-upload manual ke S3.")
    print(f"Lokasi output: {os.path.abspath(OUTPUT_DIR)}.zip")

if __name__ == "__main__":
    convert_to_hf_dataset()