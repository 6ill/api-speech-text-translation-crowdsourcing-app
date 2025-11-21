import mlflow
import transformers
import sys
import os
from pathlib import Path

# ADD PROJECT ROOT TO SYSTEM PATH ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.core.config import Config
MLFLOW_TRACKING_URI = Config.MLFLOW_TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL

HUGGINGFACE_MODEL_ID = "RedHatAI/whisper-large-v3-turbo-quantized.w4a16"  # (Contoh: 'whisper-base', 'whisper-small', dll.)

MLFLOW_MODEL_NAME = "whisper-asr-base"


print(f"Menghubungkan ke MLflow di {MLFLOW_TRACKING_URI}...")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

print(f"Mengunduh model {HUGGINGFACE_MODEL_ID} dari Hugging Face...")

# 1. Muat pipeline (model + processor) dari Hugging Face
# Ini akan mengunduh model ke cache lokal Anda terlebih dahulu
try:
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=HUGGINGFACE_MODEL_ID
    )
except Exception as e:
    print(f"Error saat memuat pipeline: {e}")
    print("Pastikan Anda memiliki koneksi internet dan library 'transformers' terinstal.")
    exit()

print("Model berhasil dimuat. Memulai run MLflow...")

# 2. Buat "run" singkat hanya untuk mendaftarkan model
try:
    with mlflow.start_run(run_name=f"Import {MLFLOW_MODEL_NAME}") as run:
        print(f"Run MLflow dimulai (ID: {run.info.run_id}). Melakukan log model...")
        
        # 3. Log model ke MLflow
        # 'mlflow.transformers' akan menyimpan semua komponen (model, config, processor)
        mlflow.transformers.log_model(
            transformers_model=pipe,
            name="model", 
            registered_model_name=MLFLOW_MODEL_NAME
        )
        
        print(f"\nBerhasil! Model '{MLFLOW_MODEL_NAME}' (Versi 1) telah didaftarkan.")
        print("Silakan cek MLflow UI Anda di tab 'Models'.")

except Exception as e:
    print(f"Terjadi error saat log atau mendaftarkan model ke MLflow: {e}")
    print("Pastikan MLflow server Anda berjalan di {MLFLOW_TRACKING_URI} dan dapat diakses.")