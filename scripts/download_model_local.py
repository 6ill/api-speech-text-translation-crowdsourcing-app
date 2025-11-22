# scripts/download_model_manual.py
import sys
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# --- SETUP PATH ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.core.config import Config
from src.core.storage import StorageClient
from src.core.logging import get_logger

logger = get_logger("ManualDownloader")

# Folder lokal untuk menyimpan model
LOCAL_MODEL_DIR = project_root / "models" / "whisper_production"


def download_model_manually():
    # 1. Setup Koneksi
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 2. Cari lokasi S3 dari model "Production"
    logger.info(f"Resolving alias 'production' for model '{Config.ASR_MODEL_NAME}'...")
    try:
        model_version = client.get_model_version_by_alias(
            name=Config.ASR_MODEL_NAME, alias="production"
        )
    except Exception as e:
        logger.error("Model not found via alias. Make sure you registered it!")
        raise e

    # Source: s3://mlflow-artifacts/0/.../artifacts/model
    # Kita butuh path relatif setelah nama bucket
    full_source_uri = model_version.source
    # Hapus 's3://mlflow-artifacts/' untuk mendapatkan prefix
    # Contoh source: s3://mlflow-artifacts/0/models/xyz/artifacts/model
    bucket_name = "mlflow-artifacts"  # Default bucket MLflow
    logger.info(full_source_uri)
    prefix = full_source_uri.replace(f"models:", "0/models")

    logger.info(f"Model found at S3 prefix: {prefix}")

    # 3. Siapkan Folder Lokal
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 4. List Objects di S3 menggunakan Boto3 (StorageClient)
    s3_client = StorageClient.get_client()

    # List semua file dalam folder model tersebut
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        logger.error("No files found in S3 bucket!")
        return

    logger.info(f"Downloading files to {LOCAL_MODEL_DIR}...")

    # 5. Download Loop (Bypass ETag Check)
    for obj in response["Contents"]:
        key = obj["Key"]
        # Nama file lokal (relative path)
        relative_name = key.replace(prefix, "").lstrip("/")

        if not relative_name:
            continue  # Skip folder itu sendiri

        local_file_path = LOCAL_MODEL_DIR / relative_name

        # Buat subfolder jika ada
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading: {relative_name}...")

        # DOWNLOAD MANUAL TANPA CEK ETAG
        s3_client.download_file(bucket_name, key, str(local_file_path))

    print("\nDOWNLOAD SUCCESS!")
    print(f"Model saved at: {LOCAL_MODEL_DIR}")
    print("Now update your inference worker to load from this path.")


if __name__ == "__main__":
    download_model_manually()
