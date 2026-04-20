import sys
import os
import torch
import mlflow.artifacts
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger("Test_Adapter_MLFlow")

def test_load_and_inference():
    model_name = Config.MT_MODEL_NAME
    alias = "production"
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI) 
        
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
        
        model_uri = f"models:/{model_name}@{alias}"
        logger.info(f"Fetching adapter from MLflow Registry: {model_uri}")

        # Download the entire PyFunc model artifact
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        
        adapter_path = None
        for root, dirs, files in os.walk(local_dir):
            if "adapter_config.json" in files:
                adapter_path = root
                break
        
        if adapter_path:
            logger.info(f"Adapter successfully located at: {adapter_path}")
            return adapter_path
        else:
            logger.warning(f"'adapter_config.json' not found inside the downloaded artifacts.")
            return None

    except Exception as e:
        logger.warning(f"Failed to fetch adapter '{model_name}@{alias}': {e}")
        return None

if __name__ == "__main__":
    test_load_and_inference()