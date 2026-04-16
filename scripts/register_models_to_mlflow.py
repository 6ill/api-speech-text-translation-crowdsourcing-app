import sys
import os
import torch
import mlflow
from pathlib import Path
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoTokenizer
)

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.core.config import Config


# Can be a HuggingFace ID (e.g., "meta-llama/Llama-3.2-3B-Instruct") 
# Or a local path (e.g., str(project_root / "models" / "whisper_local"))
MODEL_SOURCE = Config.MT_MODEL_NAME

# Task type: "asr" or "mt"
TASK_TYPE = "mt"

# The name that will appear in MLflow Registry
MLFLOW_MODEL_NAME = Config.MT_MODEL_NAME

def register_model():
    print(f"Connecting to MLflow at {Config.MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL

    print(f"Loading Model from Source: {MODEL_SOURCE}")
    
    if TASK_TYPE == "asr":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_SOURCE)
        processor = AutoProcessor.from_pretrained(MODEL_SOURCE)
        components = {
            "model": model,
            "feature_extractor": processor.feature_extractor,
            "tokenizer": processor.tokenizer
        }
        hf_task = "automatic-speech-recognition"
        
    elif TASK_TYPE == "mt":
        model = AutoModelForCausalLM.from_pretrained(MODEL_SOURCE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
        components = {
            "model": model,
            "tokenizer": tokenizer
        }
        hf_task = "text-generation"
    else:
        raise ValueError("Invalid TASK_TYPE. Must be 'asr' or 'mt'")

    mlflow.set_experiment(f"Manual_Model_Registration")
    
    try:
        with mlflow.start_run(run_name=f"Register {MLFLOW_MODEL_NAME}") as run:
            print(f"Run started. Uploading artifacts to S3 Object Storage...")
            
            # Log the model
            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path="model",
                task=hf_task,
                registered_model_name=MLFLOW_MODEL_NAME
            )
            
            # Set alias to production so our Celery worker picks it up
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])[0].version
            client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", model_version)
            
            print(f"\nSUCCESS: '{MLFLOW_MODEL_NAME}' registered to MLflow and set to 'production' alias!")

    except Exception as e:
        print(f"Failed to register model: {e}")

if __name__ == "__main__":
    register_model()