import mlflow
import os
import torch
from transformers import BitsAndBytesConfig
from typing import Optional, Any

from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger("MLflow_Client")

def load_model_from_registry(
    model_name: str, 
    alias: str = "production",
    use_quantization: bool = True
) -> Optional[Any]:
    """
    Loads a registered AI model pipeline from MLflow.
    Supports both standard Base Models and PEFT/LoRA adapters.
    """
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI) 
        
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
        
        model_uri = f"models:/{model_name}@{alias}"
        logger.info(f"Loading model from MLflow Registry: {model_uri}")

        model_kwargs = {"device_map": "auto"}
        
        if use_quantization and torch.cuda.is_available():
            logger.info("Applying 4-bit Quantization (BitsAndBytes) for inference...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_config

        # MLflow handles fetching the base model and applying LoRA automatically if applicable.
        # return_type="pipeline" ensures we get a callable HuggingFace pipeline back.
        pipeline = mlflow.transformers.load_model(
            model_uri=model_uri,
            model_kwargs=model_kwargs,
            return_type="pipeline" 
        )
        
        logger.info(f"Successfully loaded '{model_name}' (Alias: {alias}).")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to load model from MLflow Registry using alias '{alias}': {e}", exc_info=True)
        return None
    
def fetch_adapter_from_registry(
    model_name: str, 
    alias: str = "production"
) -> Optional[str]:
    """
    Downloads the raw adapter folder from the MLflow Registry.
    Returns the local path to the folder, or None if it fails/does not exist.
    """
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI) 
        
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
        
        model_uri = f"models:/{model_name}@{alias}"
        logger.info(f"Mencari adapter di MLflow Registry: {model_uri}")

        # Download the artifact. Since we used the PyFunc wrapper during registration,
        # the original adapter file is stored in the "artifacts/adapter_files" subfolder.
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        adapter_path = os.path.join(local_dir, "artifacts", "adapter_files")
        
        if os.path.exists(adapter_path):
            logger.info(f"Adapter successfully downloaded to: {adapter_path}")
            return adapter_path
        else:
            logger.warning(f"The adapter_files folder was not found in the artifact.")
            return None

    except Exception as e:
        logger.warning(f"There is no model with the alias '{alias}' in MLflow for '{model_name}'. (There may be no fine-tuning results yet). Details: {e}")
        return None