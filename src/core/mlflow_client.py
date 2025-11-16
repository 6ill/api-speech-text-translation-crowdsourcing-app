import mlflow
import os
from typing import Any, Optional

# Adjust imports to match your project structure
from src.core.config import Config 
from src.core.logging import get_logger

logger = get_logger("MLflow_Client")

def load_model_pipeline(
    model_name: str, 
    task: str,
    alias: str = "production",
) -> Optional[Any]:
    """
    Loads a registered AI model pipeline by resolving its alias to a specific version.
    """
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI) 
        
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
        
        model_uri = f"models:/{model_name}@{alias}"
        logger.info(f"Attempting to load model from URI: {model_uri}")

        asr_pipeline = mlflow.transformers.load_model(
            model_uri=model_uri,
            task=task
        )
        
        logger.info(f"Successfully loaded model '{model_name}' (Alias: {alias}).")
        return asr_pipeline

    except Exception as e:
        logger.error(f"Failed to load model from MLflow Registry using alias '{alias}': {e}", exc_info=True)
        return None