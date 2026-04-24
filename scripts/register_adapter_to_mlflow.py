import sys
import os
import mlflow
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.core.config import Config

ADAPTER_PATH = "/home/cit/Tugas-Akhir/TABillGraceHizkia/whisper-qlora-checkpoint"
MODEL_NAME = Config.ASR_MODEL_NAME

# Create a "dummy" class so that MLflow will register this as an Official Model
class AdapterWrapper(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        pass

def register_unsloth_adapter():
    print(f"Menghubungkan ke MLflow di {Config.MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    
    os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL

    mlflow.set_experiment("Manual_Model_Registration")
    
    try:
        with mlflow.start_run(run_name=f"Register Unsloth {MODEL_NAME}") as run:
            print(f"Mengunggah isi folder adapter ke S3 (via PyFunc Wrapper)...")
            
            # Log model using PyFunc. 
            mlflow.pyfunc.log_model(
                artifact_path="model_adapter",
                python_model=AdapterWrapper(),
                artifacts={"adapter_files": ADAPTER_PATH}
            )
            
            model_uri = f"runs:/{run.info.run_id}/model_adapter"
            registered_model = mlflow.register_model(model_uri, MODEL_NAME)
            
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(MODEL_NAME, "production", registered_model.version)
            
            print(f"\nUnslot adapter is successfully uploaded & registered into Registry!")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    register_unsloth_adapter()