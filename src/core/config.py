from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LOG_LEVEL: str
    # Database
    DATABASE_URL: str
    # Model Loading
    ASR_MODEL_NAME: str
    MLFLOW_TRACKING_URI: str
    MLFLOW_S3_ENDPOINT_URL: str
    MLFLOW_S3_ARTIFACT_ROOT: str
    # Object Storage
    STORAGE_ENDPOINT_URL: str
    STORAGE_BUCKET_NAME: str
    STORAGE_ACCESS_KEY: str
    STORAGE_SECRET_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


Config = Settings()