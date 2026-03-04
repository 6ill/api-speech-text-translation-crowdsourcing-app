from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LOG_LEVEL: str
    # CORS
    CORS_ORIGINS: list[str]
    ALLOWED_HOSTS: list[str]
    # Auth
    JWT_SECRET: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRY_IN_SECONDS: int
    REFRESH_TOKEN_EXPIRY_IN_SECONDS: int
    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_URL: str
    # Database
    DATABASE_URL: str
    SYNC_DATABASE_URL: str
    # Model Loading
    ASR_MODEL_NAME: str
    MT_MODEL_NAME: str
    MLFLOW_TRACKING_URI: str
    MLFLOW_S3_ENDPOINT_URL: str
    MLFLOW_S3_ARTIFACT_ROOT: str
    # Object Storage
    STORAGE_ENDPOINT_URL: str
    STORAGE_BUCKET_AUDIO: str
    STORAGE_BUCKET_TEST: str
    STORAGE_ACCESS_KEY: str
    STORAGE_SECRET_KEY: str
    # Celery
    CELERY_BROKER_URL: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


Config = Settings()
