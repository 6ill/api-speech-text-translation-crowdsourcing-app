from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.core.config import Config
from src.core.logging import get_logger, setup_global_logging
from src.core.mlflow_client import load_model_pipeline
from src.db.main import init_db

version = "v1"
setup_global_logging(Config.LOG_LEVEL)
logger = get_logger("Application_Main")

ASR_PIPELINE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ASR_PIPELINE
    logger.info("Application startup initiated. Loading models and connecting to DB...")
    await init_db()
    ASR_PIPELINE = load_model_pipeline(Config.ASR_MODEL_NAME, "automatic-speech-recognition", "production")
    yield
    logger.warning("Application is stopping")

app = FastAPI(
    version=version,
    lifespan=lifespan,
    license_info={"name": "MIT License", "url": "https://opensource.org/license/mit"},
)

