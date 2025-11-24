from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.files.router import router as files_router
from src.api.inference.router import router as inference_router
from src.api.pipeline.router import router as pipeline_router


from src.core.config import Config
from src.core.logging import get_logger, setup_global_logging
from src.core.mlflow_client import load_model_pipeline
from src.db.main import init_db

version = "v1"
setup_global_logging(Config.LOG_LEVEL)
logger = get_logger("Application_Main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated. connecting to DB...")
    await init_db()
    yield
    logger.warning("Application is stopping")

app = FastAPI(
    version=version,
    lifespan=lifespan,
    license_info={"name": "MIT License", "url": "https://opensource.org/license/mit"},
)

app.include_router(inference_router, prefix=f"/api/{version}/inference", tags=["Inference"])
app.include_router(files_router,prefix=f"/api/{version}/files", tags=["Files"])
app.include_router(pipeline_router,prefix=f"/api/{version}/pipeline", tags=["Pipeline"])
