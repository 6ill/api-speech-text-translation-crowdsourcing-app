from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.auth.router import router as auth_router
from src.api.corrections.router import router as corrections_router
from src.api.files.router import router as files_router
from src.api.inference.router import router as inference_router
from src.api.people.router import router as people_router
from src.api.pipeline.router import router as pipeline_router


from src.core.config import Config
from src.core.errors import register_all_errors
from src.core.logging import get_logger, setup_global_logging
from src.core.middlewares import register_all_middlewares
from src.core.mlflow_client import load_model_pipeline
from src.db.main import init_db

version = "v1"
setup_global_logging(Config.LOG_LEVEL)
logger = get_logger("Application_Main")
url_base = f"/api/{version}"

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

register_all_errors(app)
register_all_middlewares(app)

app.include_router(auth_router, prefix=url_base+"/auth", tags=["Auth"])
app.include_router(inference_router, prefix=url_base+"/inference", tags=["Inference"])
app.include_router(files_router,prefix=url_base+"/files", tags=["Files"])
app.include_router(people_router,prefix=url_base+"/people", tags=["People"])
app.include_router(pipeline_router,prefix=url_base+"/pipeline", tags=["Pipeline"])
app.include_router(corrections_router, prefix=url_base+"/corrections", tags=["Corrections"])