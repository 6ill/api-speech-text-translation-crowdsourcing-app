from celery import Celery
from kombu import Queue
from src.core.config import Config

# Initialize Celery
celery_app = Celery(
    "tasks",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_BROKER_URL, 
    include=[
        "src.workers.inference_tasks", 
        # "src.workers.pipeline_tasks"
    ],
)

# Define the task queues
celery_app.conf.task_queues = (
    Queue("inference_queue", routing_key="inference.#"),
    Queue("ml_pipeline_queue", routing_key="pipeline.#"),
)

# Default queue settings
celery_app.conf.task_default_queue = "inference_queue"
celery_app.conf.task_default_routing_key = "inference.default"

# Set timezone
celery_app.conf.timezone = "UTC"