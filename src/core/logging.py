import logging
from logging import LogRecord
import sys
import time
from typing import Any


LOG_FORMAT = (
    "%(levelname)s:     %(asctime)s | "
    "Module:%(name)s | "
    "Process:%(process)d | "
    "%(message)s"
)

class CustomLogFormatter(logging.Formatter):
    """
    Custom formatter to ensure ISO 8601 format with UTC timezone.
    """
    def formatTime(self, record: LogRecord, datefmt: str | None = None) -> str:
        # Use ISO 8601 format with milliseconds and UTC timezone
        dt_struct = self.converter(record.created)
        # 2025-11-14T15:30:01.123Z (similar to ISO 8601)

        time_part = time.strftime("%Y-%m-%dT%H:%M:%S", dt_struct)
        
        return time_part + f".{int(record.msecs):03d}Z"


def setup_global_logging(log_level: str = "INFO"):
    """
    Sets up global logging configuration for the application.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    
    formatter = CustomLogFormatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING) 
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)


setup_global_logging() 

def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance for a specific module/service."""
    return logging.getLogger(name)