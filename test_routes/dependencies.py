import time
import logging
from typing import Optional
import os

from tone.pipeline import StreamingCTCPipeline
MODEL_PATH = os.getenv("MODEL_PATH", "/models")

logger = logging.getLogger(__name__)

# Глобальная переменная для pipeline
pipeline: Optional[StreamingCTCPipeline] = None
start_time = time.time()

async def get_pipeline() -> StreamingCTCPipeline:
    """Получить pipeline (ленивая инициализация)"""
    global pipeline
    
    if pipeline is None:
        logger.info(f"Загружаем модель из {MODEL_PATH}")
        pipeline = StreamingCTCPipeline.from_local(MODEL_PATH)
        logger.info("Pipeline успешно загружен")
    
    return pipeline

def get_start_time() -> float:
    """Получить время запуска приложения"""
    return start_time