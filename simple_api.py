"""
Простой API для T-one ASR сервиса
Только синхронная обработка без дополнительных зависимостей
"""

import logging
import os
import time
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from tone.pipeline import StreamingCTCPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
MODEL_PATH = os.getenv("MODEL_PATH", "/models")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB

# FastAPI приложение
app = FastAPI(
    title="T-one ASR API",
    description="Простой API для распознавания речи",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для pipeline
pipeline: StreamingCTCPipeline = None

# Модели данных
class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    uptime: float

# Инициализация pipeline при запуске
@app.on_event("startup")
async def startup_event():
    """Инициализация pipeline при запуске приложения"""
    global pipeline
    try:
        logger.info(f"Загружаем модель из {MODEL_PATH}")
        pipeline = StreamingCTCPipeline.from_local(MODEL_PATH)
        logger.info("Pipeline успешно загружен")
    except Exception as e:
        logger.error(f"Ошибка загрузки pipeline: {e}")
        raise

# Эндпоинты
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса"""
    return HealthResponse(
        status="healthy",
        pipeline_loaded=pipeline is not None,
        uptime=time.time() - start_time
    )

@app.post("/transcribe", response_model=List[Dict])
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("ru")
):
    """Синхронная транскрипция аудио файла"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline не загружен")
    
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой")
    
    try:
        # Читаем аудио файл
        audio_data = await file.read()
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.int32)
        
        # Обрабатываем
        start_time = time.time()
        result = pipeline.forward_offline(audio_array)
        processing_time = time.time() - start_time
        
        # Конвертируем результат
        result_data = [
            {
                "text": phrase.text,
                "start_time": phrase.start_time,
                "end_time": phrase.end_time
            }
            for phrase in result
        ]
        logger.info(f"Транскрипция завершена за {processing_time:.2f}с")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Глобальная переменная для отслеживания времени запуска
start_time = time.time()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
