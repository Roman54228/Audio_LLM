import time
import logging
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from tone.pipeline import StreamingCTCPipeline

# Импортируем get_pipeline из main.py
from .dependencies import get_pipeline
from celery_tasks import transcribe_audio_task, get_task_status

logger = logging.getLogger(__name__)
router = APIRouter()

# Конфигурация
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@router.post("/", response_model=List[Dict])
async def transcribe_audio_sync(
    file: UploadFile = File(...),
    language: str = Form("ru"),
    pipeline: StreamingCTCPipeline = Depends(get_pipeline)
):
    """Синхронная транскрипция аудио файла (для обратной совместимости)"""
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


@router.post("/async", response_model=Dict)
async def transcribe_audio_async(
    file: UploadFile = File(...),
    language: str = Form("ru")
):
    """Асинхронная транскрипция аудио файла через Celery"""
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой")
    
    try:
        # Читаем аудио файл
        audio_data = await file.read()
        
        # Запускаем задачу в Celery
        task = transcribe_audio_task.delay(audio_data, language)
        
        logger.info(f"Запущена асинхронная транскрипция, task_id: {task.id}")
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "Задача поставлена в очередь на обработку"
        }
        
    except Exception as e:
        logger.error(f"Ошибка запуска асинхронной транскрипции: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=Dict)
async def get_transcription_status(task_id: str):
    """Получить статус задачи транскрипции"""
    try:
        result = get_task_status.delay(task_id)
        status_info = result.get()
        
        return status_info
        
    except Exception as e:
        logger.error(f"Ошибка получения статуса задачи {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{task_id}", response_model=Dict)
async def get_transcription_result(task_id: str):
    """Получить результат транскрипции по task_id"""
    try:
        from celery_config import celery_app
        result = celery_app.AsyncResult(task_id)
        
        if result.state == "SUCCESS":
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": result.result
            }
        elif result.state == "FAILURE":
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "error": str(result.info)
            }
        else:
            return {
                "task_id": task_id,
                "status": result.state,
                "message": "Задача еще не завершена"
            }
            
    except Exception as e:
        logger.error(f"Ошибка получения результата задачи {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))