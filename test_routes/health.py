from fastapi import APIRouter, Depends
from test_routes.dependencies import get_pipeline, get_start_time
from tone.pipeline import StreamingCTCPipeline
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    uptime: float
    
router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check(
    pipeline: StreamingCTCPipeline = Depends(get_pipeline),
    start_time: float = Depends(get_start_time)
):
    """Проверка состояния сервиса"""
    import time
    return HealthResponse(
        status="healthy",
        pipeline_loaded=pipeline is not None,
        uptime=time.time() - start_time
    )