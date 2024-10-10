from fastapi import APIRouter, HTTPException
from src.services.ml_model import ml_model
import psutil
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    try:
        model_status = "loaded" if ml_model.model is not None else "not loaded"
        memory_usage = psutil.virtual_memory().percent
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "memory_usage": f"{memory_usage}%"
        }
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao realizar health check")