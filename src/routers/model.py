from fastapi import APIRouter, HTTPException, UploadFile, File
from src.services.ml_model import ml_model
from src.models import ModelInfo, HistoryEntry
from src.config import settings
import logging
import os
import json


router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/load/")
async def load_model(file: UploadFile = File(...)):
    try:
        logger.info(f"Recebido arquivo de modelo: {file.filename}")
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        ml_model.load_new_model(temp_file_path)
        os.remove(temp_file_path)
        
        return {"message": "Novo modelo carregado e salvo com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao carregar o novo modelo: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao carregar o novo modelo")

@router.post("/rollback/")
async def rollback_model():
    try:
        ml_model.rollback_model()
        return {"message": "Rollback para o modelo anterior realizado com sucesso"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo anterior não encontrado")
    except Exception as e:
        logger.error(f"Erro ao fazer rollback do modelo: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao fazer rollback do modelo")

@router.get("/info", response_model=ModelInfo)
async def model_info():
    info = ml_model.get_info()
    if info is None:
        raise HTTPException(status_code=404, detail="Modelo não carregado")
    return ModelInfo(**info)

@router.get("/history/", response_model=list[HistoryEntry])
async def get_prediction_history():
    try:
        logger.info("Solicitação de histórico de previsões recebida")
        return ml_model.get_history()
    except Exception as e:
        logger.error(f"Erro ao recuperar histórico: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao recuperar histórico de previsões")
    
@router.get("/metrics")
async def get_model_metrics():
    metrics_path = "/app/models/model_metrics.json"
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return metrics
        else:
            raise HTTPException(status_code=404, detail="Model metrics not found")
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving model metrics")