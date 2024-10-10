from fastapi import APIRouter, HTTPException
from src.models import FlightInfo, PredictionResult
from src.services.ml_model import ml_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict/", response_model=PredictionResult)
async def predict_delay(flight_info: FlightInfo):
    try:
        logger.info(f"Recebida solicitação de previsão para: {flight_info}")
        
        # Fazer a previsão
        prediction = ml_model.predict(flight_info)
        
        return PredictionResult(delay_prediction=float(prediction))
    except Exception as e:
        logger.error(f"Erro ao fazer previsão: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a previsão")