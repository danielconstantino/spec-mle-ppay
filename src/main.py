from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import predict, model, health
from src.config import settings
from src.services.ml_model import ml_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Case Machine Learning Engineer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/model", tags=["predict"])
app.include_router(model.router, prefix="/model", tags=["model"])
app.include_router(health.router, tags=["health"])

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando a aplicação...")
    if settings.CURRENT_MODEL_PATH:
        try:
            ml_model.load_initial_model()
            logger.info(f"Modelo carregado automaticamente de {settings.CURRENT_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo na inicialização: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Encerrando a aplicação...")
    # Adicione aqui qualquer limpeza necessária

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)