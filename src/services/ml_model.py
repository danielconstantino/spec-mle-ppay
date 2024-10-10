import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from src.config import settings
from src.models import FlightInfo, HistoryEntry
import logging
from typing import List
import os
import mongomock
import shutil

logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        self.model: BaseEstimator = None
        self.current_model_path: str = settings.CURRENT_MODEL_PATH
        self.previous_model_path: str = settings.PREVIOUS_MODEL_PATH
        self.db = mongomock.MongoClient().db
        self.predictions = self.db.predictions

    def load_initial_model(self):
        if os.path.exists(self.current_model_path):
            try:
                self.model = joblib.load(self.current_model_path)
                logger.info(f"Modelo inicial carregado de {self.current_model_path}")
            except Exception as e:
                logger.error(f"Erro ao carregar o modelo inicial: {str(e)}")
        else:
            logger.warning(f"Arquivo de modelo não encontrado em {self.current_model_path}")

    def load_new_model(self, file_path: str):
        try:
            new_model = joblib.load(file_path)
            
            # Backup do modelo atual como modelo anterior
            if os.path.exists(self.current_model_path):
                shutil.move(self.current_model_path, self.previous_model_path)
            
            # Salvar o novo modelo como modelo atual
            os.makedirs(os.path.dirname(self.current_model_path), exist_ok=True)
            shutil.copy(file_path, self.current_model_path)
            
            # Carregar o novo modelo em memória
            self.model = new_model
            
            logger.info(f"Novo modelo carregado e salvo em {self.current_model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar o novo modelo: {str(e)}")
            raise

    def rollback_model(self):
        if not os.path.exists(self.previous_model_path):
            raise FileNotFoundError("Modelo anterior não encontrado")
        
        try:
            # Carregar o modelo anterior
            previous_model = joblib.load(self.previous_model_path)
            
            # Trocar os modelos
            shutil.move(self.current_model_path, self.current_model_path + ".bak")
            shutil.move(self.previous_model_path, self.current_model_path)
            shutil.move(self.current_model_path + ".bak", self.previous_model_path)
            
            # Atualizar o modelo em memória
            self.model = previous_model
            
            logger.info("Rollback para o modelo anterior realizado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao fazer rollback do modelo: {str(e)}")
            raise

    def predict(self, input_data: FlightInfo):
        if self.model is None:
            self.load_model()
        
        # Converter input_data para um DataFrame do pandas
        input_df = pd.DataFrame([{
            'distance': input_data.distance,
            'dep_time': input_data.dep_time,
            'air_time': input_data.air_time,
            'carrier': input_data.carrier,
            'origin': input_data.origin,
            'dest': input_data.dest,
            'month': input_data.month,
            'day_of_week': input_data.day_of_week
        }])
        

        prediction = self.model.predict(input_df)[0]
        
        self.store_prediction(input_data, prediction)
        
        return prediction

    def get_info(self):
        if self.model is None:
            return None
        return {
            "name_model": type(self.model).__name__,
            "feature_names": self.model.feature_names_in_.tolist() if hasattr(self.model, 'feature_names_in_') else None,
            "num_features": len(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else None,
        }

    def store_prediction(self, input_data: FlightInfo, prediction: float):
        self.predictions.insert_one({
            "input": input_data.dict(),
            "prediction": prediction
        })

    def get_history(self) -> List[HistoryEntry]:
        history = list(self.predictions.find({}, {"_id": 0}))
        return [HistoryEntry(**entry) for entry in history]

ml_model = MLModel()