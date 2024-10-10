from pydantic import BaseModel
from typing import List, Optional

class FlightInfo(BaseModel):
    distance: float
    dep_time: int
    air_time: float
    carrier: str
    origin: str
    dest: str
    month: int
    day_of_week: int

class PredictionResult(BaseModel):
    delay_prediction: float

class HistoryEntry(BaseModel):
    input: FlightInfo
    prediction: float

class ModelInfo(BaseModel):
    name_model: str
    feature_names: Optional[List[str]]
    num_features: Optional[int]