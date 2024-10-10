from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ALLOWED_ORIGINS: str = "*"
    MODEL_PATH: str = "/app/models/current_model.pkl"
    CURRENT_MODEL_PATH: str = "/app/models/current_model.pkl"
    PREVIOUS_MODEL_PATH: str = "/app/models/previous_model.pkl"
    MAX_HISTORY_SIZE: int = 100

    class Config:
        env_file = ".env"

settings = Settings()