from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    GEMINI_API_KEY : str = os.getenv("gemini_api_key")

    class Config:
        env_file = ".env"

settings = Settings()