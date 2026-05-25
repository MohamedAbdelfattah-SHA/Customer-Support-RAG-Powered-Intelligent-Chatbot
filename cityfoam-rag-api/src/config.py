import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Resolve the absolute path to the root of the project (/app in Docker)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Security
    CITYFOAM_SECRET_KEY = os.getenv("CITYFOAM_SECRET_KEY", "default_dev_key")
    
    # Azure OpenAI
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "cityfoam-gpt")
    
    # Paths & DB Configurations (Docker Safe)
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(BASE_DIR, "chroma_db"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cityfoam_rag")
    
    @classmethod
    def validate(cls):
        if not cls.AZURE_API_KEY or not cls.AZURE_ENDPOINT:
            raise ValueError("CRITICAL: Azure API Key or Endpoint is missing from .env!")

Config.validate()