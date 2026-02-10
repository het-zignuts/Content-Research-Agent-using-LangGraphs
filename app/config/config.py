from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Config:
    """
    This class manages the configuration settings for the application, loading values from environment variables.       
    It provides a centralized place to access configuration values such as API keys, model names, and directory paths.
    """
    
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")  # API key for GROQ vector database service
    MODEL=os.getenv("MODEL", "llama-3.1-8b-instant")  # Default language model to use
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    VECTOR_DB_DIR=os.getenv("VECTOR_DB_DIR", "app/data/vector_dbs")  # Directory to store vector databases
    UPLOAD_DIR=os.getenv("UPLOAD_DIR", "app/data/uploads")  # Directory to store uploaded documents
    REPORT_STORE_DIR=os.getenv("REPORT_STORE_DIR", "app/data/reports")  # Directory to store generated reports

    UPLOAD_PATH=(REPO_ROOT/UPLOAD_DIR).resolve()
    VECTOR_DB_PATH=(REPO_ROOT/VECTOR_DB_DIR).resolve()
    REPORT_STORE_PATH=(REPO_ROOT/REPORT_STORE_DIR).resolve()

Config.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the upload directory exists
Config.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the vector database directory exists
Config.REPORT_STORE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the report store directory exists