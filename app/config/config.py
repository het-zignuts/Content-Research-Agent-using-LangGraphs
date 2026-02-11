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
    
    REPO_ROOT=Path(__file__).resolve().parents[2] # Root directory of the repository, used as a base for constructing paths to data directories
    VECTOR_DB_DIR=os.getenv("VECTOR_DB_DIR", "app/data/vector_dbs")  # Directory to store vector databases
    UPLOAD_DIR=os.getenv("UPLOAD_DIR", "app/data/uploads")  # Directory to store uploaded documents
    REPORT_STORE_DIR=os.getenv("REPORT_STORE_DIR", "app/data/reports")  # Directory to store generated reports

    UPLOAD_PATH=(REPO_ROOT/UPLOAD_DIR).resolve() # Full path to the upload directory, resolved from the repository root and the upload directory name
    VECTOR_DB_PATH=(REPO_ROOT/VECTOR_DB_DIR).resolve() # Full path to the vector database directory, resolved from the repository root and the vector database directory name
    REPORT_STORE_PATH=(REPO_ROOT/REPORT_STORE_DIR).resolve() # Full path to the report store directory, resolved from the repository root and the report store directory name

Config.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the upload directory exists
Config.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the vector database directory exists
Config.REPORT_STORE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the report store directory exists