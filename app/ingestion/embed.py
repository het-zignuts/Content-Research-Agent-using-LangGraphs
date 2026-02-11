from app.db.vector_db import VectorDB
from app.config.config import Config
from langchain_core.documents import Document
from typing import List

def embed_documents(session_id: str, documents: List[Document]):
    """
    Embed the chunked documents and store them in the vector database. 
    This function initializes the VectorDB with the specified embedding model and database path for the given session. 
    It then loads the existing vector database or creates a new one if it doesn't exist. The provided list of chunked Document objects is added to the vector database, which generates embeddings for each document and stores them for efficient retrieval during query processing in the LangGraph.
    """
    vector_db=VectorDB(embed_model_name="sentence-transformers/all-MiniLM-L6-v2", db_path=Config.VECTOR_DB_PATH, session_id=session_id)
    vector_db.load_db() # Load or create the vector database for the session
    vector_db.add_documents(documents) # Add the documents to the vector database, which will generate embeddings and store them for retrieval