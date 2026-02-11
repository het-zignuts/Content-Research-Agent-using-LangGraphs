import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.config import Config

class VectorDB:
    """
    This class manages a vector database for storing and retrieving semantic embeddings.
     It uses FAISS for efficient similarity search and HuggingFaceEmbeddings for generating vectors.
     The database is stored locally on disk and can be loaded or created as needed.
     Each session has its own vector database file, identified by the session_id.
    """

    def __init__(self, embed_model_name="sentence-transformers/all-MiniLM-L6-v2", db_path=Config.VECTOR_DB_PATH, session_id=None):
        """
        Initialize the VectorDB with the specified embedding model, database path, and session ID.
        """
        self.embed_model=HuggingFaceEmbeddings(model_name=embed_model_name) # Initialize the embedding model
        self.db_path=os.path.join(db_path, f"{session_id}_vector_db") # Set the path for the vector database file based on the session ID
        self.session_id=session_id # Store the session ID for reference
        self.vector_db=None # Initialize the vector database attribute, which will hold the FAISS index instance

    def load_db(self):
        """
        Load the vector database from disk if it exists, otherwise create a new one.
        """
        if self.db_path and os.path.exists(self.db_path): # Check if the database file exists at the specified path
            self.vector_db=FAISS.load_local(self.db_path, self.embed_model, allow_dangerous_deserialization=True) # Load the existing vector database using FAISS and the embedding model
        else:
            self.vector_db=None # If the database file does not exist, set the vector_db attribute to None, indicating that a new database will need to be created when documents are added

    def add_documents(self, docs):
        """
        Add documents to the vector database, generating embeddings and saving the updated database to disk.
        """
        if not docs:
            return None # If no documents are provided, return None to indicate that no action was taken
        if self.vector_db is None:
            self.vector_db=FAISS.from_documents(docs, self.embed_model) # If the vector database has not been initialized (i.e., it was not loaded from disk), create a new FAISS index from the provided documents and the embedding model
        else:
            self.vector_db.add_documents(docs) # If the vector database already exists, add the new documents to the existing FAISS index, which will generate embeddings for the new documents and update the index accordingly
        self.vector_db.save_local(self.db_path) # Save the updated vector database to disk at the specified path, ensuring that the new documents and their embeddings are persisted for future retrieval
