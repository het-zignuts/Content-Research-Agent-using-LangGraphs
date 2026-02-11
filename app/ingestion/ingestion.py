from app.ingestion.loader import load_documents
from app.ingestion.chunker import split_documents
from app.ingestion.embed import embed_documents
from typing import List

def ingest_docs(session_id: str, file_paths: List[str]):
    """
    Ingest documents for a given session. 
    This function orchestrates the entire ingestion process by first loading the documents from the specified file paths, then splitting them into smaller chunks using the defined chunking strategy, and finally embedding the chunked documents and storing them in the vector database for efficient retrieval during query processing in the LangGraph. 
    The session_id is used to associate the ingested documents with a specific user session, allowing for personalized document management and retrieval based on the user's interactions with the Content Research Agent.
    """
    documents=load_documents(session_id, file_paths) # load docs from file paths
    chunked_docs=split_documents(documents) # chunk the loaded documents into smaller pieces for better embedding and retrieval performance
    embed_documents(session_id, chunked_docs) # embed the chunked documents and store them in the vector database for the session, making them available for retrieval during query processing in the LangGraph