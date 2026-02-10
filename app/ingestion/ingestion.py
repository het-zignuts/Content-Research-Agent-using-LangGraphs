from app.ingestion.loader import load_documents
from app.ingestion.chunker import split_documents
from app.ingestion.embed import embed_documents

def ingest_docs(session_id: str, file_paths: List[str]):
    documents = load_documents(session_id, file_paths)
    chunked_docs = split_documents(documents)
    embed_documents(session_id, chunked_docs)