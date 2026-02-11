from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_documents(session_id: str, file_paths: List[str]) -> List[Document]:
    documents: List[Document]=[]

    for idx, path in enumerate(file_paths):
        path=Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.suffix.lower()==".txt":
            text = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "doc_id": f"doc_{idx}",
                        "doc_name": path.name,
                        "session_id": session_id,
                        "page": 1
                    }
                )
            )

        elif path.suffix.lower()==".pdf":
            loader=PyPDFLoader(str(path))
            pages=loader.load() 
            for page in pages:
                documents.append(
                    Document(
                        page_content=page.page_content,
                        metadata={
                            **page.metadata,   
                            "doc_id": f"doc_{idx}",
                            "doc_name": path.name,
                            "session_id": session_id
                        }
                    )
                )
        else:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. Only .txt and .pdf are supported."
            )
    return documents
