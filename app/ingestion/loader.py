from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_documents(session_id: str, file_paths: List[str]) -> List[Document]:
    """
    Load documents from the specified file paths. 
    This function supports both .txt and .pdf file formats. 
    For .txt files, it reads the content directly and creates a Document object with the text content and associated metadata. 
    For .pdf files, it uses the PyPDFLoader to load the PDF and extract its pages as separate Document objects, each containing the page content and metadata. The metadata includes information such as the source file path, document ID, document name, session ID, and page number for PDFs. The resulting list of Document objects is returned for further processing in the ingestion pipeline.
    """
    documents: List[Document]=[] # initialize an empty list to store the loaded Document objects

    for idx, path in enumerate(file_paths):
        path=Path(path) # get the path
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}") # check if path exists otherwise raise error

        # handling the text format files
        if path.suffix.lower()==".txt":
            text=path.read_text(encoding="utf-8") # extract text
            # create a Document object with the text content and associated metadata.
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

        # Handling PDF files
        elif path.suffix.lower()==".pdf":
            loader=PyPDFLoader(str(path)) # load pdf file using PyPDFLoader
            pages=loader.load()  # extract pages from the pdf with metadata
            for page in pages:
                # for each page, prepare a Document object with additional metadata
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
            # raise error if the file type is not supported
            raise ValueError(
                f"Unsupported file type: {path.suffix}. Only .txt and .pdf are supported."
            )
    return documents
