from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(documents: List[Document]) -> List[Document]:
    splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)