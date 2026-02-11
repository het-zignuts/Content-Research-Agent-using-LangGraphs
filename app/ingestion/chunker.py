from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split the loaded documents into smaller chunks using RecursiveCharacterTextSplitter. This function takes a list of Document objects and applies the text splitter to create smaller chunks of text that are more manageable for embedding and retrieval. The chunk size is set to 800 characters with an overlap of 150 characters to ensure that there is some context retained between chunks. The resulting list of chunked Document objects is returned for further processing in the embedding step.
    """
    splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150) # initialize the text splitter with defined chunk size and overlap
    return splitter.split_documents(documents) # apply the splitter to the list of documents and return the resulting list of chunked documents