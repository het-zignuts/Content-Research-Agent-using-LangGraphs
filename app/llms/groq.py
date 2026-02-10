from app.config.config import Config
from langchain_groq import ChatGroq
import os

def get_groq_llm(temperature=0.0):
    """
    Initialize and return a ChatGroq LLM instance using environment settings.
    """
    return ChatGroq(
        model=Config.MODEL, # default model
        temperature=temperature, 
        api_key=Config.GROQ_API_KEY
    )