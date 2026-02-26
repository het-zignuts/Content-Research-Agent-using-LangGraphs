from app.config.config import Config
from langchain_groq import ChatGroq
import os

def get_groq_llm(temperature=0.0, **kwargs):
    """
    Initialize and return a ChatGroq LLM instance using environment settings.
    """
    return ChatGroq(
        model=Config.MODEL, # default model
        max_completion_tokens=4096,
        temperature=temperature, 
        api_key=Config.GROQ_API_KEY,
        **kwargs
    )