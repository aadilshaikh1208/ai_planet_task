import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # FreeFlow LLM API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    
    VECTOR_DB_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 4

    OCR_CONFIDENCE_THRESHOLD = 0.7
