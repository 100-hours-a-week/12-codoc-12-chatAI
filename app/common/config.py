import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient

load_dotenv()

class Settings:
    # Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    MODEL_NAME: str = "gemini-2.0-flash"
    TEMPERATURE: float = 0.7
    
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "Problems")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "384"))
    
settings = Settings()

# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    api_key=settings.GOOGLE_API_KEY,
    streaming=True,
    max_retries=3,
    timeout=60,
)

# Qdrant 클라이언트 초기화
qdrant_client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
)

QDRANT_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
QDRANT_API_KEY = settings.QDRANT_API_KEY