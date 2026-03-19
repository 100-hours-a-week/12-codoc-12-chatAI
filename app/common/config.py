import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

load_dotenv(override=True)

class Settings:
    #Runpod Serverless
    # RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
    # RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")
    # if not RUNPOD_API_KEY :
    #     raise ValueError("RUNPOD_API_KEY environment variable is not set.")
    
    # CHATBOT_MODEL_NAME : str = os.getenv("CHATBOT_MODEL_NAME", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    # TEMPERATURE : float = 0.7
    
    #Rupod Pod
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
    RUNPOD_POD_URL: str = os.getenv("RUNPOD_POD_URL", "")
    CHATBOT_MODEL_NAME: str = os.getenv("RUNPOD_MODEL_NAME", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    TEMPERATURE: float = 0.7   
    
    # Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash"
    
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "Problems")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "1024"))
    
    # PostgreSQL
    DB_HOST: str = os.getenv("DB_HOST", "")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}")
    
settings = Settings()

# LLM 초기화(임베딩 시 사용)
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL_NAME,
    temperature=settings.TEMPERATURE,
    api_key=settings.GOOGLE_API_KEY,
    max_retries=3,
    timeout=60,
)

# EXAONE 모델 변경 시 (Serverless)
# chatbot = ChatOpenAI(
#     model=settings.CHATBOT_MODEL_NAME,
#     temperature=settings.TEMPERATURE,
#     api_key = settings.RUNPOD_API_KEY,
#     base_url=os.getenv("RUNPOD_BASE_URL", ""),
#     streaming=True,
#     max_retries=3,
#     timeout=60,
# )

# EXAONE 모델 변경 시 (Pod)
chatbot = ChatOpenAI(
    model=settings.CHATBOT_MODEL_NAME,
    temperature=settings.TEMPERATURE,
    api_key = settings.RUNPOD_API_KEY,
    base_url=settings.RUNPOD_POD_URL,
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
