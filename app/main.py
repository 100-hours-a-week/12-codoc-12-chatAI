import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from app.common.exceptions.exception_handler import register_exception_handlers
from app.common.config import llm
from app.domain.chatbot.bot_router import router as bot_router
import os

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Problems")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")  # 도커 내부 통신용 
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작! Qdrant 컬렉션 체크 중...")
    
    # 도커 내부끼리 통신하므로 host 이름은 'qdrant'가 됨.
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
    )
    
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"컬렉션 '{COLLECTION_NAME}' 생성 완료!")
    else:
        print(f"컬렉션 '{COLLECTION_NAME}'이 이미 존재합니다.")
    
    yield # 여기서부터 서버 가동
    
    print("서버 종료")
    
app = FastAPI()

register_exception_handlers(app)

@app.get("/")
def read_root():
    return {"message": "Hello World!"} 

@app.get("/healthcheck")
def health_check():
    return {"status": "ok"}

app.include_router(bot_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)