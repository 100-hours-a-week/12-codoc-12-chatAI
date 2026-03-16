import asyncio
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from app.common.exceptions.exception_handler import register_exception_handlers
from app.common.config import llm, settings
from app.domain.chatbot.bot_router import router as bot_router
from app.logging_config import setup_logging
from app.middleware.request_logging import request_logging_middleware
from app.observability import PrometheusMiddleware, metrics, setting_otlp

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
APP_NAME = os.getenv("APP_NAME", "app_chatai")
OTLP_GRPC_ENDPOINT = os.getenv("OTLP_GRPC_ENDPOINT", "").strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    print(f"서버 시작! Qdrant({settings.QDRANT_HOST}) 연결 체크 중...")  
      
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
    )

    if not client.collection_exists(settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"컬렉션 '{settings.COLLECTION_NAME}' 생성 완료!")
    else:
        print(f"컬렉션 '{settings.COLLECTION_NAME}'이 이미 존재합니다.")

    yield # 여기서부터 서버 가동
    
    print("서버 종료")
    
docs_enabled = os.getenv("DOCS_ENABLED", "true").lower() == "true"
    
app = FastAPI(
    title="CodoC",
    
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None
)

# Prometheus metrics
app.add_middleware(PrometheusMiddleware, app_name=APP_NAME)
app.add_route("/metrics", metrics)
if OTLP_GRPC_ENDPOINT:
    setting_otlp(app, APP_NAME, OTLP_GRPC_ENDPOINT)

# 요청 완료 시 JSON + 텍스트 로그 기록
app.middleware("http")(request_logging_middleware)

register_exception_handlers(app)

@app.get("/")
def read_root():
    return {"message": "Hello World!"} 

@app.get("/healthcheck")
def health_check():
    return {"status": "ok"}

app.include_router(bot_router, prefix="/api/v2")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
