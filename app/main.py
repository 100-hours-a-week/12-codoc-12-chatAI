import asyncio
from fastapi import FastAPI
from fastapi import Depends
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.common.api_response import CommonResponse
from app.common.exceptions.exception_handler import register_exception_handlers
from app.common.config import llm, settings
from app.common.db import get_db
from app.domain.chatbot.bot_router import router as bot_router
from app.logging_config import setup_logging
from app.middleware.request_logging import request_logging_middleware
import os

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))


_mcp_process: asyncio.subprocess.Process | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_process

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

    # MCP 서버 subprocess 시작
    mcp_port = os.getenv("MCP_SERVER_PORT", "8001")
    try:
        _mcp_process = await asyncio.create_subprocess_exec(
            "python", "-m", "app.mcp.server",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.sleep(1.5)  # 서버 기동 대기
        print(f"✅ MCP 서버 시작 완료 (PID: {_mcp_process.pid}, port: {mcp_port})")
    except Exception as e:
        print(f"⚠️ MCP 서버 시작 실패 (QUESTION 기능 비활성화): {e}")

    yield  # 서버 가동

    # MCP 서버 종료
    if _mcp_process and _mcp_process.returncode is None:
        _mcp_process.terminate()
        await _mcp_process.wait()
        print("🛑 MCP 서버 종료 완료")

    print("서버 종료")
    
docs_enabled = os.getenv("DOCS_ENABLED", "true").lower() == "true"
    
app = FastAPI(
    lifespan=lifespan,
    title="CodoC",
    
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# 요청 완료 시 JSON + 텍스트 로그 기록
app.middleware("http")(request_logging_middleware)

register_exception_handlers(app)

@app.get("/")
def read_root():
    return {"message": "Hello World!"} 

@app.get("/healthcheck")
def health_check():
    return {"status": "ok"}

@app.get("/ping")
def ping():
    return {"status":"healthy"}


@app.get("/health/db")
async def db_health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return CommonResponse.success_response(
            message="Database connection is healthy.",
            data={"db": "connected"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=CommonResponse.fail_response(
                code="DB_CONNECTION_FAILED",
                message=f"Database connection failed: {str(e)}",
            ).model_dump(),
        )


app.include_router(bot_router, prefix="/api/v2")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
