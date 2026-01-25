import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from app.common.exceptions.exception_handler import register_exception_handlers
from app.common.config import llm
from app.domain.chatbot.bot_router import router as bot_router

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