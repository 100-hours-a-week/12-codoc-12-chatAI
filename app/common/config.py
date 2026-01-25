import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    MODEL_NAME: str = "gemini-2.0-flash"
    TEMPERATURE: float = 0.7
    
settings = Settings()

llm = ChatGoogleGenerativeAI(
    model=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    api_key=settings.GOOGLE_API_KEY,
    streaming=True
)