from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Server is running with Python 3.12!"}

@app.get("/test-langchain")
def test_chain():
    # LangChain 동작 테스트 (실제 LLM 호출 없이 구조만 확인)
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    return {"prompt_preview": prompt.format(topic="coding")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)