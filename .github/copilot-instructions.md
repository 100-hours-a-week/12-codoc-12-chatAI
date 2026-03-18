# AI Coding Agent Instructions for ktb3-codoc-AI

## Project Overview
**ktb3-codoc-AI** is a FastAPI-based backend service integrating LangChain for LLM interactions with Qdrant vector database. The project purpose: provide document/code analysis through AI with RAG (Retrieval-Augmented Generation) capabilities.

## Architecture

### Core Components
- **FastAPI App** (`app/main.py`): HTTP API server running on port 8000
  - Simple structure: currently single endpoint demonstrating pattern
  - Uses `ChatPromptTemplate` from LangChain for structured prompts
  - Entry point: `if __name__ == "__main__"` runs uvicorn server

### External Services
- **Qdrant Vector DB**: Remote vector database for semantic search/embeddings
  - Configured via `QDRANT_URL` (default: http://localhost:6333)
  - Connection via `langchain-qdrant` package
  - Local dev setup expected (see `.env.template`)

- **OpenAI API**: LLM backend via `langchain-openai`
  - Requires API key in environment (inferred from dependencies)

## Development Setup

### Environment Configuration
1. Copy `.env.template` to `.env` and populate:
   - `QDRANT_URL`: Qdrant instance endpoint
   - `QDRANT_API_KEY`: Authentication if required
   - `ENV_MODE`: Set to "dev" for development

2. Conda environment expected (current: `env_codoc`)
   ```bash
   conda activate env_codoc
   pip install -r requirements.txt
   ```

### Running the Server
```bash
python app/main.py
# Server listens on http://0.0.0.0:8000
```

## Code Patterns & Conventions

### FastAPI Endpoints
- Decorator-based routing: `@app.get("/")`, `@app.post("/path")`
- Return plain dicts (not Pydantic models in minimal examples)
- Add proper request/response models for production endpoints

### LangChain Integration
- Import from `langchain_core.prompts` for prompt templates
- Pattern: `ChatPromptTemplate` for structured LLM interactions
- Avoid direct LLM calls; route through prompt templates for consistency

### Environment Variables
- Use `python-dotenv` for local development
- `.env` excluded from git (see `.gitignore`)
- Template pattern: `.env.template` documents required vars

## Key Dependencies
- **fastapi[standard]**: Web framework with validation
- **langchain + langchain-openai + langchain-qdrant**: LLM orchestration
- **qdrant-client**: Vector DB Python client
- **uvicorn**: ASGI server
- **python-dotenv**: Environment configuration

## Important Notes
- **Branch**: Development on `dev` branch (repo info indicates current branch)
- **Vector DB**: Qdrant setup is prerequisite—integration assumes running instance
- **LLM Cost**: OpenAI API calls incur costs; watch usage in development
- **Minimal Codebase**: Project is early stage—expect rapid expansion; follow established patterns

## When Adding Features
1. Define FastAPI route with proper models (request/response)
2. Use LangChain's `ChatPromptTemplate` for prompt engineering
3. Update `.env.template` if introducing new env vars
4. Maintain separation: API layer → LangChain orchestration → External services
5. Test locally with Qdrant running; logs via FastAPI's stdout

---
*Last updated: 2026-01-21*
