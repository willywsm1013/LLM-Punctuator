# FastAPI Server Design

## Context

LLM-Punctuator is a CLI tool that adds punctuation to unpunctuated text (ASR outputs, transcripts) using constrained LLM generation. This design adds a FastAPI HTTP server to expose the punctuation functionality as an internal API for team systems to integrate with.

## Scope

- Small production service for internal team integration
- Single model loaded at startup via config/env vars
- Synchronous API (no streaming, no async job queue)
- Docker support for deployment

## API Design

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/punctuate` | Add punctuation to text |
| `GET` | `/health` | Health check with model status |
| `GET` | `/info` | Server and model information |

### `POST /api/v1/punctuate`

Request:
```json
{
    "text": "今天天氣很好出門記得帶傘",
    "language": "zh",
    "chunk_size": 50
}
```

- `text` (required): Unpunctuated input text
- `language` (optional, default: `"zh"`): Language code (`"zh"` or `"en"`)
- `chunk_size` (optional, default: `50`): Tokens per processing chunk

Response:
```json
{
    "text": "今天天氣很好，出門記得帶傘。",
    "language": "zh",
    "model": "Qwen/Qwen3-1.7B"
}
```

### `GET /health`

Response 200:
```json
{"status": "healthy", "model_loaded": true}
```

Response 503:
```json
{"status": "unhealthy", "model_loaded": false}
```

### `GET /info`

Response:
```json
{
    "model": "Qwen/Qwen3-1.7B",
    "supported_languages": ["zh", "en"],
    "version": "0.1.0"
}
```

## Architecture

### Approach: Flat module

Add a single `llm_punctuator/server.py` containing the FastAPI app, settings, and all endpoints. This matches the existing flat package structure and is appropriate for 3 endpoints.

### File Changes

| File | Action | Description |
|------|--------|-------------|
| `llm_punctuator/server.py` | New | FastAPI app, settings, endpoints |
| `llm_punctuator/schema.py` | Modify | Add request/response Pydantic models |
| `pyproject.toml` | Modify | Add `fastapi`, `uvicorn` dependencies |
| `Dockerfile` | New | Container image for deployment |
| `README.md` | Modify | Add server usage docs |

### Configuration

Environment variables with pydantic `BaseSettings`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-1.7B` | HuggingFace model name or local path |
| `DEFAULT_LANGUAGE` | `zh` | Default language |
| `DEFAULT_CHUNK_SIZE` | `50` | Default chunk size |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `LOG_LEVEL` | `info` | Log level |

### Model Lifecycle

Use FastAPI `lifespan` context manager to load model at startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.punctuator = TransformersLLMPunctuator(settings.model_name_or_path)
    yield
```

### Concurrency

Model inference is CPU/GPU bound. Use `asyncio.to_thread()` to run inference in a thread pool, preventing event loop blocking.

### Deployment

Development:
```bash
uvicorn llm_punctuator.server:app --reload
```

Docker:
```bash
docker build -t llm-punctuator .
docker run -p 8000:8000 -e MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B llm-punctuator
```

## Decisions

- **No streaming**: Synchronous endpoint only. Can be added later if needed.
- **No authentication**: Internal service, auth handled at network/infra level.
- **No rate limiting**: Internal service, not needed at this scale.
- **`/api/v1/` prefix**: For business endpoints only. `/health` and `/info` at root level.
- **Flat module**: Single `server.py` file, no sub-package. Appropriate for 3 endpoints.
