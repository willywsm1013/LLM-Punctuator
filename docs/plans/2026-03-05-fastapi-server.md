# FastAPI Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a FastAPI HTTP server that exposes LLM-Punctuator's punctuation functionality as an internal API.

**Architecture:** Single `server.py` module with FastAPI app, pydantic settings, and 3 endpoints (`POST /api/v1/punctuate`, `GET /health`, `GET /info`). Model loads at startup via lifespan. Inference runs in thread pool via `asyncio.to_thread()`.

**Tech Stack:** FastAPI, uvicorn, pydantic-settings, pytest, httpx (test client)

**Design doc:** `docs/plans/2026-03-05-fastapi-server-design.md`

---

### Task 1: Add dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add server and test dependencies**

Add `fastapi`, `uvicorn`, `pydantic-settings` to `dependencies` and `httpx`, `pytest`, `pytest-asyncio` to `dev` dependencies in `pyproject.toml`:

```toml
dependencies = [
    "transformers",
    "torch",
    "protobuf",
    "sentencepiece",
    "tqdm",
    "pydantic",
    "accelerate",
    "fastapi",
    "uvicorn",
    "pydantic-settings",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "httpx",
    "pytest",
    "pytest-asyncio",
]

[tool.uv]
dev-dependencies = [
    "ruff>=0.1.0",
    "httpx",
    "pytest",
    "pytest-asyncio",
]
```

**Step 2: Install dependencies**

Run: `uv pip install -e ".[dev]"`
Expected: Success, all packages installed.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add fastapi, uvicorn, and test dependencies"
```

---

### Task 2: Add request/response models to schema.py

**Files:**
- Modify: `llm_punctuator/schema.py`
- Create: `tests/test_schema.py`

**Step 1: Write the failing test**

Create `tests/test_schema.py`:

```python
"""Tests for request/response schema models."""

from llm_punctuator.schema import PunctuateRequest, PunctuateResponse


def test_punctuate_request_defaults():
    req = PunctuateRequest(text="hello world")
    assert req.text == "hello world"
    assert req.language == "zh"
    assert req.chunk_size == 50


def test_punctuate_request_custom():
    req = PunctuateRequest(text="hello", language="en", chunk_size=100)
    assert req.language == "en"
    assert req.chunk_size == 100


def test_punctuate_request_empty_text_rejected():
    import pytest

    with pytest.raises(Exception):
        PunctuateRequest(text="")


def test_punctuate_response():
    resp = PunctuateResponse(text="hello, world.", language="en", model="test-model")
    assert resp.text == "hello, world."
    assert resp.model == "test-model"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'PunctuateRequest'`

**Step 3: Write minimal implementation**

Add to `llm_punctuator/schema.py` (after existing code):

```python
class PunctuateRequest(BaseModel):
    """Request model for punctuation endpoint."""

    text: str = Field(..., min_length=1)
    language: str = Field(default="zh", pattern="^(zh|en)$")
    chunk_size: int = Field(default=50, gt=0)


class PunctuateResponse(BaseModel):
    """Response model for punctuation endpoint."""

    text: str
    language: str
    model: str
```

Update `__init__.py` or just rely on direct import — no change needed since tests import from `llm_punctuator.schema` directly.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add llm_punctuator/schema.py tests/test_schema.py
git commit -m "feat(schema): add PunctuateRequest and PunctuateResponse models"
```

---

### Task 3: Create server.py with settings and health/info endpoints

**Files:**
- Create: `llm_punctuator/server.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing tests**

Create `tests/test_server.py`:

```python
"""Tests for FastAPI server endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_punctuator():
    """Create a mock punctuator that avoids loading a real model."""
    mock = MagicMock()
    mock.add_punctuation.return_value = "你好，世界。"
    return mock


@pytest.fixture
def client(mock_punctuator):
    """Create a test client with mocked punctuator."""
    from llm_punctuator.server import app, get_settings

    settings = get_settings()
    app.state.punctuator = mock_punctuator
    app.state.model_loaded = True
    return TestClient(app)


def test_health_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_unhealthy():
    from llm_punctuator.server import app

    app.state.punctuator = None
    app.state.model_loaded = False
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


def test_info(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "supported_languages" in data
    assert "version" in data
    assert data["supported_languages"] == ["zh", "en"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_punctuator.server'`

**Step 3: Write minimal implementation**

Create `llm_punctuator/server.py`:

```python
"""FastAPI server for LLM Punctuator."""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Server configuration via environment variables."""

    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    default_language: str = "zh"
    default_chunk_size: int = 50
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    from llm_punctuator.punctuator import TransformersLLMPunctuator

    settings = get_settings()
    logger.info(f"Loading model: {settings.model_name_or_path}")
    try:
        app.state.punctuator = TransformersLLMPunctuator(settings.model_name_or_path)
        app.state.model_loaded = True
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model")
        app.state.punctuator = None
        app.state.model_loaded = False
    yield


app = FastAPI(title="LLM Punctuator", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    model_loaded = getattr(request.app.state, "model_loaded", False)
    if model_loaded:
        return JSONResponse({"status": "healthy", "model_loaded": True})
    return JSONResponse({"status": "unhealthy", "model_loaded": False}, status_code=503)


@app.get("/info")
async def info() -> dict:
    """Server and model information."""
    settings = get_settings()
    return {
        "model": settings.model_name_or_path,
        "supported_languages": ["zh", "en"],
        "version": "0.1.0",
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add llm_punctuator/server.py tests/test_server.py
git commit -m "feat(server): add FastAPI app with health and info endpoints"
```

---

### Task 4: Add punctuate endpoint

**Files:**
- Modify: `llm_punctuator/server.py`
- Modify: `tests/test_server.py`

**Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
def test_punctuate_zh(client, mock_punctuator):
    mock_punctuator.add_punctuation.return_value = "你好，世界。"
    response = client.post("/api/v1/punctuate", json={"text": "你好世界"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "你好，世界。"
    assert data["language"] == "zh"
    mock_punctuator.add_punctuation.assert_called_once_with(
        "你好世界", language="zh", chunk_size=50
    )


def test_punctuate_en(client, mock_punctuator):
    mock_punctuator.add_punctuation.return_value = "hello, world."
    response = client.post(
        "/api/v1/punctuate",
        json={"text": "hello world", "language": "en", "chunk_size": 100},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "hello, world."
    assert data["language"] == "en"
    mock_punctuator.add_punctuation.assert_called_once_with(
        "hello world", language="en", chunk_size=100
    )


def test_punctuate_empty_text(client):
    response = client.post("/api/v1/punctuate", json={"text": ""})
    assert response.status_code == 422


def test_punctuate_invalid_language(client):
    response = client.post(
        "/api/v1/punctuate", json={"text": "hello", "language": "fr"}
    )
    assert response.status_code == 422


def test_punctuate_model_not_loaded():
    from llm_punctuator.server import app

    app.state.punctuator = None
    app.state.model_loaded = False
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/v1/punctuate", json={"text": "hello"})
    assert response.status_code == 503
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_punctuate_zh -v`
Expected: FAIL with 404 (endpoint doesn't exist yet)

**Step 3: Write minimal implementation**

Add to `llm_punctuator/server.py`:

```python
import asyncio

from fastapi import APIRouter

from llm_punctuator.schema import PunctuateRequest, PunctuateResponse

router = APIRouter(prefix="/api/v1")


@router.post("/punctuate")
async def punctuate(request: Request, body: PunctuateRequest) -> PunctuateResponse:
    """Add punctuation to text."""
    punctuator = getattr(request.app.state, "punctuator", None)
    if punctuator is None:
        return JSONResponse(
            {"detail": "Model not loaded"}, status_code=503
        )

    settings = get_settings()
    result = await asyncio.to_thread(
        punctuator.add_punctuation,
        body.text,
        language=body.language,
        chunk_size=body.chunk_size,
    )
    return PunctuateResponse(
        text=result,
        language=body.language,
        model=settings.model_name_or_path,
    )


app.include_router(router)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add llm_punctuator/server.py tests/test_server.py
git commit -m "feat(server): add POST /api/v1/punctuate endpoint"
```

---

### Task 5: Add __main__.py entry point

**Files:**
- Create: `llm_punctuator/__main__.py`

**Step 1: Write the entry point**

Create `llm_punctuator/__main__.py`:

```python
"""Entry point for running the server with `python -m llm_punctuator`."""

import uvicorn

from llm_punctuator.server import app, get_settings


def main() -> None:
    """Run the server."""
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Verify it's importable**

Run: `python -c "from llm_punctuator.__main__ import main; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add llm_punctuator/__main__.py
git commit -m "feat(server): add __main__.py for python -m llm_punctuator"
```

---

### Task 6: Add Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY llm_punctuator/ llm_punctuator/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "llm_punctuator"]
```

**Step 2: Create .dockerignore**

```
__pycache__
*.pyc
.git
.ruff_cache
tests/
docs/
*.egg-info
```

**Step 3: Verify Dockerfile syntax**

Run: `docker build --check .` or just verify it parses (no need to actually build — model download would take too long).

**Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: add Dockerfile and .dockerignore"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Add server section to README**

Add after the "Usage" section, before "Development":

```markdown
## Server

Run as an HTTP server:

### Direct
```bash
python -m llm_punctuator
```

### With custom model
```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B PORT=8000 python -m llm_punctuator
```

### Docker
```bash
docker build -t llm-punctuator .
docker run -p 8000:8000 -e MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B llm-punctuator
```

### API Example
```bash
curl -X POST http://localhost:8000/api/v1/punctuate \
  -H "Content-Type: application/json" \
  -d '{"text": "今天天氣很好出門記得帶傘", "language": "zh"}'
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-1.7B` | HuggingFace model name or path |
| `DEFAULT_LANGUAGE` | `zh` | Default language |
| `DEFAULT_CHUNK_SIZE` | `50` | Default chunk size |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `LOG_LEVEL` | `info` | Log level |
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): add server usage and API documentation"
```

---

### Task 8: Run full test suite and lint

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run linter**

Run: `./bin/lint.sh`
Expected: No errors (or fix any issues found)

**Step 3: Fix any lint issues**

Run: `./bin/fix.sh` if needed, then re-run lint.

**Step 4: Final commit if lint fixes were needed**

```bash
git add -p  # review changes
git commit -m "style: fix lint issues"
```
