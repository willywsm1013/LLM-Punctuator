"""FastAPI server for LLM Punctuator."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from functools import lru_cache

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings

from llm_punctuator.server.schema import PunctuateRequest, PunctuateResponse

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model lifecycle states."""

    loading = "loading"
    ready = "ready"
    failed = "failed"


class Settings(BaseSettings):
    """Server configuration via environment variables."""

    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    default_language: str = "zh"
    default_chunk_size: int = 200
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


async def _load_model(app: FastAPI) -> None:
    """Load model in background thread."""
    from llm_punctuator.punctuator import TransformersLLMPunctuator

    settings = get_settings()
    logger.info("Loading model: %s", settings.model_name_or_path)
    try:
        punctuator = await asyncio.to_thread(TransformersLLMPunctuator, settings.model_name_or_path)
        app.state.punctuator = punctuator
        app.state.model_status = ModelStatus.ready
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model")
        app.state.model_status = ModelStatus.failed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start model loading in background, server accepts requests immediately."""
    app.state.punctuator = None
    app.state.model_status = ModelStatus.loading
    load_task = asyncio.create_task(_load_model(app))
    yield
    load_task.cancel()


app = FastAPI(title="LLM Punctuator", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    status = getattr(request.app.state, "model_status", ModelStatus.failed)
    if status is ModelStatus.ready:
        return JSONResponse({"status": "healthy", "model_status": status.value})
    return JSONResponse({"status": status.value, "model_status": status.value}, status_code=503)


@app.get("/info")
async def info() -> dict:
    """Server and model information."""
    settings = get_settings()
    return {
        "model": settings.model_name_or_path,
        "supported_languages": ["zh", "en"],
        "version": "0.1.0",
    }


router = APIRouter(prefix="/api/v1")


@router.post("/punctuate")
async def punctuate(request: Request, body: PunctuateRequest) -> PunctuateResponse:
    """Add punctuation to text."""
    punctuator = getattr(request.app.state, "punctuator", None)
    if punctuator is None:
        return JSONResponse({"detail": "Model not loaded"}, status_code=503)

    settings = get_settings()
    language = body.language or settings.default_language
    chunk_size = body.chunk_size or settings.default_chunk_size
    result = await asyncio.to_thread(
        punctuator.add_punctuation,
        body.text,
        language=language,
        chunk_size=chunk_size,
    )
    return PunctuateResponse(
        text=result,
        language=language,
        model=settings.model_name_or_path,
    )


app.include_router(router)
