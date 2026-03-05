"""FastAPI server for LLM Punctuator."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings

from llm_punctuator.schema import PunctuateRequest, PunctuateResponse

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
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, cleanup on shutdown."""
    from llm_punctuator.punctuator import TransformersLLMPunctuator

    settings = get_settings()
    logger.info("Loading model: %s", settings.model_name_or_path)
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


router = APIRouter(prefix="/api/v1")


@router.post("/punctuate")
async def punctuate(request: Request, body: PunctuateRequest) -> PunctuateResponse:
    """Add punctuation to text."""
    punctuator = getattr(request.app.state, "punctuator", None)
    if punctuator is None:
        return JSONResponse({"detail": "Model not loaded"}, status_code=503)

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
