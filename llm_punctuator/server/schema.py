"""Request/response models for the punctuation API."""

from pydantic import BaseModel, Field


class PunctuateRequest(BaseModel):
    """Request model for punctuation endpoint."""

    text: str = Field(..., min_length=1)
    language: str | None = Field(default=None, pattern="^(zh|en)$")
    chunk_size: int | None = Field(default=None, gt=0)


class PunctuateResponse(BaseModel):
    """Response model for punctuation endpoint."""

    text: str
    language: str
    model: str
