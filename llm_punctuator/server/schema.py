"""Request/response models for the punctuation API."""

from pydantic import BaseModel, Field, field_validator

from llm_punctuator.schema import ALLOWED_PUNCTUATIONS


class PunctuateRequest(BaseModel):
    """Request model for punctuation endpoint."""

    text: str = Field(..., min_length=1)
    language: str | None = Field(default=None, pattern="^(zh|en)$")
    chunk_size: int | None = Field(default=None, gt=0)
    punctuations: str | None = Field(default=None, min_length=1)

    @field_validator("punctuations")
    @classmethod
    def validate_punctuations(cls, v: str | None) -> str | None:
        """Validate that all characters are allowed punctuation marks."""
        if v is None:
            return v
        invalid = set(v) - ALLOWED_PUNCTUATIONS
        if invalid:
            raise ValueError(f"Invalid punctuation characters: {''.join(sorted(invalid))}")
        return v


class PunctuateResponse(BaseModel):
    """Response model for punctuation endpoint."""

    text: str
    language: str
    model: str
