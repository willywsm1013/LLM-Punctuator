"""Data models for LLM chat messages."""

from enum import Enum

from pydantic import BaseModel, Field

# Default punctuation marks for different languages
ZH_PUNCTUATIONS = "，。？！、；："
EN_PUNCTUATIONS = ",.?!;:'"


class Role(str, Enum):
    """Chat message role enumeration."""

    System = "system"
    User = "user"
    Assistant = "assistant"


class Message(BaseModel):
    """Chat message model.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
    """

    role: str
    content: str


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
