"""Data models for LLM chat messages."""

from enum import Enum

from pydantic import BaseModel

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
