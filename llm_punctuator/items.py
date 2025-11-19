from pydantic import BaseModel
from enum import Enum

class Role(str, Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"

class Message(BaseModel):
    role: str
    content: str