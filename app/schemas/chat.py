from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1200)
    result: Optional[Dict[str, Any]] = None
    history: List[ChatHistoryItem] = Field(default_factory=list, max_length=8)


class ChatResponse(BaseModel):
    reply: str
    safety_level: str
