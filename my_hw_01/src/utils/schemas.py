from pydantic import BaseModel, Field
from enum import Enum

class RequestType(str, Enum):
    QUESTION = "question"
    TASK = "task"
    SMALL_TALK = "small_talk"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"

class Classification(BaseModel):
    request_type: RequestType = Field(description="Тип запроса")
    confidence: float = Field(ge=0, le=1, description="Уверенность")
    reasoning: str = Field(description="Обоснование")

class AssistantResponse(BaseModel):
    content: str = Field(description="Ответ ассистента")
    request_type: RequestType = Field(description="Тип запроса")
    confidence: float = Field(ge=0, le=1, description="Уверенность")
    tokens_used: int = Field(ge=0, description="Использовано токенов")
