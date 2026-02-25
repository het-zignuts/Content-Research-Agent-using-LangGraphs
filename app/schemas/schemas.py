from pydantic import BaseModel
from typing import Optional, Dict

class BaseSchema(BaseModel):
    answer: str

class ExtractionResponse(BaseSchema):
    report: Optional[str]

class AgentResponse(BaseModel):
    status: int
    message: str
    data: Dict