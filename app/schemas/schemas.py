from pydantic import BaseModel
from typing import Optional, Dict

class BaseSchema(BaseModel):
    answer: str

class ExtractionResponse(BaseSchema):
    report: Optional[str]

class SummarizationResponse(BaseSchema):
    pass

class ComparisonResponse(BaseSchema):
    pass

class QnAResponse(BaseSchema):
    pass

class InsightsResponse(BaseSchema):
    pass

class AgentResponse(BaseModel):
    status: int
    message: str
    data: Dict
