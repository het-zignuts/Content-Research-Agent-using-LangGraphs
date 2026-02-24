from pydantic import BaseModel
from typing import Optional

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