from pydantic import BaseModel
from typing import Optional

class ExtractionResponse(BaseModel):
    answer: str
    report: Optional[str]