from pydantic import BaseModel
from typing import Dict
from app.schemas.schemas import *

def convert_schema(model: BaseModel)->Dict:
    output_schema={
        "response_format":{
            "type":"json_schema",
            "json_schema":{
                "name": model.__name__,
                "schema": model.model_json_schema()
            }
        }
    }
    return output_schema

ComparisonSchema=convert_schema(BaseSchema)
ExtractionSchema=convert_schema(ExtractionResponse)
InsightSchema=convert_schema(BaseSchema)
SummarizationSchema=convert_schema(BaseSchema)
QnASchema=convert_schema(BaseSchema)