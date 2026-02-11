from typing import TypedDict, List, Dict, Optional

class GraphState(TypedDict):
    session_id: str
    query: str
    answer: Optional[str]
    task: str
    report_md: Optional[str]
    documents: List[Dict]
    grouped_docs: Dict[str, List[Dict]]
    
