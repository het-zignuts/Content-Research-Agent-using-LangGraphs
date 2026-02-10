from typing import TypedDict, List, Dict, Optional

class GraphState(TypedDict):
    session_id: str
    query: str
    task: str
    documents: List[Dict]
    grouped_docs: Dict[str, List[Dict]]
    answer: Optional[str]
    report_md: Optional[str]
