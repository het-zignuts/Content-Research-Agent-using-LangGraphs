from collections import defaultdict
from app.db.vector_db import VectorDB

def retrieve_node(state):
    vector_db=VectorDB(session_id=state["session_id"])
    vector_db.load_db()
    store=vector_db.vector_db
    docs=store.similarity_search(state["query"], k=8)

    grouped=defaultdict(list)
    for d in docs:
        grouped[d.metadata["doc_id"]].append({
            "content": d.page_content,
            "doc_name": d.metadata["doc_name"],
            "page_number": d.metadata.get("page", "N/A")
        })

    return {
        "documents": docs,
        "grouped_docs": grouped
    }
