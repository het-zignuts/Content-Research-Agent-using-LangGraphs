from collections import defaultdict
from vectorstore.faiss_store import load_store

def retrieve_node(state):
    store=load_store(state["session_id"])
    docs=store.similarity_search(state["query"], k=8)

    grouped=defaultdict(list)
    for d in docs:
        grouped[d.metadata["doc_id"]].append({
            "content": d.page_content,
            "doc_name": d.metadata["doc_name"]
        })

    return {
        "documents": docs,
        "grouped_docs": grouped
    }
