from collections import defaultdict
from app.db.vector_db import VectorDB

def retrieve_node(state):
    """
    Retrieval Node for the LangGraph. This node is responsible for retrieving relevant documents from the vector database based on the user query. 
    It initializes the VectorDB instance using the session ID from the state, loads the vector database, and performs a similarity search using the user query to retrieve the top 8 relevant documents. The retrieved documents are then grouped by their document ID for easier reference in downstream nodes. The node returns both the list of retrieved documents and the grouped documents in the state for further processing by subsequent nodes in the graph.
    """
    vector_db=VectorDB(session_id=state["session_id"]) # initialize the VectorDB instance using the session ID from the state to ensure that the retrieval is specific to the user's session and context.
    vector_db.load_db() # load the db
    store=vector_db.vector_db # get vectore store
    docs=store.similarity_search(state["query"], k=8) # perform similarity search

    grouped=defaultdict(list) # initialize a defaultdict to group the retrieved documents by their document ID, allowing for easy aggregation of content from the same document across different pages or sections.
    for d in docs:
        # group the docs by doc_id with separated metadata and content...
        grouped[d.metadata["doc_id"]].append({
            "content": d.page_content,
            "doc_name": d.metadata["doc_name"],
            "page_number": d.metadata.get("page", "N/A")
        })

    # return docs and grouped docs to state
    return {
        "documents": docs,
        "grouped_docs": grouped
    }
