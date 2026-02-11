from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState
from app.graph.nodes.retrieve import retrieve_node
from app.graph.nodes.tool_selector import tool_selector_node
from app.graph.nodes.qna import qna_node
from app.graph.nodes.compare import compare_node
from app.graph.nodes.insight import insight_node
from app.graph.nodes.summarize import summarize_node
from app.graph.nodes.extract import extract_node

def build_graph():
    """
    Build the LangGraph for the Content Research Agent. This graph defines the flow of operations based on the user query and the retrieved documents. It starts with the tool selector node to classify the user query into a specific task, followed by the retrieval node to fetch relevant documents from the vector database. Based on the classified task, it conditionally routes to one of the nodes: qna, compare, insight, summarize, or extract. Each of these nodes processes the retrieved documents according to their specific functionality and returns an answer or output that is then used to generate a response for the user. The graph is compiled and returned for execution.
    """
    graph=StateGraph(GraphState) # initialize the graph with the defined state structure

    # Add nodes
    graph.add_node("tool_selector", tool_selector_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("qna", qna_node)
    graph.add_node("compare", compare_node)
    graph.add_node("insight", insight_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("extract", extract_node)

    # Add edges
    graph.add_edge(START, "tool_selector")
    graph.add_edge("tool_selector", "retrieve")
    graph.add_conditional_edges("retrieve", lambda state: state["task"], { # addidng a conditional edge from the retrieve node to route to the appropriate node based on the classified task in the state
        "qna": "qna",
        "compare": "compare",
        "insight": "insight",
        "summarize": "summarize",
        "extract": "extract"
    })
    graph.add_edge("qna", END)
    graph.add_edge("compare", END)
    graph.add_edge("insight", END)
    graph.add_edge("summarize", END)
    graph.add_edge("extract", END)

    # compile the graph to optimize it for execution and return the compiled graph object that can be executed with a given initial state to process user queries and generate responses based on the defined flow and nodes in the graph.
    return graph.compile()