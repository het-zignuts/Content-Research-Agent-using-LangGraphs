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
    graph=StateGraph(GraphState)

    graph.add_node("tool_selector", tool_selector_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("qna", qna_node)
    graph.add_node("compare", compare_node)
    graph.add_node("insight", insight_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("extract", extract_node)

    graph.add_edge(START, "tool_selector")
    graph.add_edge("tool_selector", "retrieve")
    graph.add_edge("retrieve", lambda state: state["task"], {
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

    return graph.compile()