from app.llms.groq import get_groq_llm

def tool_selector_node(state):
    TOOL_SELECTOR_PROMPT="""
    You are a strict task router.

    Classify the USER_QUERY into EXACTLY ONE of the following tasks:

    - summarize
    - qna
    - compare
    - extract
    - insight

    Definitions:

    summarize:
    Requests for overall summary, overview, or main findings.

    qna:
    Specific factual question about the documents.

    compare:
    Requests to compare multiple documents, entities, or data points.

    extract:
    Requests to extract structured information, lists, fields,
    OR generate a structured/downloadable report based on the documents.
    ANY query asking to generate a report, create a document,
    prepare a downloadable file, or formatted output MUST be extract.

    insight:
    Requests for analysis, recommendations, or interpretation.

    Rules:
    - Output ONLY one word.
    - Do NOT explain.
    - Do NOT add punctuation.
    - Must be one of:
      summarize
      qna
      compare
      extract
      insight

    USER_QUERY:
    {query}
    """
    query=state["query"]
    llm=get_groq_llm(temperature=0.0)
    response=llm.invoke(TOOL_SELECTOR_PROMPT.format(query=query))
    decision=response.content.strip().lower()
    if decision not in ["summarize", "qna", "compare", "extract", "insight"]:
        raise ValueError(f"Invalid task decision from tool selector: {decision}") 
    return {
        "task": decision
    }
