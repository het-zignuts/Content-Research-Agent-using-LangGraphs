from app.llms.groq import get_groq_llm

def tool_selector_node(state):
    TOOL_SELECTOR_PROMPT="""
    SYSTEM_INSTRUCTION:
        You are a task router.

        Given a user query, classify it into ONE of these tasks:
        - summarize
        - qna
        - compare
        - extract
        - insight

        Return ONLY the task name as a string.
        e.g if the query is "Summarize the main findings from the documents", your response should be "summarize".

    USER_QUERY:
    {query}
"""
    query=state["query"]
    llm=get_groq_llm(temperature=0.0)
    response=llm.invoke(TOOL_SELECTOR_PROMPT.format(query=query)).strip().lower()
    if response not in ["summarize", "qna", "compare", "extract", "insight"]:
        response="qna"  
    return {
        "task": response
    }
