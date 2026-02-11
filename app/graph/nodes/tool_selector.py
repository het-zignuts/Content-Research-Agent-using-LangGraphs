from app.llms.groq import get_groq_llm

def tool_selector_node(state):
    """
    Tool Selector Node for the LangGraph. This node is responsible for classifying the user query into one of the predefined tasks: summarize, qna, compare, extract, or insight. 
    It defines a prompt that instructs the LLM to analyze the user query and determine which task it corresponds to based on the definitions provided for each task. The LLM is invoked with this prompt, and the resulting classification is returned in the state for downstream processing in the graph.
    """
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
    query=state["query"] # retrieve user query
    llm=get_groq_llm(temperature=0.0) # initialize llm 
    response=llm.invoke(TOOL_SELECTOR_PROMPT.format(query=query)) # invoke llm to elicit response
    decision=response.content.strip().lower() # extract decision
    if decision not in ["summarize", "qna", "compare", "extract", "insight"]:
        raise ValueError(f"Invalid task decision from tool selector: {decision}") # raise error if the decision is not one of the predefined tasks
    return {
        "task": decision # add the tool decision to task field in state
    }
