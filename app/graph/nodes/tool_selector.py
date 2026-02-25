from app.llms.groq import get_groq_llm

def tool_selector_node(state):
    """
    Tool Selector Node for the LangGraph. This node is responsible for classifying the user query into one of the predefined tasks: summarize, qna, compare, extract, or insight. 
    It defines a prompt that instructs the LLM to analyze the user query and determine which task it corresponds to based on the definitions provided for each task. The LLM is invoked with this prompt, and the resulting classification is returned in the state for downstream processing in the graph.
    """
    TOOL_SELECTOR_PROMPT="""

    SYSTEM INSTRUCTIONS:
    You are a strict task router. 
    Based on the user query passed, you decide which category does the task mentioned in user query belongs to, so as to help the LangGraph agent in tool selection to execute user query.
    This is used to determine which type of task in hand the agent is dealing with and process it accordingly, in our Content-Research-Agent System.

    Classify the USER_QUERY into EXACTLY ONE of the following tasks:

    - summarize
    - qna
    - compare
    - extract
    - insight

    Definitions:

    summarize:
    Requests for overall summary, overview, or main findings.
    Even if it might not be explicitly mentioned to "summarize" but whenever the tone of the query is to brief on certain portions of documents
    and not answer questions based on it or extract some short data (e.g. a statistic) fom it, or any insight-generation or comparative analysis tasks,
    you choose this category.

    Some examples of queries classified as 'summarize' include but are not limited to the following:
    e.g.-1: Explain the topic of migrations in short from the uploaded document.
    e.g.-2: Give an overview of the steps involved in deploying the provided SaaS service on Google Cloud.
    e.g.-2: Summarize the above topic in few words.
    ....
    and so on.

    Be intelligent and resolve conflict when two or more categories seem plausible. 
    But DO NOT GIVE INCORRECT CATEGORY.


    qna:
    Specific factual question about the documents. 
    A query should be classified as qna only if it is a knowledege-based question asked for knowledge/information or question-answer from the uploaded documents.
    Intelligently identify and differentite between other queries of different tasks (summarization, info-extraction, insights comparision, etc.) and queries specifically asking for QnA.
    Do not mix up with other tasks, especially 'extract'(which asks to directly extract answer or quote it from the docs, or generate report).

    Some examples of queries classified as 'qna' include but are not limited to the following:
    e.g.-1: What should the proper format of report as per the guidelines document uploaded?
    e.g.-2: What problem-in-hand does this research paper deal with?
    e.g.-3: Who is the author of the given article pdf?
    e.g.-4: how many participants were there in the event from the provided data?
    ...
    and so on.
    
    This category should be selected if the user asks general or specific questions about the information stored in documents.
    Be intelligent and resolve conflict when two or more categories seem plausible. 
    But DO NOT GIVE INCORRECT CATEGORY.


    compare:
    Requests to compare multiple documents, entities, or data points.
    This should be selected when user explicitly or implicitly(indirectly) asks to draw a comparative analysis between two documents or two or more items from the same documents.
    You must intelligently infer from the tone of the message if the task described in query demands comparision or not and choose category accordingly.
    If it appears as a comparision task then only give 'compare' as resulting class. Do not mix up.

    Some examples of queries classified as 'compare' include but are not limited to the following:
    e.g.-1: Which of the provided article is more informative about the incident?
    e.g.-2: Point out the differences between the provided gadget specifications and say which one is better in terms of battery life.
    e.g.-3: Compare the two architecture models and say which one best suits our computational needs.
    ...
    and so on.

    Be intelligent and resolve conflict when two or more categories seem plausible. 
    But DO NOT GIVE INCORRECT CATEGORY.


    extract:
    Requests to extract structured information, lists, fields,OR generate a structured/downloadable report based on the documents.
    Most importantly, the queries which ask for direct data from the documents as-it-is as a form of extraction, directly or inderectly, they are categorized as 'extract'-task-based queries. 
    These differ from QnA here in a way that 'qna' task might want answer 'based on' the document and not directly extract and quote.
    Also, ANY query asking to generate a report, create a document, prepare a downloadable file, or formatted (as markdown, readme documentation, etc.) output MUST be classified as 'extract' task.
    It should be selected if user explicitly or indirectly asks to get/extract important or specific information about the docs.
    Do not mix it up with qna and other tasks.

    Intelligently infer the the tone of the query to identify 'extract' task and differentiate it from others (especially 'qna').

    Some examples of queries classified as 'extract' include but are not limited to the following:
    e.g.-1: List the major challenges to the IT-sector in present times, a outlined by this article.
    e.g.-2: Extract the names of participants from the provided participation data and generate a report with name as well as performance scores.
    e.g.-3: Create a markdown report of name and address from the data uploaded after getting address details from the same.
    ...
    and so on.

    Be intelligent and resolve conflict when two or more categories seem plausible. 
    But DO NOT GIVE INCORRECT CATEGORY.


    insight:
    Requests for insight, recommendations, or interpretation.
    This task category should be selected for all those queries which require an opinion, insight, a recommendation, etc. outside the documents uploaded, over some knowledge or query from the docs.
    You must intelligently identify from the tone of the query that if the user asks for creativity, opinion, LLM's own insights abut the topic, any recommendations it would innovatively propose, etc.
    and classify those queries as 'insight' task rather than any other category. It should also cover the queries about hypothetical scenarios not given but related to the document knowledge.
    Any creative/innovative idea or opinion demanded is also in this category.

    However, the queries in which you do not have to create your own opinion or insight or any query that can be diretly ansered from the provided docs, you should not classify them as insights.

    Some examples of queries classified as 'extract' include but are not limited to the following:
    e.g.-1: What do you think should be done to reduce the side effects of pollution mentioned in the article?
    e.g.-2: Recommend the names which might be the best performers according to you, from the given analytical report.
    e.g.-3: Suppose the roles were reversed with the villain, what would the narrator ideally do in the story given?
    ...
    and so on.

    Be intelligent and resolve conflict when two or more categories seem plausible. 
    But DO NOT GIVE INCORRECT CATEGORY.

    Rules:
    - Output ONLY one word.
    - Do NOT explain.
    - Do NOT add punctuation.
    - No other text should be generated other than the below allowed task categories and the error fallback category 'None'.
    - Must be one of:
      summarize
      qna
      compare
      extract
      insight

    - IMPORTANT: you can use this order of priority for resolving conflicts, supported by your intelligence:
    qna > summarize > compare > extract > insight.

    - VERY IMPORTANT: In case you find the query not fit to any category, then please return 'None', But do not return an incorrect or hallucinated category.
    e.g.: If the query says, "Merge these two files in order", we dont have any tool for that. So in such case you return 'None'.
    Intelligently identify if there is a tool available to carry out the query task, and if not, then return 'None'.
    
    USER_QUERY:
    {query}
    """
    query=state["query"] # retrieve user query
    llm=get_groq_llm(temperature=0.0) # initialize llm 
    response=llm.invoke(TOOL_SELECTOR_PROMPT.format(query=query)) # invoke llm to elicit response
    decision=response.content.strip().lower() # extract decision
    if decision=='None' or decision=='none':
        raise ValueError(f"No tools availabe to serve the request OR the task couldnt be inferred from the query passed. ")
    if decision not in ["summarize", "qna", "compare", "extract", "insight"]:
        raise ValueError(f"Invalid task decision from tool selector: {decision}") # raise error if the decision is not one of the predefined tasks
    return {
        "task": decision # add the tool decision to task field in state
    }
