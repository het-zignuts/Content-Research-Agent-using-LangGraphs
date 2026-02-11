from app.llms.groq import get_groq_llm

def extract_node(state):
    """
    Extraction Node for the LangGraph. This node takes the retrieved documents from the vector database and extracts key information based on the user query. It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. 
    The node then defines an extraction prompt that instructs the LLM to extract relevant information from the context in relation to the user query and generate a research report in markdown format. The LLM is invoked with this prompt, and the generated markdown report is returned in the state for downstream processing or response generation.
    """
    EXTRACTION_PROMPT="""
        SYSTEM INSTRUCTION:
            You are an info-extraction and report generation assistant.
            Extract the key information from the retrieved content provided as context below, based on the user query.
            Then, generate a research report in markdown format based on the extracted information.

            Return only the .md format report as a string, for example:
            report=f"# Research Report .... "

            No other text format should be returned other than the markdown report.

        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """

    llm=get_groq_llm(temperature=0.0) # initializing the llm
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"]) # prepare the context
    report=llm.invoke(
        EXTRACTION_PROMPT.format(context=context, query=state["query"]) # invoke the llm with the formatted prompt
    )
    return {"report_md": report.content} # add the generated markdown report to the state under the key "report_md", which can be used by downstream nodes in the graph to provide a response to the user or for further processing.